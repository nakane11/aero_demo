#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""複数人の見張り: 全員の体の骨格を軽く取り、手を差し出していそうな人を選ぶ.

Holistic (``PeoplePoseEstimator``) は追跡モードで一度捕まえた人の周りしか
探さないので、ほかの人が手を出しても気が付かない。ここでは

1. ``PoseWatcher``: MediaPipe Tasks の PoseLandmarker (IMAGE モード、毎回検出
   し直す。見つけた人を塗って探し直す) で全員の体の骨格と bbox を取る。
2. ``PersonTracker``: 体幹の 3 次元位置 (深度があれば) か bbox の中心で、
   フレーム間の人に番号を振る。
3. ``offer_candidate_score``: 体の骨格だけで「手を出していそうか」を 0〜1 で
   付ける (手の点は使わない粗い値。確定は今までどおり Holistic と
   ``OfferedHandSelector`` で行う)。
4. ``TargetChooser``: 詳しく見る人を決める。ほかの人の候補スコアが今の人より
   明らかに高い状態が続けば切り替える。誰も手を出していなければ、正面の
   近くでこちらを向いた人 (``idle_cost``) を見る。

をまとめた ``MultiPersonWatcher`` を使う。Holistic にはほかの人を黒く塗った
画像全体を渡す (``PeoplePoseEstimator`` の ``mask``)。ROS には依存しない。
"""

import math

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

__all__ = ['PoseWatcher', 'PersonTracker', 'TargetChooser',
           'MultiPersonWatcher', 'offer_candidate_score', 'idle_cost']

# MediaPipe Pose の番号 (人物自身の左右)。
POSE_INDEX = dict(
    Nose=0, LShoulder=11, RShoulder=12, LElbow=13, RElbow=14,
    LWrist=15, RWrist=16, LPinky=17, RPinky=18, LIndex=19, RIndex=20,
    LHip=23, RHip=24, LKnee=25, RKnee=26, LAnkle=27, RAnkle=28)
TORSO_JOINTS = ('LShoulder', 'RShoulder', 'LHip', 'RHip')


def box_overlap(a, b):
    """``a`` と ``b`` の重なりの面積 / 小さい方の面積."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(iw * ih / max(area, 1e-6))


def _ramp(value, low, high):
    if value is None:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


class PoseWatcher(object):
    """PoseLandmarker で画像中の全員の体の骨格を取る (状態を持たない).

    PoseLandmarker は ``num_poses`` を増やしても画像全体ではほぼ 1 人しか
    返さない (11/23 に実機の合成画像・CMU Panoptic で確認)。そこで見つけた人の
    bbox を黒く塗って探し直すのを、新しい人が出なくなるまで (最大
    ``max_people`` 人) 繰り返す。1 回 30〜40 ms (960x540、full モデル)。
    """

    def __init__(self, model_path, num_poses=2, max_people=4,
                 min_detection_confidence=0.5, min_presence_confidence=0.5,
                 min_visibility=0.5, min_torso_joints=3,
                 mask_pad=(0.3, 0.15), max_overlap=0.4):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence)
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self.max_people = max_people
        self.min_visibility = min_visibility
        self.min_torso_joints = min_torso_joints
        self.mask_pad = mask_pad
        self.max_overlap = max_overlap

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    def detect(self, bgr_img):
        """人ごとに ``dict(box=[x0, y0, x1, y1], pixels={名前: [u, v, vis]})``."""
        h, w = bgr_img.shape[:2]
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        people = []
        for _ in range(self.max_people + 1):
            found = self._detect_once(rgb, w, h)
            if not found:
                break
            for person in found:
                # 塗り残し (腕・脚) から同じ人がもう一度出たものは数えない。
                if (len(people) < self.max_people
                        and all(box_overlap(person['box'], q['box'])
                                <= self.max_overlap for q in people)):
                    people.append(person)
                x0, y0, x1, y1 = person['box']
                px = self.mask_pad[0] * (x1 - x0)
                py = self.mask_pad[1] * (y1 - y0)
                rgb[int(max(0, y0 - py)):int(min(h, y1 + py)),
                    int(max(0, x0 - px)):int(min(w, x1 + px))] = 0
        return people

    def _detect_once(self, rgb, w, h):
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect(image)
        people = []
        for landmarks in result.pose_landmarks:
            pixels = {}
            for name, idx in POSE_INDEX.items():
                lm = landmarks[idx]
                pixels[name] = [lm.x * w, lm.y * h, float(lm.visibility)]
            seen = [p for p in pixels.values() if p[2] > self.min_visibility]
            n_torso = sum(pixels[n][2] > self.min_visibility
                          for n in TORSO_JOINTS)
            if n_torso < self.min_torso_joints or len(seen) < 4:
                continue
            us = [p[0] for p in seen]
            vs = [p[1] for p in seen]
            people.append(dict(
                box=[float(max(0, min(us))), float(max(0, min(vs))),
                     float(min(w, max(us))), float(min(h, max(vs)))],
                pixels=pixels))
        return people


def lift_to_3d(pixels, depth_img, intrinsics, transform=None,
               min_visibility=0.5, patch=5):
    """``pixels`` の点を深度で 3 次元にし ``transform`` (4x4) の座標系で返す."""
    if depth_img is None or intrinsics is None:
        return {}
    half = patch // 2
    out = {}
    for name, (u, v, vis) in pixels.items():
        if vis <= min_visibility:
            continue
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < depth_img.shape[0] and 0 <= ui < depth_img.shape[1]):
            continue
        region = depth_img[max(0, vi - half):vi + half + 1,
                           max(0, ui - half):ui + half + 1]
        valid = region[np.isfinite(region) & (region > 0)]
        if valid.size == 0:
            continue
        z = float(np.median(valid))
        point = np.array([(u - intrinsics.cx) * z / intrinsics.fx,
                          (v - intrinsics.cy) * z / intrinsics.fy, z, 1.0])
        if transform is not None:
            point = np.asarray(transform, dtype=np.float64).dot(point)
        out[name] = point[:3]
    return out


def plausible_body(pixels, points_camera, intrinsics,
                   shoulder_width=(0.15, 0.7), torso_length=(0.25, 0.9),
                   min_visibility=0.5):
    """肩幅・肩→腰の長さ [m] が人らしいか (測れなければ True).

    PoseLandmarker は椅子やロボットにも骨格を付けることがある (11/23 の実機の
    録画で PR2・椅子)。長さは画像上の長さを体幹 4 点の深度の下側の中央値
    (小さい方から 2 番目) で m にする。肩・腰は体の縁で深度が背景に抜け
    やすく (横向きの人は 4 点中 2 点が抜けた)、背景は人より奥にあるため。
    """
    torso_z = sorted(points_camera[n][2] for n in TORSO_JOINTS
                     if n in points_camera)
    if len(torso_z) < 2 or intrinsics is None:
        return True
    z = float(torso_z[(len(torso_z) - 1) // 2])
    f = 0.5 * (intrinsics.fx + intrinsics.fy)

    def length(a, b):
        pa, pb = pixels.get(a), pixels.get(b)
        if pa is None or pb is None or min(pa[2], pb[2]) <= min_visibility:
            return None
        return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) * z / f
    width = length('LShoulder', 'RShoulder')
    # 横向きの人は肩幅が狭く見えるので下限は見ない。
    if width is not None and width > shoulder_width[1]:
        return False
    lengths = [v for v in (length('LShoulder', 'LHip'),
                           length('RShoulder', 'RHip')) if v is not None]
    if lengths and not (torso_length[0] <= max(lengths) <= torso_length[1]):
        return False
    return True


def joints_box(joints_2d, min_score=0.5):
    """``PeoplePoseEstimator`` の 2D 関節 (``x``/``y``/``score``) の bbox."""
    pts = [(j['x'], j['y']) for j in joints_2d or ()
           if j.get('score', 0.0) > min_score]
    if not pts:
        return None
    us, vs = zip(*pts)
    return [float(min(us)), float(min(vs)), float(max(us)), float(max(vs))]


HEIGHT_JOINTS = ('Nose', 'LShoulder', 'RShoulder')


def upper_body_height(points):
    """鼻・両肩のうち深度で測れた点の高さ [m] の最大 (床が z=0 の座標系).

    測れた点が無ければ None。11/24 の実機の録画では、立った人は肩 1.13〜
    1.22 m、椅子などの誤検出は肩 0.62 m 以下か深度が無かった。
    """
    heights = [float(points[n][2]) for n in HEIGHT_JOINTS if n in points]
    return max(heights) if heights else None


def torso_center(points):
    torso = [points[n] for n in TORSO_JOINTS if n in points]
    if len(torso) < 2:
        return None
    return np.mean(np.asarray(torso), axis=0)


IDLE_BEARING_WEIGHT = 0.5  # [m/rad] ロボットの正面からの方位の重み
IDLE_FACING_WEIGHT = 0.5  # [m] 背を向けた人に足す量 (横向きはその半分)


def idle_cost(points, up=(0.0, 0.0, 1.0)):
    """誰も手を出していないときに見る人を決める値 [m 相当] (小さいほど優先).

    ``points`` は base_link 系 (x 前・y 左・z 上) の骨格。ロボットからの水平
    距離に、正面からの方位 (``IDLE_BEARING_WEIGHT`` 倍) と、体がロボットを
    向いていない度合い (肩の線の向きから、``IDLE_FACING_WEIGHT`` 倍) を足す。
    部屋の入口に人が集まったとき、正面の近くでこちらを向いた人を見るため。
    体幹が測れなければ None。
    """
    center = torso_center(points)
    if center is None:
        return None
    up = np.asarray(up, dtype=np.float64)
    c = center - center.dot(up) * up
    distance = float(np.linalg.norm(c))
    if distance < 1e-6:
        return 0.0
    bearing = abs(math.atan2(float(c[1]), float(c[0])))
    facing = 0.0  # 分からなければ横向きと同じ扱い
    if 'LShoulder' in points and 'RShoulder' in points:
        s = (np.asarray(points['LShoulder'], dtype=np.float64)
             - np.asarray(points['RShoulder'], dtype=np.float64))
        s = s - s.dot(up) * up
        # 人物自身の左肩 - 右肩と上向きの外積が体の正面。
        front = np.cross(s, up)
        norm = float(np.linalg.norm(front))
        if norm > 1e-6:
            facing = float(np.dot(front / norm, -c / distance))
    return (distance + IDLE_BEARING_WEIGHT * bearing
            + IDLE_FACING_WEIGHT * 0.5 * (1.0 - facing))


def offer_candidate_score(points, robot_position, up=(0.0, 0.0, 1.0),
                          max_distance=3.0):
    """体の骨格 (3 次元、z 上向きの座標系) だけで差し出していそうかを付ける.

    左右それぞれ、手 (手首・人差し指・小指の付け根の平均) が
      ``reach``: 肩からロボットの方へ水平にどれだけ出ているか (0.10→0.30 m)
      ``height``: 腰からどれだけ上にあるか (-0.10→0.10 m)
    を 0〜1 にし、0.6 / 0.4 で足す。返り値は ``(score, side, 内訳)``。
    """
    up = np.asarray(up, dtype=np.float64)
    hip = [points[n] for n in ('LHip', 'RHip') if n in points]
    shoulders = [points[n] for n in ('LShoulder', 'RShoulder') if n in points]
    if not hip and not shoulders:
        return 0.0, None, None
    if hip:
        hip_c = np.mean(hip, axis=0)
    else:
        # 腰が見えなければ肩から 0.5 m 下を腰とみなす。
        hip_c = np.mean(shoulders, axis=0) - 0.5 * up
    robot = np.asarray(robot_position, dtype=np.float64)
    to_robot = robot - hip_c
    to_robot = to_robot - to_robot.dot(up) * up
    distance = float(np.linalg.norm(to_robot))
    if distance < 1e-6 or (max_distance is not None
                           and distance > max_distance):
        return 0.0, None, dict(distance=distance, too_far=True)
    direction = to_robot / distance
    best = (0.0, None)
    details = dict(distance=distance)
    for side in ('R', 'L'):
        shoulder = points.get(side + 'Shoulder')
        hand_pts = [points[side + n] for n in ('Wrist', 'Index', 'Pinky')
                    if side + n in points]
        if shoulder is None or not hand_pts:
            details[side] = None
            continue
        hand = np.mean(hand_pts, axis=0)
        offset = hand - shoulder
        reach = float((offset - offset.dot(up) * up).dot(direction))
        height = float((hand - hip_c).dot(up))
        score = 0.6 * _ramp(reach, 0.10, 0.30) + 0.4 * _ramp(height,
                                                             -0.10, 0.10)
        details[side] = dict(reach=reach, height=height, score=score)
        if score > best[0]:
            best = (score, side)
    return best[0], best[1], details


class _Track(object):
    def __init__(self, track_id, person, t):
        self.id = track_id
        self.first_seen = t
        self.score_ema = None
        self.hits = 0
        self.update(person, t)

    def update(self, person, t):
        self.hits += 1
        self.person = person
        self.last_seen = t
        self.box = person['box']
        self.center = person.get('center')


class PersonTracker(object):
    """体幹の位置 (無ければ bbox の中心) の最近傍で人に番号を振る."""

    def __init__(self, max_dist_3d=0.6, max_dist_2d=0.15, lost_timeout=1.0):
        self.max_dist_3d = max_dist_3d
        self.max_dist_2d = max_dist_2d  # 画像の幅に対する割合
        self.lost_timeout = lost_timeout
        self.tracks = {}
        self._next_id = 0

    def reset(self):
        self.tracks = {}

    @staticmethod
    def _box_center(box):
        return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])

    def _cost(self, track, person, image_width):
        """3 次元 (あれば) と画像上の距離の、閾値で割った値の小さい方.

        体幹の深度が背景に抜けると 3 次元の位置が飛ぶので、画像上でも見る。
        """
        d2 = float(np.linalg.norm(self._box_center(track.box)
                                  - self._box_center(person['box'])))
        cost = d2 / (self.max_dist_2d * image_width)
        if track.center is not None and person.get('center') is not None:
            d3 = float(np.linalg.norm(track.center - person['center']))
            cost = min(cost, d3 / self.max_dist_3d)
        return cost

    def update(self, people, t, image_width):
        """``people`` に ``track_id`` を付けて、今見えている ``_Track`` を返す."""
        for track_id in [k for k, tr in self.tracks.items()
                         if t - tr.last_seen > self.lost_timeout]:
            del self.tracks[track_id]
        pairs = []
        for i, person in enumerate(people):
            for track_id, track in self.tracks.items():
                cost = self._cost(track, person, image_width)
                if cost <= 1.0:
                    pairs.append((cost, i, track_id))
        pairs.sort()
        used_people, used_tracks = set(), set()
        for _cost, i, track_id in pairs:
            if i in used_people or track_id in used_tracks:
                continue
            used_people.add(i)
            used_tracks.add(track_id)
            self.tracks[track_id].update(people[i], t)
            people[i]['track_id'] = track_id
        for i, person in enumerate(people):
            if i in used_people:
                continue
            track = _Track(self._next_id, person, t)
            self._next_id += 1
            self.tracks[track.id] = track
            person['track_id'] = track.id
        return [self.tracks[p['track_id']] for p in people]


class TargetChooser(object):
    """詳しく見る人を決める.

    - 対象がいない・見失った: 誰かのスコアが ``idle_score`` 以上ならスコアが
      最も高い人 (同点なら近い人)。誰も手を出していなければ ``idle_cost``
      (近さ・正面からの方位・こちらを向いているか) が最も小さい人。
    - ほかの人のスコア (指数移動平均) が ``switch_score`` 以上で、今の人より
      ``switch_margin`` 以上高い状態が ``switch_hold`` 秒続いたら切り替える。
    - 今の人もほかの人もスコアが ``idle_score`` 未満のときは、``idle_cost``
      が今の人より ``idle_margin`` 以上小さい人がいる状態が ``idle_hold``
      秒続いたら切り替える (正面に近づいてきた人へ移る)。
    - ``lock()`` 中 (差し出しを確定した後など) は切り替えない。
    - ``hold(until)`` の時刻までは、今の人を見失っても選び直さない (首を
      よそへ向けて見渡している間)。手を出した人への切り替えは行う。
    - 切り替え先は ``min_hits`` 回以上見えた人だけ (一瞬の誤検出へ移らない)。
    """

    def __init__(self, switch_score=0.5, switch_margin=0.25, switch_hold=0.1,
                 lost_timeout=0.7, ema_tau=0.15, min_hits=3, idle_score=0.3,
                 idle_margin=0.3, idle_hold=1.0):
        self.min_hits = min_hits
        self.switch_score = switch_score
        self.switch_margin = switch_margin
        self.switch_hold = switch_hold
        self.lost_timeout = lost_timeout
        self.ema_tau = ema_tau
        self.idle_score = idle_score
        self.idle_margin = idle_margin
        self.idle_hold = idle_hold
        self.target_id = None
        self.locked = False
        self._challenger = None  # (track_id, 越え始めた時刻, 続ける秒数)
        self._last_t = None
        self._hold_until = None
        # 今の人に決めた時刻と、最後に手を出しかけていた (スコアが
        # idle_score 以上の) 時刻。``quiet_for`` で使う。
        self._target_since = None
        self._target_active_t = None

    def reset(self):
        self.target_id = None
        self.locked = False
        self._challenger = None
        self._hold_until = None
        self._target_since = None
        self._target_active_t = None

    def lock(self, locked=True):
        self.locked = bool(locked)
        self._challenger = None

    def hold(self, until):
        """``until`` (画像の時刻) までは今の人を見失っても選び直さない."""
        self._hold_until = until

    def quiet_for(self, t):
        """今の人が手を出しかけずにいる秒数 (人がいなければ 0)."""
        if self.target_id is None or self._target_since is None:
            return 0.0
        start = self._target_since
        if self._target_active_t is not None:
            start = max(start, self._target_active_t)
        return max(0.0, t - start)

    def update(self, tracks, all_tracks, t):
        """今の対象の番号と、切り替えたかどうかを返す."""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        alpha = 1.0 if dt <= 0 else 1.0 - math.exp(-dt / self.ema_tau)
        for track in tracks:
            raw = track.person.get('score', 0.0)
            track.score_ema = (raw if track.score_ema is None
                               else track.score_ema + alpha
                               * (raw - track.score_ema))
        visible = {tr.id: tr for tr in tracks
                   if tr.hits >= self.min_hits}
        previous = self.target_id
        target = all_tracks.get(self.target_id)
        target_lost = (target is None
                       or t - target.last_seen > self.lost_timeout)
        holding = (self.target_id is not None and self._hold_until is not None
                   and t < self._hold_until)
        if target_lost and not (self.locked and target is not None):
            if holding:
                # 見渡している間は選び直さない。ただし手を出した人には移る。
                self._consider_switch(target, visible, t, offers_only=True)
            else:
                self.target_id = self._best(list(visible.values()) or tracks)
                self._challenger = None
        elif not self.locked and target is not None:
            self._consider_switch(target, visible, t, offers_only=holding)
        if self.target_id != previous:
            self._target_since = t
            self._target_active_t = None
        current = all_tracks.get(self.target_id)
        if (current is not None and current.score_ema is not None
                and current.score_ema >= self.idle_score):
            self._target_active_t = t
        # 初めて決めたときは捨てる履歴がないので「切り替え」にしない
        # (Holistic を作り直すと 1 人のときも今の方式と結果が変わる)。
        switched = (previous is not None and self.target_id is not None
                    and self.target_id != previous)
        return self.target_id, switched

    @staticmethod
    def _key(track):
        distance = track.person.get('distance')
        return (track.score_ema or 0.0,
                -(distance if distance is not None else 1e9))

    @staticmethod
    def _cost(track):
        cost = track.person.get('idle_cost')
        return float('inf') if cost is None else cost

    def _best(self, tracks):
        if not tracks:
            return None
        if all((tr.score_ema or 0.0) < self.idle_score for tr in tracks):
            # 誰も手を出していない: 正面の近くでこちらを向いた人。
            nearest = min(tracks, key=self._cost)
            if self._cost(nearest) < float('inf'):
                return nearest.id
        return max(tracks, key=self._key).id

    def _consider_switch(self, target, visible, t, offers_only=False):
        own = 0.0 if target is None else (target.score_ema or 0.0)
        target_id = self.target_id
        others = [tr for tid, tr in visible.items() if tid != target_id]
        rival = max(others, key=self._key) if others else None
        if (rival is not None and (rival.score_ema or 0.0) >= self.switch_score
                and (rival.score_ema or 0.0) - own >= self.switch_margin):
            self._challenge(rival.id, t, self.switch_hold)
            return
        if (offers_only or target is None or not others
                or own >= self.idle_score
                or any((tr.score_ema or 0.0) >= self.idle_score
                       for tr in others)):
            self._challenger = None
            return
        # 誰も手を出していない: 正面に近い人がはっきりいれば移る。
        nearest = min(others, key=self._cost)
        if self._cost(target) - self._cost(nearest) >= self.idle_margin:
            self._challenge(nearest.id, t, self.idle_hold)
        else:
            self._challenger = None

    def _challenge(self, track_id, t, hold):
        if self._challenger is None or self._challenger[0] != track_id:
            self._challenger = (track_id, t, hold)
            return
        if t - self._challenger[1] >= self._challenger[2]:
            self.target_id = track_id
            self._challenger = None


class MultiPersonWatcher(object):
    """見張り一式. ``update`` が詳しく見る人以外を塗る範囲 (``view``) を返す.

    Parameters
    ----------
    interval : float
        PoseLandmarker を回す間隔 [s] (0 なら毎フレーム)。間は前回の範囲を使う。
    mask_pad, keep_pad : (float, float)
        ほかの人の bbox (塗る) と対象の bbox (塗らない) の幅・高さに足す割合。
        見張りの結果は 0.1〜0.2 秒遅れるので、塗る方は動いても隠れるよう広め。
    mask_keep_time : float
        見えなくなったほかの人をこの秒数は塗り続ける。
    min_body_height : float
        鼻・肩の高さ (床から、``camera_to_base`` の z) がこれ未満の骨格は人と
        みなさない (椅子など)。座った大人の肩はおよそ 0.9 m 以上。
    not_person_time, suppress_time : float
        Holistic の骨格が人らしくない (``min_body_height`` 未満) 状態が
        ``not_person_time`` 秒続いたら、その範囲を ``suppress_time`` 秒塗って
        Holistic を作り直させる (``check_focus``)。

    Holistic には切り出しではなく、ほかの人を黒く塗った画像全体を渡す
    (切り出す範囲が動くと Holistic の追跡がずれて骨格が崩れるため。
    11/23 に実機の録画で、範囲の飛びで誤って差し出しが確定するのを確認)。

    条件付きの見渡し: ``glance_enabled`` の間、今の人が ``glance_quiet`` 秒
    手を出しかけずにいたら ``glance_request`` を立てる。首を動かす側が
    ``take_glance_request`` で受け取って見渡し、``hold_for`` で見渡している
    間は今の人を選び直さないようにし、終わったら ``glance_done`` を呼ぶ。
    前の見渡しが終わってから ``glance_interval`` 秒は次を頼まない。
    """

    def __init__(self, model_path, num_poses=2, interval=0.0,
                 mask_pad=(0.35, 0.15), keep_pad=(0.25, 0.1),
                 mask_keep_time=0.5, max_distance=3.0, tracker=None,
                 chooser=None, same_person_overlap=0.6, min_body_height=0.8,
                 not_person_time=0.5, suppress_time=5.0,
                 suppress_pad=(0.2, 0.2), glance_quiet=8.0,
                 glance_interval=6.0):
        self.glance_enabled = False
        self.glance_quiet = glance_quiet
        self.glance_interval = glance_interval
        self.glance_request = None
        self._glance_active = False
        self._last_glance_t = None
        self.watcher = PoseWatcher(model_path, num_poses=num_poses)
        self.min_body_height = min_body_height
        self.not_person_time = not_person_time
        self.suppress_time = suppress_time
        self.suppress_pad = suppress_pad
        self._suppressed = []  # [(bbox, 塗るのをやめる時刻)]
        self._not_person_since = None
        self.tracker = tracker or PersonTracker()
        self.chooser = chooser or TargetChooser()
        self.interval = interval
        self.mask_pad = mask_pad
        self.keep_pad = keep_pad
        self.mask_keep_time = mask_keep_time
        self.max_distance = max_distance
        self.same_person_overlap = same_person_overlap
        self.view = None
        self.tracks = []
        self._last_watch = None
        self._last_target_box = None

    def close(self):
        self.watcher.close()

    def reset(self):
        self.tracker.reset()
        self.chooser.reset()
        self.view = None
        self.tracks = []
        self._last_watch = None
        self._last_target_box = None
        self._suppressed = []
        self._not_person_since = None
        self.glance_request = None
        self._glance_active = False
        self._last_glance_t = None

    @property
    def target_id(self):
        return self.chooser.target_id

    def take_glance_request(self):
        """立っている見渡しの依頼を取り出す (無ければ None)."""
        request, self.glance_request = self.glance_request, None
        return request

    def hold_for(self, seconds):
        """最後の画像の時刻から ``seconds`` 秒は今の人を選び直さない."""
        if self.chooser._last_t is not None:
            self.chooser.hold(self.chooser._last_t + seconds)

    def glance_done(self):
        """見渡しが終わった (やめた)。次はここから ``glance_interval`` 秒後."""
        self._glance_active = False
        self._last_glance_t = self.chooser._last_t

    def _update_glance(self, t):
        """条件を満たしたら ``glance_request`` を立てる (``apply`` から)."""
        if not self.glance_enabled or self.chooser.locked:
            self.glance_request = None
            self._glance_active = False
            return
        if (self._glance_active or self.glance_request is not None
                or self.chooser.quiet_for(t) < self.glance_quiet
                or (self._last_glance_t is not None
                    and t - self._last_glance_t < self.glance_interval)):
            return
        self._glance_active = True
        self.glance_request = dict(t=t, target_id=self.chooser.target_id)

    def lock(self, locked=True):
        self.chooser.lock(locked)

    def update(self, bgr_img, t, depth_img=None, intrinsics=None,
               camera_to_base=None, base_to_world=None, robot_position=None,
               up=(0.0, 0.0, 1.0)):
        """1 フレーム分を処理する.

        ``camera_to_base`` は骨格を z 上向きの座標系に移す 4x4、``base_to_world``
        は番号付けに使う静止座標系 (odom など) への 4x4。``robot_position`` は
        ``camera_to_base`` の座標系のロボット手先。

        Returns
        -------
        (view, switched)
            ``apply`` と同じ。``switched`` は詳しく見る人が変わったか
            (Holistic を作り直す合図)。
        """
        if (self._last_watch is not None and self.interval > 0
                and t - self._last_watch < self.interval):
            return self.view, False
        self._last_watch = t
        people = self.detect(bgr_img, depth_img, intrinsics, camera_to_base,
                             base_to_world, robot_position, up)
        return self.apply(people, t, bgr_img.shape[1], bgr_img.shape[0])

    def detect(self, bgr_img, depth_img=None, intrinsics=None,
               camera_to_base=None, base_to_world=None, robot_position=None,
               up=(0.0, 0.0, 1.0)):
        """重い方 (PoseLandmarker と 3 次元化・候補スコア)。別スレッドで回せる."""
        people = []
        for person in self.watcher.detect(bgr_img):
            points_camera = lift_to_3d(person['pixels'], depth_img,
                                       intrinsics)
            if not plausible_body(person['pixels'], points_camera, intrinsics):
                continue
            points = (points_camera if camera_to_base is None else {
                name: np.asarray(camera_to_base, dtype=np.float64).dot(
                    np.append(p, 1.0))[:3]
                for name, p in points_camera.items()})
            if camera_to_base is not None and depth_img is not None:
                # 鼻・肩が床から低い、または深度が無い骨格は人とみなさない。
                height = upper_body_height(points)
                if height is None or height < self.min_body_height:
                    continue
            people.append(person)
            person['points'] = points
            center = torso_center(points)
            if center is not None and base_to_world is not None:
                center = np.asarray(base_to_world, dtype=np.float64).dot(
                    np.append(center, 1.0))[:3]
            person['center'] = center
            score, side, details = (0.0, None, None)
            if robot_position is not None and points:
                score, side, details = offer_candidate_score(
                    points, robot_position, up=up,
                    max_distance=self.max_distance)
            person.update(score=score, side=side, details=details,
                          distance=(details or {}).get('distance'),
                          idle_cost=(idle_cost(points, up)
                                     if camera_to_base is not None and points
                                     else None))
        return people

    def apply(self, people, t, w, h):
        """軽い方 (番号付け・対象の切り替え・塗る範囲)。``t`` は画像の時刻.

        Returns
        -------
        (view, switched)
            ``view`` は ``PeoplePoseEstimator`` の ``mask`` に渡す
            ``dict(keep=対象の bbox, boxes=[ほかの人の bbox])`` (対象がいなければ
            None)。ほかの人がいなければ ``boxes`` は空で、画像はそのまま使われる。
        """
        self.tracks = self.tracker.update(people, t, w)
        previous = self.tracker.tracks.get(self.chooser.target_id)
        previous_box = self._last_target_box
        target_id, switched = self.chooser.update(
            self.tracks, self.tracker.tracks, t)
        self._update_glance(t)
        target = self.tracker.tracks.get(target_id)
        if (switched and target is not None and previous_box is not None
                and (previous is None or previous.last_seen < t)
                and box_overlap(previous_box, target.box)
                >= self.same_person_overlap):
            # 番号が付け直されただけ (前の対象を見失った所に重なって出た)。
            # Holistic や静止判定を作り直すと本命の確定が遅れるので続ける。
            switched = False
        if switched and sum(t - tr.last_seen <= self.mask_keep_time
                            for tr in self.tracker.tracks.values()) <= 1:
            # 見えているのが 1 人だけなら Holistic は作り直さなくてもその人を
            # 見る。見失った同じ人が離れた所で見つかり直したときに作り直して、
            # 本命の確定を遅らせない (11/24 の実機で 1 人なのに作り直した)。
            switched = False
        self._last_target_box = None if target is None else list(target.box)
        suppressed = self._suppressed_boxes(t)
        if target is None:
            self.view = (None if not suppressed
                         else dict(keep=None, boxes=suppressed))
            return self.view, switched
        # 塗るのは続けて見えた人だけ。塗ると重ならない所でも Holistic の出力が
        # 少し変わるので、一瞬の誤検出 (椅子など) は塗らない (1 人のときに
        # 今の方式と結果を変えない)。
        others = [tr for tr in self.tracker.tracks.values()
                  if tr.id != target.id
                  and tr.hits >= self.chooser.min_hits
                  and t - tr.last_seen <= self.mask_keep_time]
        self.view = dict(
            keep=self._padded(target.box, w, h, self.keep_pad),
            boxes=[self._padded(tr.box, w, h, self.mask_pad)
                   for tr in others] + suppressed)
        return self.view, switched

    def looks_like_person(self, points):
        """鼻・肩の高さが ``min_body_height`` 以上か (測れなければ True)."""
        height = None if not points else upper_body_height(points)
        return height is None or height >= self.min_body_height

    def check_focus(self, points, box, t, w, h):
        """Holistic が人でないもの (椅子など) を見続けていないかを確かめる.

        ``points`` は Holistic の骨格 (床が z=0 の座標系、無ければ None)、
        ``box`` はその画像上の bbox。鼻・肩が ``min_body_height`` 未満の状態が
        ``not_person_time`` 秒続いたら、その範囲を ``suppress_time`` 秒塗る
        範囲に足して True (Holistic を作り直す合図) を返す。高さを測れない
        フレームは数えない (近づいた人の肩が画面から出たときなど)。
        """
        height = None if not points else upper_body_height(points)
        if box is None or height is None:
            return False
        if height >= self.min_body_height:
            self._not_person_since = None
            return False
        if self._not_person_since is None or t < self._not_person_since:
            self._not_person_since = t
        if t - self._not_person_since < self.not_person_time:
            return False
        self._not_person_since = None
        padded = self._padded(box, w, h, self.suppress_pad)
        self._suppressed.append((padded, t + self.suppress_time))
        # 次の見張りの結果を待たずに、次のフレームから塗る。
        view = self.view or dict(keep=None, boxes=[])
        self.view = dict(view, boxes=list(view['boxes']) + [padded])
        return True

    def _suppressed_boxes(self, t):
        self._suppressed = [(box, until) for box, until in self._suppressed
                            if t <= until]
        return [box for box, _ in self._suppressed]

    @staticmethod
    def _padded(box, w, h, pad):
        x0, y0, x1, y1 = box
        px, py = pad[0] * (x1 - x0), pad[1] * (y1 - y0)
        return (max(0.0, x0 - px), max(0.0, y0 - py),
                min(float(w), x1 + px), min(float(h), y1 + py))

    def summary(self):
        """ログ・表示用の人ごとの内訳."""
        return [dict(track_id=tr.id, box=[float(v) for v in tr.box],
                     score=float(tr.person.get('score', 0.0)),
                     score_ema=(None if tr.score_ema is None
                                else float(tr.score_ema)),
                     side=tr.person.get('side'),
                     distance=tr.person.get('distance'),
                     idle_cost=tr.person.get('idle_cost'),
                     target=(tr.id == self.chooser.target_id))
                for tr in self.tracks]
