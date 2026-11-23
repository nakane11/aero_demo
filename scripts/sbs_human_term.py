# -*- coding:utf-8 -*-
# 横並び移動 (side_by_side_transition) の人の楽さの指標に使う人の項の式。
# 式は develop の scripts/goal_ik/human_palm.py と同じ (変えるときは合わせる)。
"""横並び後・寄せ後の人の掌の変数と、人の項 (楽さ) の誤差項。

人の掌の位置 3・回転 3 (回転ベクトル、初期の掌の向きに左から掛ける) を PoseGoalSolver の変数にし、
ロボットの手先の目標を掌の変数から作る (``press_target`` と同じ関係 = 掌の座標系に固定した剛体の変換)。
掌の変数を使う段階 (横並び後・寄せ後) は ``palm_active`` = 1、他の段階は 0 で掌を初期値に留める。

人の項は ``hand_comfort_jax`` の ``errors`` と可操作度のヒンジ:
  [モーメント, LUBA 7 成分 (肩の屈曲・外転は負の側だけ), 届かない量, 上腕の挙上, 可操作度のヒンジ] の 11 成分。
重みは全部 1、ε は ``hand_comfort_jax`` の既定。手首は既定 (掌から 4.1cm)。

座標: 人の肩を原点、右手の正準座標 (x = 前、y = 左、z = 上、差し出す腕の外向き = -y)。左手は y を反転して写す
(``canonical_basis``)。位置は人ごとの腕の長さ (上腕 + 前腕) で割る。

``xp`` は jax.numpy (このモジュールは jax を import しない)。
"""

import math

import numpy as np

# --- hand_comfort.py / hand_comfort_jax.py と同じ定数 (変えたら両方を変える) ---
# 標準の腕 (合成データ 67 人の中央値)。人の項は掌の位置を人ごとの腕の長さで割り、この腕に写して測る。
UPPER = 0.236
FORE = 0.240
LENGTH = UPPER + FORE
# 押し込み時の手首の、掌の座標系 (指先・法線・指先 x 法線) での位置 [m] (既定の手首)。
WRIST_IN_PALM = np.array([-0.041, 0.0, 0.0])
# Winter の標準値 (質量比、重心の近位からの割合)。
MASS = (('upper', 0.028), ('fore', 0.016), ('hand', 0.006))
COM = dict(upper=0.43, fore=0.43, hand=0.50)
JOINTS = ('shoulder_flexion', 'shoulder_abduction', 'shoulder_rotation',
          'elbow_flexion', 'pronation', 'wrist_flexion', 'wrist_deviation')
ROTATION_WEIGHT_DEG = (10.0, 30.0)
MANIP_THETA = np.radians(np.arange(0.0, 180.01, 0.5))
MANIP_AXIS_EPS = 1e-9
MOMENT_EPS = 0.0099
REACH_EPS = 0.05
ELEVATION_EPS = 90.0
# 可操作度のヒンジの下限 (最大値に対する割合、ε = 下限)。
MANIP_MIN = 0.4
# 人の項の成分の名前 (残差の並び)。
TERM_NAMES = ('moment',) + JOINTS + ('reach', 'elevation', 'manipulability')
N_TERMS = len(TERM_NAMES)

DOWN = np.array([0.0, 0.0, -1.0])
UP = np.array([0.0, 0.0, 1.0])
FWD = np.array([1.0, 0.0, 0.0])
OUT = np.array([0.0, -1.0, 0.0])
EXT_SIGN = 1.0 if float(np.dot(np.cross(FWD, OUT), DOWN)) > 0.0 else -1.0
MEDIAL = 1.0 if float(np.dot(np.cross(FWD, DOWN), -OUT)) > 0.0 else -1.0

# 掌の変数を使わない段階で、人の項を (重み 0 で) 評価する点。勾配の NaN を避けるための正常な腕の姿勢
# (肩の真下やや前、指先が下・掌が人の側)。値は使わない。
SAFE_POS = np.array([0.1, -0.05, -0.85])
SAFE_ROT = np.column_stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

# PoseParam に足す並び (pose_solver.param_layout の最後)。
PARAM_FIELDS = (
    ('palm_active', 1), ('palm0_pos', 3), ('palm0_rot', 9),
    ('human_shoulder', 3), ('human_basis', 9), ('human_length', 1), ('human_ratio', 1),
    ('human_w', N_TERMS),
    # 台車の向きの制約 (|台車 yaw − 人の向き| ≤ 幅)。角度の制約なので解いた後に切り詰める。
    ('yaw_active', 1), ('yaw_lower', 1), ('yaw_upper', 1),
)
# 掌を留める項の許容幅 (使わない段階で、掌の変数の LM の歩幅を 0 にするため)。
PALM_PIN_EPS = 1e-3


def luba_limits():
    """LUBA の限界 [度] (正の側、負の側。無い側は inf)。``hand_comfort._LIMITS`` と同じ。"""
    import side_by_side_transition as sbs
    pos, neg = [], []
    for key in JOINTS:
        p, n = sbs.luba_limit(key, True), sbs.luba_limit(key, False)
        pos.append(math.inf if p is None else p)
        neg.append(math.inf if n is None else n)
    return np.array(pos), np.array(neg)


def finger_length():
    import side_by_side_transition as sbs
    return float(sbs.HUMAN_FINGER_LENGTH)


def constants():
    """人の項が焼き込む定数 (export の鍵にも入れる)。"""
    pos, neg = luba_limits()
    return dict(limit_pos=pos, limit_neg=neg, finger_length=finger_length())


# --- 掌の変数とロボットの手先の目標 ------------------------------------------

def rotvec_matrix(xp, w):
    """回転ベクトル ``w`` の回転行列 (w = 0 でも値・勾配が有限)。"""
    theta = xp.sqrt(xp.sum(w * w) + 1e-12)
    k = xp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    a = xp.sinc(theta / math.pi)
    b = 0.5 * xp.sinc(theta / (2.0 * math.pi)) ** 2
    return xp.eye(3) + a * k + b * (k @ k)


def palm_pose(xp, palm, p):
    """掌の変数 ``palm`` (位置 3・回転ベクトル 3) の世界の掌 (位置, 回転行列)。回転は初期の掌に左から掛ける。"""
    r0 = p['palm0_rot'].reshape(3, 3)
    return palm[:3], rotvec_matrix(xp, palm[3:]) @ r0


def palm_target(xp, palm, target, p):
    """手先の目標 (位置, 回転行列)。``palm_active`` が 1 なら初期の掌への目標 ``target`` を、掌の変数の
    動き (初期の掌 → 今の掌) で動かす (目標は掌の座標系に固定)。0 なら ``target`` そのもの。"""
    tpos = target[:3]
    trot = target[3:12].reshape(3, 3)
    rw = rotvec_matrix(xp, palm[3:])
    act = p['palm_active'][0]
    new_pos = palm[:3] + rw @ (tpos - p['palm0_pos'])
    new_rot = rw @ trot
    return (act * new_pos + (1.0 - act) * tpos,
            act * new_rot + (1.0 - act) * trot)


def palm_pin_residual(xp, palm, p):
    """使わない段階で掌を初期値に留める項 (使う段階は 0)。"""
    x0 = xp.concatenate([p['palm0_pos'], xp.zeros(3)])
    return (1.0 - p['palm_active'][0]) * (palm - x0) / PALM_PIN_EPS


# --- 人の項 (hand_comfort_jax.comfort / errors と同じ式) -----------------------

def _unit(xp, v):
    return v / xp.sqrt(xp.sum(v * v) + 1e-12)


def _skew(xp, v):
    return xp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _rotation_between(xp, a, b):
    a, b = _unit(xp, a), _unit(xp, b)
    k = _skew(xp, xp.cross(a, b))
    c = xp.dot(a, b)
    return xp.eye(3) + k + k @ k / (1.0 + c + 1e-6)


def _signed_angle(xp, a, b, axis):
    axis = _unit(xp, axis)
    a = a - xp.dot(a, axis) * axis
    b = b - xp.dot(b, axis) * axis
    return xp.degrees(xp.arctan2(xp.dot(xp.cross(a, b), axis), xp.dot(a, b)))


def _two_link_elbow(xp, wrist, upper, fore):
    dist = xp.sqrt(xp.sum(wrist * wrist) + 1e-12)
    e1 = wrist / dist
    reach = xp.minimum(dist, upper + fore - 1e-6)
    a = (upper ** 2 - fore ** 2 + reach ** 2) / (2.0 * reach)
    r = xp.sqrt(xp.maximum(upper ** 2 - a ** 2, 0.0) + 1e-10)
    pref = xp.asarray(DOWN + 0.3 * OUT)
    q = pref - xp.dot(pref, e1) * e1
    return a * e1 + r * _unit(xp, q), dist


def comfort(xp, pos_n, rot, upper_ratio, consts, elbow_n=None, wrist_n=None):
    """``hand_comfort_jax.comfort`` と同じ (正準座標の掌 ``pos_n``・``rot``、上腕の比 ``upper_ratio``)。

    ``elbow_n``・``wrist_n`` (正準座標、腕の長さで割った肘・手首) を与えると、
    手首を掌から 4.1 cm・肘を 2 リンクの規則で決める代わりに、その肘・手首を使う (``sbs.HumanArm`` の腕)。
    角度の定義は ``sbs`` の ``human_arm_joint_angles``・``human_wrist_angles`` と同じ式。"""
    L = LENGTH
    upper = upper_ratio * L
    fore = L - upper
    pos = pos_n * L
    if wrist_n is None:
        wrist = pos + rot @ xp.asarray(WRIST_IN_PALM)
        elbow, dist = _two_link_elbow(xp, wrist, upper, fore)
    else:
        wrist = xp.asarray(wrist_n) * L
        elbow = xp.asarray(elbow_n) * L
        dist = xp.sqrt(xp.sum(wrist * wrist) + 1e-12)
    ua = _unit(xp, elbow)
    fa = _unit(xp, wrist - elbow)
    f, o, up, down = (xp.asarray(v) for v in (FWD, OUT, UP, DOWN))
    normal = rot[:, 1]
    sagittal = xp.sqrt(xp.dot(ua, f) ** 2 + xp.dot(ua, up) ** 2)
    flexion = xp.where(sagittal > 0.2, xp.degrees(xp.arctan2(
        xp.dot(ua, f), -xp.dot(ua, up))), 0.0)
    abduction = xp.degrees(xp.arcsin(xp.clip(xp.dot(ua, o), -0.999999, 0.999999)))
    elbow_flex = xp.degrees(xp.arctan2(
        xp.sqrt(xp.sum(xp.cross(ua, fa) ** 2) + 1e-12), xp.dot(ua, fa)))
    carry = _rotation_between(xp, down, ua)
    neutral_fore = carry @ f
    rotation = EXT_SIGN * _signed_angle(xp, neutral_fore, fa, ua)
    hi = elbow_flex >= 20.0
    ref_hi = MEDIAL * xp.cross(fa, ua)
    ref_lo = _rotation_between(xp, ua, fa) @ (MEDIAL * xp.cross(neutral_fore, ua))
    ref = xp.where(hi, ref_hi, ref_lo)
    pronation = -MEDIAL * _signed_angle(xp, ref, normal, fa)
    fingers, radial = rot[:, 0], rot[:, 2]
    along = xp.dot(fa, fingers)
    wrist_flexion = xp.degrees(xp.arctan2(-xp.dot(fa, normal), along))
    wrist_dev = xp.degrees(xp.arctan2(-xp.dot(fa, radial), along))
    angles = xp.stack([flexion, abduction, rotation, elbow_flex, pronation,
                       wrist_flexion, wrist_dev])
    lim_pos = xp.asarray(consts['limit_pos'])
    lim_neg = xp.asarray(consts['limit_neg'])
    limit = xp.where(angles >= 0.0, lim_pos, lim_neg)
    finite = xp.isfinite(limit)
    usage = xp.where(finite, xp.abs(angles) / xp.where(finite, limit, 1.0), 0.0)
    lo, hi_deg = ROTATION_WEIGHT_DEG
    rotation_weight = xp.clip((elbow_flex - lo) / (hi_deg - lo), 0.0, 1.0)
    usage = usage * xp.asarray([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]) + \
        usage * xp.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * rotation_weight
    tip = pos + fingers * consts['finger_length']
    coms = dict(upper=COM['upper'] * elbow,
                fore=elbow + COM['fore'] * (wrist - elbow),
                hand=wrist + COM['hand'] * (tip - wrist))
    moment = sum(m * xp.sqrt(coms[k][0] ** 2 + coms[k][1] ** 2 + 1e-12)
                 for k, m in MASS) / L
    elevation = xp.degrees(xp.arccos(xp.clip(xp.dot(ua, down), -0.999999, 0.999999)))
    pref = xp.asarray(DOWN + 0.3 * OUT)
    axis = _unit(xp, xp.cross(elbow, wrist - elbow) + MANIP_AXIS_EPS * xp.cross(elbow, pref))
    js = xp.cross(xp.eye(3), pos).T
    je = xp.cross(axis, pos - elbow)
    jac = xp.concatenate([js, je[:, None]], axis=1) / L
    w = xp.sqrt(xp.maximum(xp.linalg.det(jac @ jac.T), 0.0) + 1e-12)
    l1 = upper / L
    l2 = 1.0 - l1
    th = xp.asarray(MANIP_THETA)
    w_max = xp.max(l1 * l2 * xp.sin(th) * xp.sqrt(l1 ** 2 + l2 ** 2 + 2.0 * l1 * l2 * xp.cos(th)))
    return dict(moment=moment, usage=usage, reach=dist / L, angles=angles,
                elbow_flex=elbow_flex, elevation=elevation, manipulability=w / w_max)


def _bdot(a, b):
    return np.einsum('ni,ni->n', a, b)


def _bunit(v):
    return v / np.sqrt(np.sum(v * v, axis=-1, keepdims=True) + 1e-12)


def _bskew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], axis=-1),
                     np.stack([v[:, 2], z, -v[:, 0]], axis=-1),
                     np.stack([-v[:, 1], v[:, 0], z], axis=-1)], axis=-2)


def _brotation_between(a, b):
    a, b = _bunit(a), _bunit(b)
    k = _bskew(np.cross(a, b))
    c = _bdot(a, b)
    return np.eye(3)[None] + k + np.matmul(k, k) / (1.0 + c + 1e-6)[:, None, None]


def _bsigned_angle(a, b, axis):
    axis = _bunit(axis)
    a = a - _bdot(a, axis)[:, None] * axis
    b = b - _bdot(b, axis)[:, None] * axis
    return np.degrees(np.arctan2(_bdot(np.cross(a, b), axis), _bdot(a, b)))


def luba_batch(pos_n, rot, upper_ratio, consts):
    """``comfort`` の LUBA 7 関節の角度 ``angles`` (n, 7) と使用量 ``usage`` (n, 7) だけを、正準座標の掌
    (n, 3)・(n, 3, 3) についてまとめて求める (numpy)。式は ``comfort`` と同じ順。"""
    pos_n = np.asarray(pos_n, dtype=np.float64)
    rot = np.asarray(rot, dtype=np.float64)
    n = len(pos_n)
    L = LENGTH
    upper = upper_ratio * L
    fore = L - upper
    pos = pos_n * L
    wrist = pos + np.einsum('nij,j->ni', rot, WRIST_IN_PALM)
    # _two_link_elbow
    dist = np.sqrt(np.sum(wrist * wrist, axis=1) + 1e-12)
    e1 = wrist / dist[:, None]
    reach = np.minimum(dist, upper + fore - 1e-6)
    a = (upper ** 2 - fore ** 2 + reach ** 2) / (2.0 * reach)
    r = np.sqrt(np.maximum(upper ** 2 - a ** 2, 0.0) + 1e-10)
    pref = DOWN + 0.3 * OUT
    q = pref[None] - (e1 @ pref)[:, None] * e1
    elbow = a[:, None] * e1 + r[:, None] * _bunit(q)
    ua = _bunit(elbow)
    fa = _bunit(wrist - elbow)
    f = np.broadcast_to(FWD, (n, 3))
    o = np.broadcast_to(OUT, (n, 3))
    up = np.broadcast_to(UP, (n, 3))
    down = np.broadcast_to(DOWN, (n, 3))
    normal = rot[:, :, 1]
    sagittal = np.sqrt(_bdot(ua, f) ** 2 + _bdot(ua, up) ** 2)
    flexion = np.where(sagittal > 0.2, np.degrees(np.arctan2(_bdot(ua, f), -_bdot(ua, up))), 0.0)
    abduction = np.degrees(np.arcsin(np.clip(_bdot(ua, o), -0.999999, 0.999999)))
    elbow_flex = np.degrees(np.arctan2(
        np.sqrt(np.sum(np.cross(ua, fa) ** 2, axis=1) + 1e-12), _bdot(ua, fa)))
    carry = _brotation_between(down, ua)
    neutral_fore = np.einsum('nij,j->ni', carry, FWD)
    rotation = EXT_SIGN * _bsigned_angle(neutral_fore, fa, ua)
    hi = elbow_flex >= 20.0
    ref_hi = MEDIAL * np.cross(fa, ua)
    ref_lo = np.einsum('nij,nj->ni', _brotation_between(ua, fa), MEDIAL * np.cross(neutral_fore, ua))
    ref = np.where(hi[:, None], ref_hi, ref_lo)
    pronation = -MEDIAL * _bsigned_angle(ref, normal, fa)
    fingers, radial = rot[:, :, 0], rot[:, :, 2]
    along = _bdot(fa, fingers)
    wrist_flexion = np.degrees(np.arctan2(-_bdot(fa, normal), along))
    wrist_dev = np.degrees(np.arctan2(-_bdot(fa, radial), along))
    angles = np.stack([flexion, abduction, rotation, elbow_flex, pronation,
                       wrist_flexion, wrist_dev], axis=1)
    lim_pos = np.asarray(consts['limit_pos'], dtype=np.float64)
    lim_neg = np.asarray(consts['limit_neg'], dtype=np.float64)
    limit = np.where(angles >= 0.0, lim_pos, lim_neg)
    finite = np.isfinite(limit)
    usage = np.where(finite, np.abs(angles) / np.where(finite, limit, 1.0), 0.0)
    lo, hi_deg = ROTATION_WEIGHT_DEG
    rotation_weight = np.clip((elbow_flex - lo) / (hi_deg - lo), 0.0, 1.0)
    usage = usage * np.asarray([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]) + \
        usage * np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]) * rotation_weight[:, None]
    return angles, usage


def canonical_batch(palm_pos, palm_rot, p):
    """``canonical`` を掌 (n, 3)・(n, 3, 3) についてまとめて (numpy、``palm_active`` = 1)。"""
    basis = np.asarray(p['human_basis'], dtype=np.float64).reshape(3, 3)
    length = p['human_length'][0] + (1.0 - p['palm_active'][0])
    pos = (np.asarray(palm_pos, dtype=np.float64) - p['human_shoulder']) @ basis.T / length
    rot = np.matmul(basis[None], np.asarray(palm_rot, dtype=np.float64))
    fingers, normal = rot[:, :, 0], rot[:, :, 1]
    return pos, np.stack([fingers, normal, np.cross(fingers, normal)], axis=2)


def errors(xp, pos_n, rot, upper_ratio, consts, elbow_n=None, wrist_n=None):
    """人の項の 11 成分 (``TERM_NAMES`` の順、各 e/ε)。最初の 10 は ``hand_comfort_jax.errors`` と同じ。
    ``elbow_n``・``wrist_n`` は ``comfort`` と同じ。"""
    c = comfort(xp, pos_n, rot, upper_ratio, consts, elbow_n, wrist_n)
    first_two = xp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e_joint = c['usage'] * (1.0 - first_two * (c['angles'] >= 0.0))
    e_reach = xp.maximum(c['reach'] - 1.0, 0.0) / REACH_EPS
    e_elev = c['elevation'] / ELEVATION_EPS
    e_manip = xp.maximum(MANIP_MIN - c['manipulability'], 0.0) / MANIP_MIN
    return xp.concatenate([xp.reshape(c['moment'] / MOMENT_EPS, (1,)), e_joint,
                           xp.stack([e_reach, e_elev, e_manip])])


def canonical(xp, palm_pos, palm_rot, p):
    """世界の掌を人の正準座標へ (位置は腕の長さで割る)。``hand_comfort_current.to_canonical`` と同じ。"""
    basis = p['human_basis'].reshape(3, 3)
    # 使わない段階 (値が全部 0) でも 0 で割らない。
    length = p['human_length'][0] + (1.0 - p['palm_active'][0])
    pos = basis @ (palm_pos - p['human_shoulder']) / length
    rot = basis @ palm_rot
    fingers, normal = rot[:, 0], rot[:, 1]
    return pos, xp.stack([fingers, normal, xp.cross(fingers, normal)], axis=1)


def human_residual(xp, palm, p, consts):
    """人の項 (11 成分 × 重み ``human_w``)。``palm_active`` が 0 の段階は正常な点で評価して重み 0。"""
    act = p['palm_active'][0]
    pos_w, rot_w = palm_pose(xp, palm, p)
    pos_c, rot_c = canonical(xp, pos_w, rot_w, p)
    pos_c = act * pos_c + (1.0 - act) * xp.asarray(SAFE_POS)
    rot_c = act * rot_c + (1.0 - act) * xp.asarray(SAFE_ROT)
    ratio = act * p['human_ratio'][0] + (1.0 - act) * (UPPER / LENGTH)
    return p['human_w'] * errors(xp, pos_c, rot_c, ratio, consts)


# --- 1 人分の値 (numpy) ---------------------------------------------------

def canonical_basis(forward, outward):
    """正準座標への行列 (行が 前・-外向き・上)。左手は鏡映 (行列式 -1) になり、y を反転して右手に写す。"""
    f = np.array([forward[0], forward[1], 0.0], dtype=np.float64)
    f = f / np.linalg.norm(f)
    o = np.asarray(outward, dtype=np.float64)
    return np.array([f, -o, [0.0, 0.0, 1.0]])


def person_values(joints, hand):
    """人の肩・正準座標の行列・腕の長さ・上腕の比 (``sbs.HumanArm`` と同じ骨格の値)。"""
    import side_by_side_transition as sbs
    import solve_palm_ik as spik
    shoulder = np.asarray(joints['{}Shoulder'.format(hand)], dtype=np.float64)
    elbow = np.asarray(joints['{}Elbow'.format(hand)], dtype=np.float64)
    wrist = np.asarray(joints['{}Wrist'.format(hand)], dtype=np.float64)
    upper = float(np.linalg.norm(elbow - shoulder))
    fore = float(np.linalg.norm(wrist - elbow))
    facing = spik.human_facing_direction(joints)
    basis = canonical_basis(facing, sbs._outward(joints, hand))
    return dict(human_shoulder=shoulder, human_basis=basis.ravel(),
                human_length=[upper + fore], human_ratio=[upper / (upper + fore)])


def param_values(joints, hand, palm_pos, palm_rot, active, yaw_range=None):
    """PARAM_FIELDS の値 (``build_param`` に渡す dict)。``active`` が False なら掌は初期値に留める。
    ``yaw_range`` (下限, 上限) [rad] を与えると台車の向きの制約を有効にする。"""
    out = dict(palm_active=[1.0 if active else 0.0],
               palm0_pos=np.asarray(palm_pos, dtype=np.float64),
               palm0_rot=np.asarray(palm_rot, dtype=np.float64).ravel(),
               human_w=np.ones(N_TERMS) if active else np.zeros(N_TERMS))
    out.update(person_values(joints, hand))
    if yaw_range is not None:
        out.update(yaw_active=[1.0], yaw_lower=[yaw_range[0]], yaw_upper=[yaw_range[1]])
    return out


def initial_palms(n, palm_pos):
    """掌の変数の初期値 (n, 6): 初期の掌の位置・回転 0。"""
    return np.tile(np.concatenate([np.asarray(palm_pos, dtype=np.float64), np.zeros(3)]), (n, 1))


def palm_from_var(x, palm_rot0):
    """解いた掌の変数 (6) から世界の掌 (位置, 回転行列) (numpy)。"""
    from scipy.spatial.transform import Rotation
    x = np.asarray(x, dtype=np.float64)
    return x[:3].copy(), Rotation.from_rotvec(x[3:]).as_matrix() @ np.asarray(palm_rot0)


def palm_values(palm_pos, palm_rot, joints, hand, consts=None):
    """1 つの掌の人の項 (numpy で評価した dict: 成分・肘・上腕挙上・肩外転)。報告用。"""
    import jax.numpy as jnp
    consts = consts or constants()
    p = {k: np.asarray(v, dtype=np.float64).reshape(-1)
         for k, v in person_values(joints, hand).items()}
    p = {k: jnp.asarray(v) for k, v in p.items()}
    p['palm_active'] = jnp.ones(1)
    pos_c, rot_c = canonical(jnp, jnp.asarray(palm_pos), jnp.asarray(palm_rot), p)
    c = comfort(jnp, pos_c, rot_c, p['human_ratio'][0], consts)
    e = errors(jnp, pos_c, rot_c, p['human_ratio'][0], consts)
    return dict(terms=[float(v) for v in np.asarray(e)], total=float(np.sum(np.asarray(e) ** 2)),
                elbow=float(c['elbow_flex']), elevation=float(c['elevation']),
                abduction=float(c['angles'][1]), manipulability=float(c['manipulability']),
                reach=float(c['reach']))
