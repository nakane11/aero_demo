"""見張りの「誰も手を出していないときは近さ・正対で選ぶ」と条件付きの見渡し."""
import math

import numpy as np
import pytest

mpw = pytest.importorskip('aero_demo.multi_person_watcher')

W = 640


def _body(x, y, facing_robot=True):
    """base 系 (x 前・y 左) の (x, y) に立つ人の骨格。"""
    # ロボットを向いた人の左肩はロボットから見て右 (y が小さい)。
    d = -0.2 if facing_robot else 0.2
    return dict(LShoulder=np.array([x, y + d, 1.4]),
                RShoulder=np.array([x, y - d, 1.4]),
                LHip=np.array([x, y + d / 2, 0.9]),
                RHip=np.array([x, y - d / 2, 0.9]))


def test_idle_cost_prefers_near_front_and_facing():
    front = mpw.idle_cost(_body(1.5, 0.0))
    side = mpw.idle_cost(_body(1.5, 1.0))
    far = mpw.idle_cost(_body(2.5, 0.0))
    back = mpw.idle_cost(_body(1.5, 0.0, facing_robot=False))
    assert front == pytest.approx(1.5, abs=1e-6)
    assert front < side and front < far and front < back
    assert back == pytest.approx(1.5 + mpw.IDLE_FACING_WEIGHT, abs=1e-6)
    # 横の人は体が -x を向き、ロボットの方とは atan2(1.0, 1.5) ずれる。
    bearing = math.atan2(1.0, 1.5)
    assert side == pytest.approx(
        math.hypot(1.5, 1.0) + mpw.IDLE_BEARING_WEIGHT * bearing
        + mpw.IDLE_FACING_WEIGHT * 0.5 * (1.0 - math.cos(bearing)), abs=1e-6)


def _person(u, score, cost, center=None):
    return dict(box=[u - 30, 50, u + 30, 300], score=score, idle_cost=cost,
                distance=cost, center=center)


class _Sim(object):
    """PersonTracker + TargetChooser を画像の時刻 t で進める."""

    def __init__(self, **kw):
        self.tracker = mpw.PersonTracker()
        self.chooser = mpw.TargetChooser(**kw)
        self.t = 0.0

    def step(self, people, dt=0.1):
        self.t += dt
        tracks = self.tracker.update(people, self.t, W)
        target, switched = self.chooser.update(tracks, self.tracker.tracks,
                                               self.t)
        ids = {i: p['track_id'] for i, p in enumerate(people)}
        return target, switched, ids


def test_initial_choice_is_nearest_front_when_nobody_offers():
    sim = _Sim()
    # 0 番は手の位置で少しスコアが高いが遠い横の人、1 番は正面の近くの人。
    for _ in range(5):
        target, _, ids = sim.step([_person(100, 0.2, 3.0),
                                   _person(320, 0.05, 1.4)])
    assert target == ids[1]


def test_offer_still_switches_by_score():
    sim = _Sim()
    for _ in range(5):
        target, _, ids = sim.step([_person(100, 0.0, 3.0),
                                   _person(320, 0.0, 1.4)])
    assert target == ids[1]
    for _ in range(5):
        target, _, ids = sim.step([_person(100, 0.9, 3.0),
                                   _person(320, 0.0, 1.4)])
    assert target == ids[0]


def test_switch_works_when_lock_is_called_every_update():
    """ノードは見張りの結果ごとに lock(False) を呼ぶ。それでも切り替わること."""
    sim = _Sim()
    for _ in range(5):
        sim.chooser.lock(False)
        target, _, ids = sim.step([_person(100, 0.0, 3.0),
                                   _person(320, 0.0, 1.4)])
    assert target == ids[1]
    for _ in range(5):
        sim.chooser.lock(False)
        target, _, ids = sim.step([_person(100, 0.9, 3.0),
                                   _person(320, 0.0, 1.4)])
    assert target == ids[0]
    # 確定後 (lock(True)) は切り替えない。
    for _ in range(5):
        sim.chooser.lock(True)
        target, _, ids = sim.step([_person(100, 0.0, 3.0),
                                   _person(320, 0.9, 1.4)])
    assert target == ids[0]


def test_idle_switch_to_clearly_nearer_person_after_hold():
    sim = _Sim(idle_hold=1.0)
    for _ in range(5):
        target, _, ids = sim.step([_person(100, 0.0, 2.0),
                                   _person(320, 0.0, 2.1)])
    assert target == ids[0]
    # 1 番が正面に近づく (差 0.6 m 相当)。1 秒は今の人のまま。
    target, _, ids = sim.step([_person(100, 0.0, 2.0),
                               _person(320, 0.0, 1.4)])
    assert target == ids[0]
    for _ in range(12):
        target, _, ids = sim.step([_person(100, 0.0, 2.0),
                                   _person(320, 0.0, 1.4)])
    assert target == ids[1]


def test_small_cost_difference_does_not_switch():
    sim = _Sim()
    for _ in range(30):
        target, _, ids = sim.step([_person(100, 0.0, 2.0),
                                   _person(320, 0.0, 1.85)])
    first = target
    assert first == ids[1]
    for _ in range(30):
        target, _, _ = sim.step([_person(100, 0.0, 1.75),
                                 _person(320, 0.0, 1.85)])
    assert target == first


def test_hold_keeps_target_while_it_is_out_of_view():
    sim = _Sim()
    for _ in range(5):
        target, _, ids = sim.step([_person(320, 0.0, 1.4),
                                   _person(560, 0.0, 2.5)])
    first = ids[0]
    assert target == first
    sim.chooser.hold(sim.t + 2.0)
    # 首を振って 0 番が見えなくなり、2 番目の人だけが見える。
    for _ in range(15):
        target, _, _ = sim.step([_person(200, 0.0, 2.5)])
    assert target == first
    for _ in range(10):
        target, _, _ = sim.step([_person(200, 0.0, 2.5)])
    assert target != first


class _Watcher(mpw.MultiPersonWatcher):
    def __init__(self):  # PoseLandmarker は使わない
        self.glance_enabled = True
        self.glance_quiet = 2.0
        self.glance_interval = 6.0
        self.glance_request = None
        self._glance_active = False
        self._last_glance_t = None
        self.tracker = mpw.PersonTracker()
        self.chooser = mpw.TargetChooser()
        self.mask_keep_time = 0.5
        self.mask_pad = (0.35, 0.15)
        self.keep_pad = (0.25, 0.1)
        self.same_person_overlap = 0.6
        self._last_target_box = None
        self._suppressed = []


def _run(w, t, people, seconds):
    for _ in range(int(round(seconds / 0.1))):
        t += 0.1
        w.apply([dict(p) for p in people], t, W, 480)
    return t


def test_glance_after_quiet_target_then_interval_after_done():
    w = _Watcher()
    front = _person(320, 0.0, 1.4)
    t = _run(w, 0.0, [front], 1.5)
    assert w.take_glance_request() is None  # まだ 2 秒たっていない
    t = _run(w, t, [front], 1.0)
    request = w.take_glance_request()
    assert request is not None and request['target_id'] == w.target_id
    # 見渡している間は次を頼まない。
    t = _run(w, t, [front], 5.0)
    assert w.take_glance_request() is None
    w.glance_done()
    t = _run(w, t, [front], 5.0)
    assert w.take_glance_request() is None  # 終わってから 6 秒たっていない
    t = _run(w, t, [front], 1.5)
    assert w.take_glance_request() is not None


def test_no_glance_while_target_is_offering():
    w = _Watcher()
    offering = _person(320, 0.6, 1.4)
    _run(w, 0.0, [offering], 5.0)
    assert w.take_glance_request() is None


def test_hold_for_keeps_target_during_glance():
    w = _Watcher()
    front = _person(320, 0.0, 1.4)
    other = _person(100, 0.0, 2.5)
    t = _run(w, 0.0, [front, other], 1.0)
    first = w.target_id
    w.hold_for(3.0)
    t = _run(w, t, [_person(500, 0.0, 2.5)], 2.5)
    assert w.target_id == first
    t = _run(w, t, [_person(500, 0.0, 2.5)], 1.0)
    assert w.target_id != first
