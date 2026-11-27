import numpy as np

from aero_demo.people_pose_estimator import PeoplePoseEstimator


def _scene(w=400, h=200):
    """左半分の奥 (4 m) に背景、中央に手前 (1.5 m) の人、右端は深度なし."""
    img = np.full((h, w, 3), 200, np.uint8)
    depth = np.full((h, w), 4.0, np.float32)
    depth[:, 180:220] = 1.5
    depth[:, 360:] = 0.0
    return img, depth


def test_far_pixels_are_painted_and_near_margin_kept():
    img, depth = _scene()
    out = PeoplePoseEstimator.mask_far_pixels(img, depth, 2.5,
                                              keep_margin=0.05)
    painted = out.max(axis=2) == 0
    assert not painted[:, 180:220].any()      # 手前の人
    assert not painted[:, 360:].any()         # 深度なし
    assert not painted[:, 165:180].any()      # 人の縁 (幅の 5% = 20px 以内)
    assert painted[:, :140].all()             # 奥
    assert painted[:, 250:340].all()


def test_without_margin_paints_right_up_to_the_person():
    img, depth = _scene()
    out = PeoplePoseEstimator.mask_far_pixels(img, depth, 2.5,
                                              keep_margin=0.0, step=1)
    painted = out.max(axis=2) == 0
    assert painted[:, :180].all()
    assert not painted[:, 180:220].any()


def test_disabled_or_mismatched_returns_input():
    img, depth = _scene()
    assert PeoplePoseEstimator.mask_far_pixels(img, depth, 0) is img
    assert PeoplePoseEstimator.mask_far_pixels(img, depth[:100], 2.5) is img
    near = np.full_like(depth, 1.0)
    assert PeoplePoseEstimator.mask_far_pixels(img, near, 2.5) is img
