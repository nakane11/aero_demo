"""計画した握手の軌道を、実機と同じ時間で動かしたときの姿勢と状態 (ROS 不要)。

表示ツールのシミュレーション表示が使う。区間の時間と補間は実機ノードと同じ
``execution_timing`` で決め、実機で長さの決まらない状態 (うなずき、hover での補正、
横並び移動の開始待ち) は ``handshake_stages.SIMULATED_DURATIONS`` の固定値で近似する。
状態は実機ノードの ``~status`` と同じ (``handshake_stages``)。
"""

from aero_demo import execution_timing
from aero_demo import handshake_stages as stages


class PlanTimeline(object):
    """表示用 waypoint 列 (lead-in + 接近 + 押し込み + 横並び移動) を、実機の
    実行と同じ区間 (接近 / 押し込み / 横並び移動) と待ちに分けて時刻で引けるように
    したもの。

    ``display_waypoints[:n_prepend + n_approach]`` が接近 (末尾が hover)、
    ``[n_prepend + n_approach - 1 : -n_transition]`` が押し込み、末尾の
    ``n_transition`` 個が横並び移動 (押し込みの最後から始める)。
    """

    def __init__(self, display_waypoints, n_prepend, n_approach, n_transition,
                 joint_names, joints_by_name):
        self.joint_names = list(joint_names)
        n_total = len(display_waypoints)
        n_body = n_total - n_transition
        reach = min(max(n_prepend + n_approach, 1), n_body)
        self.phases = []
        self._first_segment = None
        t = 0.0

        def add_move(waypoints, first_index, stage_at, min_segment_time):
            nonlocal t
            segment = execution_timing.SegmentMotion(
                waypoints, self.joint_names, joints_by_name,
                min_segment_time=min_segment_time)
            if self._first_segment is None:
                self._first_segment = segment
            self.phases.append(dict(start=t, duration=segment.duration,
                                    segment=segment, first_index=first_index,
                                    stage_at=stage_at))
            t += segment.duration
            return segment

        def add_wait(stage, index):
            nonlocal t
            duration = stages.SIMULATED_DURATIONS[stage[0]]
            self.phases.append(dict(start=t, duration=duration, segment=None,
                                    first_index=index,
                                    stage_at=lambda _t: stage))
            t += duration

        spiral_start = []  # 接近の区間を作ってから決まる

        def approach_stage(local_t):
            return (stages.STAGE_LEAD_IN if local_t < spiral_start[0]
                    else stages.STAGE_SPIRAL)

        add_wait(stages.STAGE_NOD_WAIT, 0)
        approach = add_move(display_waypoints[:reach], 0, approach_stage,
                            execution_timing.MIN_SEGMENT_TIME)
        spiral_start.append(approach.waypoint_time(n_prepend))
        if reach < n_body:
            for stage in (stages.STAGE_BASE_CORRECTION,
                          stages.STAGE_HOVER_SETTLE,
                          stages.STAGE_PRESS_REFINE):
                add_wait(stage, reach - 1)
            add_move(display_waypoints[reach - 1:n_body], reach - 1,
                     lambda _t: stages.STAGE_PRESS_IN,
                     execution_timing.MIN_SEGMENT_TIME)
            add_wait(stages.STAGE_GRASP, n_body - 1)
            if n_transition:
                add_wait(stages.STAGE_TRANSITION_WAIT, n_body - 1)
                add_move(display_waypoints[n_body - 1:], n_body - 1,
                         lambda _t: stages.STAGE_TRANSITION,
                         stages.TRANSITION_MIN_SEGMENT_TIME)
        self.duration = t

    def sample(self, t):
        """時刻 ``t`` [秒] の ``dict(angles (joint_names の順), base (x, y, yaw),
        stage ((見出し, 補足)), index (表示用 waypoint の番号、小数),
        moving (区間を動いている途中か))``。待ちの間は直前の区間の最後の姿勢
        (最初のうなずき待ちは最初の姿勢) のまま。最後まで進んだら実行の終了。"""
        t = min(max(t, 0.0), self.duration)
        last = None  # (区間, その区間の最初の waypoint の番号)
        # 最後の時刻は (区間の時間の足し合わせの誤差によらず) 実行の終了にする。
        phases = self.phases if t < self.duration - 1e-6 else []
        for phase in self.phases:
            if phase['segment'] is not None:
                last_segment = (phase['segment'], phase['first_index'])
        for phase in phases:
            segment = phase['segment']
            local_t = t - phase['start']
            if segment is not None:
                last = (segment, phase['first_index'])
            if local_t < phase['duration']:
                stage = phase['stage_at'](local_t)
                if segment is None:
                    if last is None:  # 最初の区間の前
                        last, local_t = (self._first_segment, 0), 0.0
                    else:
                        local_t = last[0].duration
                angles, base, progress = last[0].sample(local_t)
                return dict(angles=angles, base=base, stage=stage,
                            index=last[1] + progress,
                            moving=segment is not None)
        segment, first_index = last_segment
        angles, base, progress = segment.sample(segment.duration)
        return dict(angles=angles, base=base, stage=stages.STAGE_DONE,
                    index=first_index + progress, moving=False)
