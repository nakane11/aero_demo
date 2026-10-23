"""実機ノード (``run_camera_pipeline_test.py``) のログ (``person_NN.log``) から、
ロボットの状態 (``handshake_stages``) が切り替わった時刻を読み取る (ROS 不要)。

``~status`` を記録していない bag を表示ツールで再生するときに、当時の状態を
出すのに使う。ログの各行は ``HH:MM:SS.mmm`` (ノードの PC の時計) で始まる。
ログに出ていない切り替わり (関節の到着待ちの始まりなど) は推測せず、前の
状態のままにする。
"""

import bisect
import datetime
import re

from aero_demo import handshake_stages as stages

_TIME = re.compile(r'^(\d{2}):(\d{2}):(\d{2})\.(\d{3}) (.*)$')

# (行の本文の正規表現, 切り替わる状態)。上から順に照合し、最初に合ったものを使う。
# 'after' があるものは、その状態の後に出たときだけ使う (同じ書式の行が区間ごとに
# 出るため)。
_RULES = [
    (re.compile(r'^\[ARMED\] 試行\d+: .*始めます'), stages.STAGE_PLAN_IK, None),
    (re.compile(r'^\[debug\] \{"event": "motion"'),
     stages.STAGE_PLAN_TRANSITION, None),
    (re.compile(r'^\[result\] 試行'), stages.STAGE_RESULT, None),
    (re.compile(r'^\[auto-execute\]'), stages.STAGE_NOD_WAIT, None),
    (re.compile(r'^\[execute\] 実機で waypoint を'), stages.STAGE_LEAD_IN, None),
    (re.compile(r'^\[head\] 差し出し手の掌を追います'), stages.STAGE_SPIRAL,
     (stages.STAGE_LEAD_IN,)),
    (re.compile(r'^\[debug\]\[segment\] ゴール完了待ち'),
     stages.STAGE_BASE_CORRECTION, (stages.STAGE_LEAD_IN, stages.STAGE_SPIRAL)),
    (re.compile(r'^\[debug\]\[joint\] hover 到達直後'),
     stages.STAGE_PRESS_REFINE, None),
    (re.compile(r'^\[execute\]\[refine\]'), stages.STAGE_PRESS_IN, None),
    (re.compile(r'^\[debug\]\[joint\] 押し込み終了'), stages.STAGE_GRASP, None),
    (re.compile(r'^\[execute\] ハンドを握ります'),
     stages.STAGE_TRANSITION_WAIT, None),
    (re.compile(r'^\[execute\] 手を下ろしながら横並びの位置へ'),
     stages.STAGE_TRANSITION, None),
    (re.compile(r'^\[execute\] 実行を終了しました'), stages.STAGE_DONE, None),
    (re.compile(r'^\[RESET\]'), stages.STAGE_IDLE, None),
    (re.compile(r'^\[ARM\] ARMED になりました'), stages.STAGE_ARMED, None),
]


class LogStages(object):
    """ログから読み取った状態の切り替わり ``[(時刻 [epoch 秒], (見出し, 補足))]``。"""

    def __init__(self, path, reference_time, utc_offset_hours=None):
        """``reference_time``: ログと同じ日の時刻 (epoch 秒。bag の時刻など)。
        ログの時刻 (時:分:秒) をその日付で epoch 秒にする。``utc_offset_hours``
        はログを書いた PC の時差 (None ならこの PC の地方時)。日付をまたぐ
        ログは、前の行より時刻が戻ったら翌日とみなす。"""
        if utc_offset_hours is None:
            tz = datetime.datetime.fromtimestamp(reference_time).astimezone().tzinfo
        else:
            tz = datetime.timezone(datetime.timedelta(hours=utc_offset_hours))
        day = datetime.datetime.fromtimestamp(reference_time, tz).date()
        self.events = []
        previous = None
        last_stage = None
        with open(path, encoding='utf-8') as f:
            for line in f:
                match = _TIME.match(line.rstrip('\n'))
                if match is None:
                    continue
                h, m, s, ms, text = match.groups()
                stamp = datetime.datetime.combine(
                    day, datetime.time(int(h), int(m), int(s), int(ms) * 1000),
                    tz).timestamp()
                if previous is not None and stamp < previous - 3600:
                    day += datetime.timedelta(days=1)
                    stamp += 86400
                previous = stamp
                for pattern, stage, after in _RULES:
                    if not pattern.search(text):
                        continue
                    if after is not None and last_stage not in after:
                        continue
                    if stage != last_stage:
                        self.events.append((stamp, stage))
                        last_stage = stage
                    break
        self._times = [t for t, _ in self.events]

    def stage_at(self, t):
        """時刻 ``t`` [epoch 秒] の ``(状態, その状態になった時刻)``。ログの
        最初の出来事より前なら ``(None, None)``。"""
        index = bisect.bisect_right(self._times, t) - 1
        if index < 0:
            return None, None
        stamp, stage = self.events[index]
        return stage, stamp
