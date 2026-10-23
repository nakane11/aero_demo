"""握手デモのロボットの状態 (実機ノードの ``~status`` と表示ツールが共有する)。

各値は (見出し, 補足) で、細かい処理の単位。表示ではこれを ``STATE_GROUPS`` の
ロボットの状態 (待機中・掌検出中・計算中など) にまとめ、その並び (1 から) で
番号を付ける (``state_of``)。
"""

STAGE_STARTING = ('起動中', '')
STAGE_IDLE = ('待機中', 'ARM ボタンを押すと手を差し出す人を待ちます')
STAGE_INITIAL_POSE = ('初期姿勢へ戻っています', '')
STAGE_ARMED = ('差し出した手を待っています', '手を差し出してください')
STAGE_ARMED_RETRY = ('差し出した手を待っています',
                     '今の手を動かすか向きを変えてから、もう一度差し出してください')
STAGE_PLAN_IK = ('hover・押し込みの IK',
                 '第1段バッチ IK (台車も) → hover の事後検証 → 第2段 押し込み + 視線')
STAGE_PLAN_APPROACH = ('接近軌道の計画',
                       'lead-in と 20 waypoint の螺旋、全 waypoint の干渉検証')
STAGE_PLAN_TRANSITION = ('横並び移動の計画',
                         '移動先のバッチ IK と手を下ろす経路')
STAGE_RESULT = ('計算結果を表示中', '実機は動かしません (RESET で最初から)')
STAGE_NOD_WAIT = ('うなずきの完了待ち', '')
STAGE_LEAD_IN = ('接近開始位置へ', 'その場回転 → 直進 (首は顔を追う)')
STAGE_SPIRAL = ('螺旋で hover へ',
                '手を中心に公転 + 自転、腕を下ろした姿勢から hover へ (首は掌を追う)')
STAGE_BASE_CORRECTION = ('台車位置の補正', 'hover で /scan 照合によりスリップを直す')
STAGE_HOVER_SETTLE = ('関節の到着待ち', 'hover の姿勢に腕・脚が着くのを待つ')
STAGE_PRESS_REFINE = ('押し込み直前の補正',
                      '人の手を検出し直し、押し込みの IK を解き直す')
STAGE_PRESS_IN = ('押し込み', '指先を押し出しながら掌へ詰める')
STAGE_GRASP = ('手を握っています', '')
STAGE_TRANSITION_WAIT = ('横並び移動の開始待ち', '')
STAGE_TRANSITION = ('横並び移動',
                    'つないだ手を体の横へ下ろしながら横並びへ (首は顔を追う)')
STAGE_DONE = ('実行を終了しました', 'RESET で最初から')
STAGE_ERROR = ('実行中にエラーが起きました', '端末のログを確認してください')
# 裏の計算 (状態の補足に出す)。
BACKGROUND_TRANSITION_REFINE = '裏で横並びの寄せを計算中'

# ロボットの状態 (番号はこの順に 1 から) と、それにまとめる処理。
STATE_GROUPS = (
    ('待機中', (STAGE_STARTING, STAGE_IDLE)),
    ('掌検出中', (STAGE_INITIAL_POSE, STAGE_ARMED)),
    ('計算中', (STAGE_PLAN_IK, STAGE_PLAN_APPROACH, STAGE_PLAN_TRANSITION,
               STAGE_RESULT, STAGE_NOD_WAIT)),
    ('接近中', (STAGE_LEAD_IN, STAGE_SPIRAL)),
    ('hover で補正中', (STAGE_BASE_CORRECTION, STAGE_HOVER_SETTLE,
                     STAGE_PRESS_REFINE)),
    ('押し込み中', (STAGE_PRESS_IN,)),
    ('手を握って待機中', (STAGE_GRASP, STAGE_TRANSITION_WAIT)),
    ('横並び移動中', (STAGE_TRANSITION,)),
    ('終了', (STAGE_DONE,)),
    ('エラー', (STAGE_ERROR,)),
)
_STATES = {title: (number, name)
           for number, (name, members) in enumerate(STATE_GROUPS, start=1)
           for title, _ in members}


def state_of(title):
    """処理の見出し ``title`` が属するロボットの状態の ``(番号, 名前)``
    (知らない見出しは None)。"""
    return _STATES.get(title)


def numbered_title(title):
    """ロボットの状態の番号と名前を付けた見出し (``'3. 計算中: 接近軌道の計画'``)。"""
    state = state_of(title)
    if state is None:
        return title
    number, name = state
    if title == name:
        return '{}. {}'.format(number, title)
    return '{}. {}: {}'.format(number, name, title)


# シミュレーション表示で、実機では長さが決まらない状態に当てる時間 [秒]。
SIMULATED_DURATIONS = {
    STAGE_NOD_WAIT[0]: 1.0,
    STAGE_BASE_CORRECTION[0]: 1.0,
    STAGE_HOVER_SETTLE[0]: 0.5,
    STAGE_PRESS_REFINE[0]: 0.5,
    # 押し込みの後、横並び移動を始めるまで (実機ノードの --side-by-side-delay の
    # 既定 4 秒を、手を握るのと開始待ちに分ける)。
    STAGE_GRASP[0]: 1.0,
    STAGE_TRANSITION_WAIT[0]: 3.0,
}
# 横並び移動の区間の時間の下限 (実機ノードの --side-by-side-segment-time の既定)。
TRANSITION_MIN_SEGMENT_TIME = 0.2
