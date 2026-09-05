#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人間の掌の位置姿勢 JSON (``estimate_palm_poses.py`` の出力) を入力とし、
ベース移動型ロボット (Aero) が全身 IK (台車の平面移動を含む) を解いて手を
繋ぐ姿勢を求め、その結果 (関節角・台車位置・手先姿勢) を JSON として保存
する。``rospy`` は import せず、保存済みの JSON を読んで 1 回だけ IK を
解くオフライン処理。

対象にするのは、掌推定が「人がこの手を差し出している」と判定した手
(掌 JSON の ``offered_hand`` が ``'L'``/``'R'``) を持つ人物だけ。
``offered_hand`` が ``null`` の人物は IK を解かず、入力と同じファイル名で
``target: false`` の JSON (IK の結果は持たない) を書き出す
(``view_handshake_poses.py`` が「対象外」として表示できるようにするため)。
使うロボットの腕は既定で人間の手の反対側 (``--robot-arm`` で上書き可)。
向かい合う握手ではなく人間と同じ方向を向いて反対側の手で繋ぐ想定なので、
掌の向きは鏡写しにせずそのまま使う。

Aero は常にワールド原点・台車位置固定で IK を開始する (``seed_arm_pose``
参照) ため、人物側 (骨格の全関節位置・掌の目標位置) を x/y 方向に平行移動
し、その人物の立ち位置がちょうど Aero の前方 ``HUMAN_FRONT_DISTANCE`` [m]
に来るようにしてから IK を解く (``human_translation_offset``/
``translate_joint_positions``/``translate_palm`` 参照)。台車の移動範囲は
``--base-x-range``/``--base-y-range``/``--base-yaw-range`` (既定はこの
立ち位置を中心とした ``BASE_MOVABLE_HALF_RANGE`` m の正方形) で指定する。

人体を障害物とした干渉回避も行う。``--skeleton-dir`` から人物ごとの全身の
関節位置を読み、体幹・頭部・四肢 (差し出している側の腕・手も含めて全身)
を ``skrobot.model.primitives.Cylinder`` で近似し、``batch_inverse_
kinematics`` の ``collision_obstacles`` に渡す。干渉を避ける対象のロボット
リンクは台車・胴体・頭部・両腕を含む全身 (``collision_link_list_for_arm``
参照)。ロボット側の干渉ジオメトリは実メッシュではなく、``view_aero_
collision_model.py`` と同じ方法で生成・キャッシュした box/cylinder/sphere
のプリミティブ近似形状を使う (``apply_collision_model`` 参照)。人体側・
ロボット側のどちらも部位による除外はせず、代わりに IK の目標位置を掌から
``TARGET_HOVER_OFFSET`` だけ浮かせることで、目標そのものが人体の干渉回避
ジオメトリと重ならないようにしている。これはソフトなペナルティ項なので、
干渉のない解が必ず得られるとは限らない上、``batch_inverse_kinematics`` の
収束判定 (``success_flags``) は位置・姿勢誤差だけを見て干渉ペナルティの
残差を見ないため、``solve_person_ik`` は収束した候補について採用前に必ず
``collision_pairs_min_distance`` で厳密な形状による事後検証を行い、
``collision_pairs`` に指定した組み合わせが実際には貫通したままの解は
棄却する (``--collision-verify-tolerance`` 参照)。

さらに、干渉検証まで通った候補についても、``TARGET_HOVER_OFFSET`` で
浮かせた目標に届いたというだけでは実際に人間に掌を押し付けられる保証は
無いため、``pick_verified_candidate`` は候補ごとに最後に後処理判定
(``solve_post_process``) を行う。台車を動かさず (人体との干渉回避も外した)
通常のヤコビアン法 IK で、腕を目標位置が掌へわずかにめり込む位置
(``POST_PROCESS_TARGET_HOVER_OFFSET``) まで詰め直すのと、ロボットが人間の
差し出している手 (掌) を見るよう首を向けるのを、``robot.inverse_
kinematics`` に ``move_target``/``target_coords`` 等をリストで渡す 1 回の
呼び出しで同時に解く (``solve_post_process`` 参照)。視線の目標は人間の
掌の位置 (IK を解く前から分かっている固定点) であり、ロボット自身の手先の
収束後の位置には依存しないため、腕と首を別々の呼び出しに分けて逐次解く
必要が無い。後処理判定に失敗した候補は棄却し、次の候補 (同じ向きの別の
初期値の解、それも尽きれば ``TURN_CANDIDATES_DEG`` の次の向き。干渉回避
付きバッチ IK は既に全向き × 全初期値を一括で解いてあるので、やり直すのは
干渉検証・後処理判定だけ) を試す。
全ての候補で後処理判定に失敗した場合のみ、干渉検証を通過した最初の候補を
後処理前のまま (``post_process`` キーを ``null`` にして) 採用する
フォールバックを行う (``pick_verified_candidate`` 参照)。採用された候補
は、後処理前の位置姿勢に加え、後処理に成功していれば後処理後の位置姿勢も
``post_process`` キーに持つ (``solved_result`` 参照)。

人体だけでなく、ロボット自身のリンク同士の干渉 (自己干渉) も既定で回避
する (``batch_inverse_kinematics`` の ``self_collision=True``、``--no-
self-collision`` で無効化できる)。チェックする組み合わせ (自己干渉の
ロボットリンク同士、および人体との干渉のロボットリンク×人体セグメント)
は、常に ``--collision-pairs`` (JSON: 2 要素の名前のリストのリスト。
``build_collision_pairs.py`` が ``analyze_collision_pairs.py`` の出力から
生成する) で明示的に指定した組み合わせだけに限る (``load_collision_
pairs`` 参照。各ペアの 2 要素目がロボットのリンク名なら自己干渉ペア、
``human_obstacle_names`` の人体セグメント名なら人体との干渉ペアとして
扱われる)。このファイルが既定のパスに無ければ、干渉回避 (自己干渉・
人体との干渉の両方) を丸ごと無効にして、通常のヤコビアン法の IK に
フォールバックする。

向きを 0/±90 度 (``TURN_CANDIDATES_DEG``) ずらした目標を、それぞれ
``--attempts-per-pose`` 個の初期値 (初期値 0 が「肩を開いた種の姿勢」+
台車はワールド原点、残りは関節範囲・台車可動範囲の一様乱数) から解き、
そうしてできる ``len(TURN_CANDIDATES_DEG) * --attempts-per-pose`` 個の解
**すべて** を候補として順に試して、干渉回避付きバッチ IK が解けて
(収束・事後の干渉検証を通過して)、かつ後処理判定
(``solve_post_process``) にも成功した最初のものを採用する
(``pick_verified_candidate`` 参照)。``batch_inverse_kinematics`` は既定
では 1 目標につき最良の 1 解しか返さない (``attempts_per_pose`` の中で
誤差最小のものを選ぶ) が、位置・姿勢誤差が最小の解が後処理判定にも通る
とは限らないので、``return_all_attempts=True`` を渡して集約させず、
初期値ごとの解を全て受け取る (``solve_person_ik`` 参照。捨てられていた
解も元々解かれてはいるので、計算時間・JAX のコンパイル結果は変わらない)。
IK は 1 目標ずつ逐次に解くのではなく、
``batch_inverse_kinematics`` で **人物 1 人ごとに 1 回** 解く (その人の
全向き × 全初期値を 1 バッチにまとめる)。``collision_obstacles`` は
1 回のバッチ呼び出し全体で 1 つの集合しか渡せない (skrobot 側の制約) ため、
人物ごとに「その人自身の身体」を障害物にする必要があり、人物をまたいで
バッチをまとめることはできない。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo estimate_palm_poses.py
    rosrun aero_demo solve_palm_ik.py

(いずれも --input-dir/--output-dir を省略すると、scripts/ 直下の
random_human_poses/ -> random_palm_poses/ -> random_handshake_poses/ を
共通の入出力先として自動的につながる)
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PKG_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)

# jax の永続コンパイルキャッシュを有効にする。skrobot は backend='jax' の
# バッチ IK を呼ぶ際に jax.jit したソルバをプロセス内 (``RobotModel.
# _batch_ik_collision_solver_cache`` 等) にキャッシュするので、このスクリプト
# 1 回の実行内で人物をまたいでの再コンパイルは (パラメータが同じ限り) 既に
# 起きない。しかしそのキャッシュはプロセスを終了すると消えるため、
# スクリプトを起動し直すたびに ~2 分以上かかる初回コンパイルが毎回走る。
# jax はコンパイル結果をディスクにも永続化できる (jax.jit の入力形状・
# 制御フローが同じであれば、次回起動時はディスクのキャッシュを読むだけで
# 済む) ので、jax を import する前にキャッシュ先を環境変数で指定しておく
# (jax_backend.py が Darwin 判定を jax import 前の環境変数で行っているのと
# 同じ要領。既に設定済みならユーザーの指定を優先し上書きしない)。
os.environ.setdefault(
    'JAX_COMPILATION_CACHE_DIR',
    os.path.expanduser('~/.cache/jax_compilation_cache'))
# 既定はコンパイルに 1 秒以上かかった計算しかディスクに保存しないが、この
# スクリプトが使う干渉回避付きバッチ IK のコンパイルは 1 回で数分かかる
# ヘビーなものだけなので、閾値を 0 にしても実害はなく取りこぼしを防げる。
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import matrix2ypr, normalize_mask  # noqa: E402
from skrobot.model import RobotModel  # noqa: E402
from skrobot.model.primitives import Cylinder  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.planner.trajectory_optimization.collision import (  # noqa: E402
    create_self_collision_pairs)

from view_aero_collision_model import build_collision_model_urdf  # noqa: E402

# 掌のローカル +Y (甲->掌方向) まわりにこの角度ずつ向きをずらした候補を
# 順に試し、IK が解けた最初のものを採用する。0 度 (掌の向きをそのまま
# 使う) を最初に試し、それで解けなければ ±90 度を試す (平面フィットの
# 誤差次第で、特定の向きのままだと IK が解けないことがあるため)。
TURN_CANDIDATES_DEG = (0.0, 90.0, -90.0)

# 人間の手 (掌 JSON の ``offered_hand``) に対して既定で使うロボットの腕
# (``--robot-arm auto``)。向かい合う握手ではなく、人間と同じ方向を向いて
# 反対側の手で繋ぐ想定なので、人の手とは反対側の腕を使う。
DEFAULT_ROBOT_ARM = {'L': 'r', 'R': 'l'}

# Aero は常にワールド原点で IK を開始する (``seed_arm_pose`` 参照) ため、
# 人物側をこの距離だけ Aero の前方 (+x) に平行移動してから IK を解く
# (``human_translation_offset`` 参照)。3m は干渉回避に最低限必要な距離
# (体格 0.3m 程度 + 余裕) よりも十分大きく、Aero が向き合う相手として
# 現実的な距離でもある。
HUMAN_FRONT_DISTANCE = 3.0  # [m]

# 台車の既定の移動可能領域の半幅 (人物の立ち位置を中心とした前後左右の
# 距離)。人物は常に (``HUMAN_FRONT_DISTANCE``, 0) に固定されるので、
# ``DEFAULT_BASE_X_RANGE``/``DEFAULT_BASE_Y_RANGE`` はこの値からそのまま
# 計算できる。
BASE_MOVABLE_HALF_RANGE = 5.0  # [m]

# 台車 (use_base='planar' の仮想関節) の既定の移動範囲。IK 開始時の台車
# 位置 (常にワールド原点, ``seed_arm_pose`` 参照) を基準にした [x, y, yaw]
# の (下限, 上限)。乱数初期値もこの範囲から引かれる。人物の立ち位置
# (``HUMAN_FRONT_DISTANCE``, 0) を中心に前後左右 ``BASE_MOVABLE_HALF_
# RANGE`` m の正方形にしてある。
DEFAULT_BASE_X_RANGE = (HUMAN_FRONT_DISTANCE - BASE_MOVABLE_HALF_RANGE,
                        HUMAN_FRONT_DISTANCE + BASE_MOVABLE_HALF_RANGE)
DEFAULT_BASE_Y_RANGE = (-BASE_MOVABLE_HALF_RANGE, BASE_MOVABLE_HALF_RANGE)
DEFAULT_BASE_YAW_RANGE = (-math.pi / 2.0, math.pi / 2.0)

# 1 目標姿勢あたりに振る初期値の数 (バッチ IK の attempts_per_pose)。
# 初期値 0 は seed_arm_pose の種の姿勢 (台車はワールド原点)、残りは関節
# 範囲・台車の可動範囲の一様乱数。初期値ごとの解は集約させずに全て受け取る
# (``return_all_attempts``。``solve_person_ik`` 参照)。
DEFAULT_ATTEMPTS_PER_POSE = 64

# 干渉回避 (collision_obstacles) 付きバッチ IK の収束判定。勾配降下法は
# 通常のヤコビアン法より収束が遅いため、既定よりも反復回数を増やし閾値を
# ゆるめている。
DEFAULT_COLLISION_IK_STOP = 500
DEFAULT_COLLISION_IK_THRE = 0.03  # [m]
DEFAULT_COLLISION_IK_RTHRE = math.radians(8.0)  # [rad]

# 干渉回避ペナルティの重み・マージン (skrobot の batch_inverse_kinematics
# の既定値と同じ)。
DEFAULT_COLLISION_WEIGHT = 10.0
DEFAULT_COLLISION_MARGIN = 0.05

# 自己干渉 (ロボット自身のリンク同士の干渉) 回避ペナルティのマージン
# (skrobot の batch_inverse_kinematics の既定値と同じ)。重みは既定で
# ``collision_weight`` (人体との干渉回避と同じ重み) を使う
# (``self_collision_weight=None`` のときの skrobot 側の既定挙動)。
DEFAULT_SELF_COLLISION_MARGIN = 0.02

# IK のターゲットを掌からどれだけ浮かせるか (法線方向) [m]。目標そのものが
# 人体の干渉回避ジオメトリと重ならないようにするためのオフセット。
TARGET_HOVER_OFFSET = 0.08  # [m]

# 事後の後処理判定 (``solve_post_process``) で使う、掌からのオフセット
# [m]。負の値 (掌の内側にわずかにめり込む位置) にすることで、実際に掌へ
# 手を近づけられる (押し付けられる) 姿勢が取れるかを検証する。
POST_PROCESS_TARGET_HOVER_OFFSET = -0.01  # [m]

# 後処理判定の腕 IK (通常のヤコビアン法, 干渉回避なし・台車なし) の最大
# 反復回数。干渉回避付きバッチ IK が採用した姿勢からの微小な追い込みなので
# 収束は速いはずだが、余裕を持った値にしておく。
DEFAULT_POST_PROCESS_IK_STOP = 200

# 後処理判定の腕 IK の収束閾値 (位置[m]/姿勢[rad])。干渉回避付きバッチ IK
# (事前段階) 自体が ``DEFAULT_COLLISION_IK_THRE``/``DEFAULT_COLLISION_IK_
# RTHRE`` というずっと緩い閾値で「収束」と判定した候補を初期値に使うため、
# skrobot 既定の厳しい基準 (位置 1mm・姿勢 1度) では後処理が失敗しやすく、
# それに合わせて緩めてある。
DEFAULT_POST_PROCESS_IK_THRE = 0.025  # [m]
DEFAULT_POST_PROCESS_IK_RTHRE = math.radians(5.0)  # [rad]

# 後処理判定の視線 IK (人間の手を見る首 3 関節) の最大反復回数・収束閾値
# (姿勢のみ, ``rotation_mask='xy'`` で視線軸まわりの回転は見ない)。
# ``RobotModel.look_at`` はこれらを外部から指定する手段が無く、内部で
# 呼ぶ ``inverse_kinematics_loop_for_look_at`` もキーワード引数
# ``stop`` を実質無視して (中身で使われず) skrobot 既定の 50 回に、
# ``rthre`` も既定 0.001rad (約 0.057 度) という極めて厳しい値になる
# ため、必要な首振り角度が数十度あるような (人体の干渉回避で台車・肩が
# 大きく振られた) 姿勢からはまず収束しない。この後処理も
# ``solve_post_process`` と同じく合否ではなくラフな確認なので、
# ``look_at`` を経由せず ``robot.inverse_kinematics`` (腕の押し付け目標と
# 同じ呼び出しに ``move_target``/``target_coords`` 等をリストで渡す) を
# 直接使い、反復回数・閾値をここで明示的に緩める。
DEFAULT_POST_PROCESS_GAZE_IK_STOP = 300
DEFAULT_POST_PROCESS_GAZE_IK_RTHRE = math.radians(5.0)  # [rad]

# 肘関節 (``{r,l}_elbow_joint``) の可動域制限 [deg]。この関節は 0 度が
# 腕をまっすぐ伸ばした状態、負方向が肘を曲げる方向で、URDF 上の可動域は
# 0 度 〜 -172 度弱 (機械的なストッパーの手前で前腕がほぼ上腕と平行に
# 折り畳まれる直前まで許容している)。この可動域いっぱいまで曲げた IK 解は
# 前腕と上腕が平行に近づく不自然な姿勢になりやすいため、IK を解く前に
# 下限だけこの値まで緩め、そこまでは曲がらないようにする
# (``restrict_elbow_range`` 参照)。上限 (0 度, まっすぐ) は変更しない。
ELBOW_MIN_ANGLE_DEG = -120.0

# 干渉回避付きバッチ IK (``solve_person_ik``) だけに適用する、関節可動域の
# 上下マージン比率。可動域の上限・下限それぞれこの割合だけ内側に制限した
# 範囲で解かせる (``restrict_joint_range_margin`` 参照)。干渉回避のペナル
# ティに引っ張られて関節が可動域の端に張り付いた解になりやすく、それが
# 後段の後処理 IK (``solve_post_process``) の収束を妨げる一因になるため、
# あらかじめ端を使用禁止にしておく。``solve_person_ik`` 内のバッチ IK
# 呼び出しにのみ適用し、その前後で元の可動域に戻す -- 後処理・事後の干渉
# 検証は本来の可動域全体を使ってよい。
DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO = 0.1

# 人体の干渉回避用ジオメトリ: (骨格の関節名 A, 関節名 B, 半径[m]) の
# タプルの並び。BODY_JOINT_NAMES (generate_random_human_poses.py) の
# 関節を結ぶ主要な骨を、成人の平均的な太さを目安にした半径の円柱
# (skrobot 内部では両端に半球を持つ Capsule として扱われる) で近似する。
# 差し出している側の腕も含めて全身を障害物にする (``TARGET_HOVER_OFFSET``
# により目標そのものとは重ならないので、除外は不要)。
HUMAN_COLLISION_SEGMENTS = (
    ('Nose', 'Neck', 0.10),
    ('Neck', 'RShoulder', 0.09),
    ('Neck', 'LShoulder', 0.09),
    ('RShoulder', 'RElbow', 0.06),
    ('LShoulder', 'LElbow', 0.06),
    ('RElbow', 'RWrist', 0.04),
    ('LElbow', 'LWrist', 0.04),
    ('Neck', 'RHip', 0.13),
    ('Neck', 'LHip', 0.13),
    ('RHip', 'LHip', 0.13),
    ('RHip', 'RKnee', 0.09),
    ('RKnee', 'RAnkle', 0.06),
    ('LHip', 'LKnee', 0.09),
    ('LKnee', 'LAnkle', 0.06),
)

# 手 (指先まで) の干渉回避用ジオメトリ。骨格の関節位置は手首までしか
# 無く、指の分だけ実際の手はそこから先に伸びているので、``HUMAN_
# COLLISION_SEGMENTS`` の前腕の円柱だけでは手の体積を近似できない (手首
# より先の指が障害物なしにすり抜けてしまう)。``generate_random_human_
# poses.py`` が骨格と一緒に出す MediaPipe 形式の手のランドマーク
# (``{R,L}Hand0``..``{R,L}Hand20``, 0 が手首, 1-4/5-8/9-12/13-16/17-20 が
# それぞれ親指/人差し指/中指/薬指/小指) を使い、掌は手首と 4 本の指の
# 付け根 (MCP) を囲む平たい ``Cylinder``、各指は付け根から指先までを結ぶ
# 細い ``Cylinder`` で近似する (差し出している側も含む)。
HAND_PALM_LANDMARKS = (0, 5, 9, 13, 17)  # 手首 + 4 指の付け根 (MCP)
HAND_FINGER_LANDMARKS = (
    (1, 4),    # 親指: CMC -> 指先
    (5, 8),    # 人差し指: MCP -> 指先
    (9, 12),   # 中指: MCP -> 指先
    (13, 16),  # 薬指: MCP -> 指先
    (17, 20),  # 小指: MCP -> 指先
)
# ``HAND_FINGER_LANDMARKS`` の各指に対応する名前 (``--collision-pairs``
# JSON で人体側のオブジェクトを指すのに使う、``human_obstacle_names``
# 参照)。``analyze_collision_pairs.py`` の ``_FINGER_LABELS`` と同じ順序。
HAND_FINGER_LABELS = ('thumb', 'index', 'middle', 'ring', 'pinky')
HAND_PALM_RADIUS = 0.05  # [m] 掌の円柱の半径
HAND_PALM_HEIGHT = 0.02  # [m] 掌の円柱の厚み (平たくする)
HAND_FINGER_RADIUS = 0.008  # [m] 指の円柱の半径 (細くする)

# ``human_body_obstacles`` が返す障害物の個数を人物によらず常に固定にする
# ため (JAX の再コンパイルを避けるため)、骨格の関節が欠けている骨・手を
# 埋めるダミー障害物をこの距離だけ離れた位置に置く [m] (干渉回避のコストに
# 影響しない十分な距離)。
DUMMY_OBSTACLE_DISTANCE = 100.0  # [m]


def _turn_about_y(rot, turn_deg):
    """``rot`` の局所 +X/+Z を、+Y 軸まわりに ``turn_deg`` 度だけ回す
    (+Y はそのまま)。"""
    x_axis, y_axis, z_axis = rot[:, 0], rot[:, 1], rot[:, 2]
    phi = math.radians(turn_deg)
    turned_x = math.cos(phi) * x_axis + math.sin(phi) * z_axis
    turned_z = -math.sin(phi) * x_axis + math.cos(phi) * z_axis
    return np.column_stack([turned_x, y_axis, turned_z])


def _correct_grasp_frame(rot, arm):
    """左腕用に +Y/+Z を反転する.

    ``l_eef_grasp_link`` は ``r_eef_grasp_link`` に対して +X (指方向)
    まわりに 180 度ずれている (URDF が左右ミラーで作られているため) ので、
    右腕用に組んだ ``rot`` を左腕で使うにはこの補正が要る。
    """
    if arm != 'l':
        return rot
    return np.column_stack([rot[:, 0], -rot[:, 1], -rot[:, 2]])


def palm_to_target_rots(palm, robot_arm):
    """掌の位置姿勢 JSON (``estimate_palm_poses.PalmPoseEstimator`` の
    出力の 1 手分) から、ロボットの手先座標系 (``{arm}_eef_grasp_link``,
    +X=指方向, +Y=甲->掌方向, +Z=+X×+Y) で表した目標姿勢の候補群
    (``TURN_CANDIDATES_DEG`` の数だけ) を返す。

    向かい合う握手ではなく、人間と同じ方向を向いて反対側の手で繋ぐ想定
    なので、指方向 (+X) は鏡写しにせずそのまま使う。掌の法線 (``y_axis``, 手の甲
    ->掌方向, 体の外側を向く) に対し、ロボットの掌は人間の掌に正対する
    向きにしたいので、ロボットの +Y は ``-y_axis``。
    """
    x_axis = np.asarray(palm['x_axis'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    y_axis = -normal
    z_axis = np.cross(x_axis, y_axis)
    base_rot = np.column_stack([x_axis, y_axis, z_axis])
    return [_correct_grasp_frame(_turn_about_y(base_rot, deg), robot_arm)
           for deg in TURN_CANDIDATES_DEG]


def palm_target_position(palm):
    """掌から ``TARGET_HOVER_OFFSET`` だけ浮かせた位置 (法線方向) を IK の
    目標位置として返す。全身を干渉回避の対象にできるよう、実機の初期
    アプローチ用オフセット (``palm_plane.CONTACT_OFFSET``) よりも大きく
    浮かせてある (``TARGET_HOVER_OFFSET`` の注記を参照)。"""
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    return position + normal * TARGET_HOVER_OFFSET


def human_standing_xy(joint_positions):
    """人物の立ち位置 (x, y) の目安を骨格の関節位置から求める.

    骨盤 (``RHip``/``LHip`` の中点) を優先する。``generate_random_human_
    poses.py`` が生成する人物はこの骨盤がほぼワールド原点に置かれる
    (``root_pos`` 参照) ので、``human_translation_offset`` の平行移動量の
    計算は実質この関節が無いと機能しない。腰が両方とも欠けていれば
    ``Neck``、それも無ければ ``None`` (平行移動は行わない) を返す。
    """
    hips = [joint_positions[name] for name in ('RHip', 'LHip')
           if name in joint_positions]
    if hips:
        xy = np.mean(np.asarray(hips, dtype=np.float64), axis=0)[:2]
    elif 'Neck' in joint_positions:
        xy = np.asarray(joint_positions['Neck'], dtype=np.float64)[:2]
    else:
        return None
    return xy


def human_translation_offset(joint_positions, front_distance=HUMAN_FRONT_DISTANCE):
    """人物をちょうど Aero の前方 ``front_distance`` [m] に置くための
    平行移動量 (dx, dy) を返す.

    Aero は常にワールド原点、向き +x で IK を開始する (``seed_arm_pose``
    参照) ので、人物の立ち位置 (``human_standing_xy``) が
    ``(front_distance, 0.0)`` にちょうど一致するように移動する量を返す。
    立ち位置が骨格から求まらない (``joint_positions`` に骨盤・首の関節が
    無い) ときは ``(0.0, 0.0)`` (平行移動なし) を返す -- この場合、その
    人物は Aero の前方に置かれない点に注意 (``main`` の呼び出し箇所参照)。
    """
    person_xy = human_standing_xy(joint_positions)
    if person_xy is None:
        return (0.0, 0.0)
    target_xy = np.array([front_distance, 0.0])
    offset = target_xy - person_xy
    return (float(offset[0]), float(offset[1]))


def translate_joint_positions(joint_positions, offset):
    """骨格の全関節位置 (``joint_positions``) を x/y 方向にだけ ``offset``
    平行移動したコピーを返す (z は身長方向なので変えない)。``offset`` が
    ``(0.0, 0.0)`` のときは ``joint_positions`` をそのまま返す。
    """
    dx, dy = offset
    if dx == 0.0 and dy == 0.0:
        return joint_positions
    translated = {}
    for name, pos in joint_positions.items():
        pos = np.array(pos, dtype=np.float64)
        pos[0] += dx
        pos[1] += dy
        translated[name] = pos.tolist()
    return translated


def translate_palm(palm, offset):
    """掌の位置姿勢 JSON の 1 手分 (``palm``) の ``position`` を x/y 方向に
    だけ ``offset`` 平行移動したコピーを返す。向き (``x_axis``/``y_axis``)
    は平行移動の影響を受けないのでそのまま。``offset`` が ``(0.0, 0.0)``
    のときは ``palm`` をそのまま返す。
    """
    dx, dy = offset
    if dx == 0.0 and dy == 0.0:
        return palm
    translated = dict(palm)
    position = list(palm['position'])
    position[0] += dx
    position[1] += dy
    translated['position'] = position
    return translated


def seed_arm_pose(robot, robot_arm):
    """バッチ IK の初期値 0 (attempt 0) に使う「種の姿勢」をロボットに作る.

    向かい合わず、人間と同じ方向を向いて手を繋ぐ構え (肩を横に開き、
    手首はひねらないニュートラルな姿勢)。種のままだと目標との姿勢差が
    大きすぎて IK が迷走しやすいため。台車は常にワールド原点に置く
    (``base_limits`` はこの原点を基準にした範囲になる)。人物側を
    ``HUMAN_FRONT_DISTANCE`` だけ Aero の前方に平行移動しておくことで
    (``human_translation_offset`` 参照)、台車を人物から離す必要が無くなる
    ため、台車の開始位置は常にこの原点で固定でよい。

    ``robot.reset_pose()`` は関節角度だけを戻し、台車の位置姿勢
    (``robot``/``base_link`` のワールド座標) は変えない。前の人物の
    ``solved_result``/``unsolved_result`` が ``robot.newcoords(base_pose)``
    で台車を動かしたままになっているので、ここで明示的にワールド原点
    (単位姿勢) へ戻す。Aero では ``root_link`` が ``base_link`` そのもの
    だが、``base_link.worldpos()`` は「親 (``robot``) の変換」×
    「``base_link`` のローカル変換」で決まるため、``robot`` と
    ``base_link`` の両方を単位姿勢に戻す必要がある (どちらか一方だけでは
    二重適用や中途半端な位置になる)。
    """
    robot.reset_pose()
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    mirror = 1.0 if robot_arm == 'l' else -1.0
    getattr(robot, '{}_shoulder_p_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_shoulder_r_joint'.format(robot_arm)) \
        .joint_angle(0.8 * mirror)
    getattr(robot, '{}_shoulder_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_elbow_joint'.format(robot_arm)).joint_angle(-1.0)
    getattr(robot, '{}_wrist_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_wrist_p_joint'.format(robot_arm)).joint_angle(0.087)
    getattr(robot, '{}_wrist_r_joint'.format(robot_arm)).joint_angle(0.0)


def restrict_elbow_range(robot, min_angle_deg=ELBOW_MIN_ANGLE_DEG):
    """``r_elbow_joint``/``l_elbow_joint`` の下限 (最大屈曲角) を
    ``min_angle_deg`` まで緩める (``ELBOW_MIN_ANGLE_DEG`` 参照)。

    ``batch_inverse_kinematics`` は各関節の ``min_angle``/``max_angle`` を
    呼び出しのたびに読み直して最適化の範囲・乱数初期値の範囲を決めるので、
    ここで書き換えておけば以降の全人物の IK に効く (``use_base='planar'``
    を渡す呼び出しはリンク単位の識別子でソルバをキャッシュしないため、
    毎回この制限が反映される)。``rarm_whole_body``/``larm_whole_body`` も
    ``robot`` と同じ ``Joint`` オブジェクトを共有しているので、``robot``
    側だけ書き換えれば両方に反映される。
    """
    min_angle = math.radians(min_angle_deg)
    for arm in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(arm)).min_angle = min_angle


def restrict_joint_range_margin(link_list, margin_ratio):
    """``link_list`` 中の各関節の可動域を、上下限それぞれ全域幅の
    ``margin_ratio`` だけ内側に一時的に制限する
    (``DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO`` 参照)。

    ``batch_inverse_kinematics`` は各関節の ``min_angle``/``max_angle`` を
    呼び出しのたびに読み直す (``restrict_elbow_range`` の docstring 参照)
    ため、この関数で書き換えてから呼べば効く。可動域が有限でない関節
    (``min_angle``/``max_angle`` が ``-inf``/``inf``。台車の仮想関節等)
    や、全域幅が非正の関節は対象外とする -- マージン比率を掛ける基準の
    「全域幅」が定義できない/意味を持たないため。

    Parameters
    ----------
    link_list : list[skrobot.model.Link]
        対象のリンク (``link.joint`` が対象の関節)。``batch_inverse_
        kinematics`` に渡す ``link_list`` と同じものを渡す想定。
    margin_ratio : float
        上下限それぞれ全域幅に掛けるマージンの比率 (0 以上 0.5 未満)。

    Returns
    -------
    callable
        呼び出すと各関節の可動域を書き換え前の値に戻す関数
        (``try``/``finally`` で呼び出し側が必ず呼ぶ想定)。
    """
    joints = []
    seen_ids = set()
    for link in link_list:
        joint = link.joint
        if joint is None or id(joint) in seen_ids:
            continue
        seen_ids.add(id(joint))
        joints.append(joint)

    originals = []
    for joint in joints:
        lo, hi = float(joint.min_angle), float(joint.max_angle)
        width = hi - lo
        if not np.isfinite(lo) or not np.isfinite(hi) or width <= 0:
            continue
        originals.append((joint, lo, hi))
        margin = width * margin_ratio
        joint.min_angle = lo + margin
        joint.max_angle = hi - margin

    def restore():
        for joint, lo, hi in originals:
            joint.min_angle = lo
            joint.max_angle = hi

    return restore


def _cylinder_between(p0, p1, radius):
    """``p0``-``p1`` を結ぶ線分を近似する ``Cylinder`` (骨格の 1 本の骨)
    を作る。円柱のローカル +Z が線分の向きになるよう回転させる。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    height = float(np.linalg.norm(diff))
    if height < 1e-6:
        # 関節位置がほぼ同一 (推定誤差等) のときに退化しないよう、
        # ごく小さい円柱にする。
        height = 1e-6
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = diff / height
    # z_axis とほぼ平行にならない適当な軸から正規直交基底を作る。
    seed = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return Cylinder(radius=radius, height=height,
                    pos=((p0 + p1) / 2.0).tolist(), rot=rot)


def _palm_obstacle(points):
    """掌を近似する平たい ``Cylinder`` を作る。``points`` は手首 + 4 指の
    付け根 (``HAND_PALM_LANDMARKS`` の順, MediaPipe 手ランドマーク) の
    座標。円柱の軸 (厚み方向) は掌面の法線 (手首->人差し指付け根,
    手首->小指付け根 の外積) にする。"""
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    wrist, index_mcp, pinky_mcp = points[0], points[1], points[-1]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    norm = np.linalg.norm(normal)
    z_axis = normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    # z_axis とほぼ平行にならない適当な軸から正規直交基底を作る
    # (円柱は軸まわり対称なので x/y の向きは何でもよい)。
    seed = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return Cylinder(radius=HAND_PALM_RADIUS, height=HAND_PALM_HEIGHT,
                    pos=center.tolist(), rot=rot)


def _dummy_cylinder(radius):
    """``DUMMY_OBSTACLE_DISTANCE`` 参照。骨・掌・指が欠けている場合に
    個数を揃えるためのダミー ``Cylinder`` (干渉回避のコストに影響しない
    十分遠い位置に置く。向き・高さは何でもよいので単位円柱にする)。"""
    return Cylinder(radius=radius, height=1e-3,
                    pos=[DUMMY_OBSTACLE_DISTANCE] * 3)


def human_body_obstacles(joint_positions):
    """骨格の関節位置 (``generate_random_human_poses.py`` の
    ``skeleton.joint_positions``) から、干渉回避の障害物として使う
    ``Cylinder`` のリストを作る (``HUMAN_COLLISION_SEGMENTS``/
    ``HAND_PALM_LANDMARKS``/``HAND_FINGER_LANDMARKS`` 参照)。差し出して
    いる側の腕・手も含め、全身を障害物にする (IK の目標を掌から浮かせて
    重ならないようにしているので、部位による除外は不要。
    ``TARGET_HOVER_OFFSET`` 参照)。

    常に ``len(HUMAN_COLLISION_SEGMENTS) + 2 * (1 + len(HAND_FINGER_
    LANDMARKS))`` 個 (骨格検出が全身分揃っているときの最大数, 手 1 つ
    あたり掌 1 個・指 5 個) を返す -- 関節位置が片方でも欠けている骨・
    掌・指は読み飛ばすのではなく、``DUMMY_OBSTACLE_DISTANCE`` だけ離れた
    ダミーの ``Cylinder`` で埋める (``DUMMY_OBSTACLE_DISTANCE`` の注記を
    参照。指は付け根から指先までを 1 本の円柱で近似する単純化も、この
    固定個数を小さく保つためのもの -- 指の関節ごとに円柱を分けると個数が
    大きく増え、干渉回避付きバッチ IK の JAX コンパイルがさらに重く
    なる)。"""
    obstacles = []
    for name_a, name_b, radius in HUMAN_COLLISION_SEGMENTS:
        if name_a in joint_positions and name_b in joint_positions:
            obstacles.append(_cylinder_between(
                joint_positions[name_a], joint_positions[name_b], radius))
        else:
            obstacles.append(_dummy_cylinder(radius))
    for side in ('R', 'L'):
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        if all(name in joint_positions for name in palm_names):
            obstacles.append(_palm_obstacle(
                [joint_positions[name] for name in palm_names]))
        else:
            obstacles.append(_dummy_cylinder(HAND_PALM_RADIUS))
        for base_idx, tip_idx in HAND_FINGER_LANDMARKS:
            base_name = '{}Hand{}'.format(side, base_idx)
            tip_name = '{}Hand{}'.format(side, tip_idx)
            if base_name in joint_positions and tip_name in joint_positions:
                obstacles.append(_cylinder_between(
                    joint_positions[base_name], joint_positions[tip_name],
                    HAND_FINGER_RADIUS))
            else:
                obstacles.append(_dummy_cylinder(HAND_FINGER_RADIUS))
    return obstacles


def human_obstacle_names():
    """``human_body_obstacles`` が返すリストと同じ順序・同じ個数の名前の
    リストを返す (``joint_positions`` の中身に依存しない構造だけの情報)。

    ``--collision-pairs`` (JSON) で人体側のオブジェクトを指定するときの
    名前、および ``analyze_collision_pairs.py`` が書き出す
    ``collision_pair_analysis.json`` の ``human_collision_min_dist`` の
    キー (``"{ロボットリンク名}|{この関数が返す名前}"``) の後半と対応する
    (``analyze_collision_pairs.human_capsules`` が同じ順序・同じ命名規則で
    作る名前のリストと一致させてある)。``load_collision_pairs`` がこの
    リストを使い、JSON 中の名前がロボットのリンク名でなければ人体
    セグメント名とみなして ``human_body_obstacles`` の出力中の対応する
    インデックスに解決する。
    """
    names = ['{}-{}'.format(name_a, name_b)
            for name_a, name_b, _ in HUMAN_COLLISION_SEGMENTS]
    for side in ('R', 'L'):
        names.append('{}_palm'.format(side))
        for label in HAND_FINGER_LABELS:
            names.append('{}_{}'.format(side, label))
    return names


def segment_points_distance(p0, p1, points):
    """線分 ``p0``-``p1`` と、複数の点 ``points`` (``(N, 3)``) それぞれとの
    最短距離 (``(N,)``)。``analyze_collision_pairs.py`` の集計処理と
    ``collision_pairs_min_distance`` (事後の厳密な干渉検証) の両方から
    使う共通処理なのでここに置く。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        t = np.zeros(len(points))
    else:
        t = np.clip((points - p0) @ d / denom, 0.0, 1.0)
    closest = p0 + t[:, np.newaxis] * d
    return np.linalg.norm(points - closest, axis=1)


def human_capsules(joint_positions):
    """``human_body_obstacles`` と同じ順序・同じ部位のカプセル (線分 2 端点
    + 半径) のリストと、対応する名前 (``human_obstacle_names`` と同じ) の
    リストを返す。骨格の関節が欠けている部位は ``DUMMY_OBSTACLE_DISTANCE``
    だけ離れた点に潰す (``human_body_obstacles`` のダミー障害物と同じ意味
    づけ)。``analyze_collision_pairs.py`` (干渉ペア候補の分析) と
    ``collision_pairs_min_distance`` (採用しようとしている IK 解の事後
    検証) の両方が、``human_body_obstacles`` が作る ``Cylinder`` の代わりに
    素の (線分, 半径) を使いたいときに使う。"""
    caps = []
    names = []
    dummy = np.array([DUMMY_OBSTACLE_DISTANCE] * 3)
    for name_a, name_b, radius in HUMAN_COLLISION_SEGMENTS:
        if name_a in joint_positions and name_b in joint_positions:
            p0 = np.asarray(joint_positions[name_a], dtype=np.float64)
            p1 = np.asarray(joint_positions[name_b], dtype=np.float64)
        else:
            p0 = p1 = dummy
        caps.append((p0, p1, radius))
        names.append('{}-{}'.format(name_a, name_b))
    for side in ('R', 'L'):
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        if all(name in joint_positions for name in palm_names):
            pts = np.array([joint_positions[name] for name in palm_names],
                           dtype=np.float64)
            center = pts.mean(axis=0)
        else:
            center = dummy
        caps.append((center, center, HAND_PALM_RADIUS))
        names.append('{}_palm'.format(side))
        for (base_idx, tip_idx), label in zip(
                HAND_FINGER_LANDMARKS, HAND_FINGER_LABELS):
            base_name = '{}Hand{}'.format(side, base_idx)
            tip_name = '{}Hand{}'.format(side, tip_idx)
            if base_name in joint_positions and tip_name in joint_positions:
                p0 = np.asarray(joint_positions[base_name], dtype=np.float64)
                p1 = np.asarray(joint_positions[tip_name], dtype=np.float64)
            else:
                p0 = p1 = dummy
            caps.append((p0, p1, HAND_FINGER_RADIUS))
            names.append('{}_{}'.format(side, label))
    return caps, names


# collision_pairs_min_distance が「実際には貫通していた」と判定する際の
# 許容誤差 [m] (プリミティブ形状のポリゴン近似誤差を吸収する程度の値)。
DEFAULT_COLLISION_VERIFY_TOLERANCE = 0.001  # [m]


def collision_pairs_min_distance(robot, collision_pairs, joint_positions):
    """``robot`` の現在の姿勢 (``angle_vector``/``base_pose`` 適用済み) で、
    ``collision_pairs`` (``load_collision_pairs`` が返す ``(Link, Link)``/
    ``(Link, int)`` 混在のリスト) の中で最も干渉している (最小の) 距離
    [m] を返す。負の値は貫通していることを意味する。``collision_pairs`` が
    空/``None`` のときは ``float('inf')`` を返す (検証対象なし)。

    ``analyze_collision_pairs.py`` が干渉ペア候補を洗い出すのに使ったのと
    同じ厳密な形状 (``apply_collision_model`` が差し替えた
    ``collision_mesh`` の頂点そのもの) を使って距離を計算する。
    """
    if not collision_pairs:
        return float('inf')
    caps = human_capsules(joint_positions)[0] if joint_positions else None
    min_dist = float('inf')
    world_vertices_by_link = {}

    def _world_vertices(link):
        if link not in world_vertices_by_link:
            local = np.asarray(link.collision_mesh.vertices, dtype=np.float64)
            world_vertices_by_link[link] = (
                local @ link.worldrot().T + link.worldpos())
        return world_vertices_by_link[link]

    for link_a, other in collision_pairs:
        verts_a = _world_vertices(link_a)
        if isinstance(other, int):
            if caps is None:
                # collision_obstacles が空 (骨格 JSON が無い) のときに
                # ここに来ることはない (solve_person_ik が effective_
                # collision_pairs で人体セグメント参照を除外済み) が、
                # 念のため検証をスキップする (呼び出し側の責務にしない)。
                continue
            p0, p1, radius = caps[other]
            dist = float(segment_points_distance(p0, p1, verts_a).min()) \
                - radius
        else:
            verts_b = _world_vertices(other)
            dist = float(np.linalg.norm(
                verts_a[:, np.newaxis, :] - verts_b[np.newaxis, :, :],
                axis=-1).min())
        if dist < min_dist:
            min_dist = dist
    return min_dist



def apply_collision_model(robot, primitive_type=None, force_convert=False,
                          collision_urdf_path=None):
    """``robot`` (実メッシュの Aero) の各リンクの ``collision_mesh`` を、
    ``view_aero_collision_model.py`` と同じ方法 (``skrobot.urdf.
    convert_meshes_to_primitives``) で生成したプリミティブ近似形状に
    差し替える。

    干渉回避 (``batch_inverse_kinematics`` の ``collision_link_list``) は
    各リンクの ``collision_mesh`` (``trimesh.Trimesh``) を最小外接円柱の
    球群に変換してコストを計算するため、実メッシュ (凹形状・高頂点数) の
    ままだと近似の質・計算コストの両面で不利になりやすい。box/cylinder/
    sphere に近似したプリミティブ形状に差し替えることで、``view_aero_
    collision_model.py`` で目視確認した干渉モデルと同じ形状を IK の干渉
    回避にも使う。

    ``build_collision_model_urdf`` がプリミティブ近似 URDF をファイルと
    してキャッシュする (既に生成済みならそれを再利用し、``force_convert``
    を指定したときだけ作り直す) ので、このスクリプトを繰り返し実行しても
    重い変換処理は初回のみで済む。

    差し替えは ``robot`` の各リンクを直接書き換えて行う (``RobotModel.
    load_urdf_file`` でロボット全体を作り直すのではない) ため、Aero
    クラスが提供する ``rarm_end_coords``/``rarm_whole_body`` などの
    キネマティクス関連の属性やジョイント名はそのまま使える。

    Parameters
    ----------
    collision_urdf_path : str or None
        指定すると、``build_collision_model_urdf`` によるプリミティブ近似
        URDF の自動生成・キャッシュを使わず、このパスの URDF から干渉
        ジオメトリを読み込む (``primitive_type``/``force_convert`` とは
        併用できない)。リンク名で ``robot`` 側と対応づけるので、ロボット
        本体 (IK を解く実際のキネマティクス) はそのままに、干渉回避
        (``collision_link_list_for_arm``) が使うジオメトリ・対象リンクの
        集合だけを差し替えられる -- 干渉計算を高速化する目的で手動で
        間引いた/単純化した独自の干渉用 URDF を使いたい場合などに使う。
        既定の自動生成モードと異なり、このモードでは指定した URDF に
        存在しない (対応するリンク名が見つからない、またはメッシュを
        持たない) リンクは干渉回避の対象から明示的に外す
        (``collision_mesh`` を ``None`` にする。``collision_link_list_
        for_arm`` はメッシュを持たないリンクを自動で除外する) ので、
        既定の自動生成モードでは残ってしまう実メッシュ (プリミティブ
        変換対象外・重い) のリンクが紛れ込まない。
    """
    if collision_urdf_path is not None:
        if primitive_type is not None or force_convert:
            raise ValueError(
                'collision_urdf_path (--collision-urdf) は '
                'primitive_type (--collision-primitive-type) / '
                'force_convert (--force-convert-collision-model) と '
                '併用できません。')
    else:
        collision_urdf_path = build_collision_model_urdf(
            robot.urdf_path, primitive_type=primitive_type,
            force=force_convert)

    collision_robot = RobotModel()
    collision_robot.load_urdf_file(
        str(collision_urdf_path), include_mimic_joints=False)
    collision_links_by_name = {
        link.name: link for link in collision_robot.link_list}

    # --collision-urdf 指定時は「このURDFが干渉回避の対象を規定する」
    # ものとして扱い、対応するリンクが見つからない/メッシュが無い場合は
    # 実メッシュを残さず明示的に None にして除外する (既定の自動生成
    # モードでは、常に robot 全リンクをカバーするプリミティブ URDF が
    # 生成されるので、この違いは表面化しない)。
    explicit_exclude = collision_urdf_path is not None

    n_replaced = 0
    n_excluded = 0
    for link in robot.link_list:
        collision_link = collision_links_by_name.get(link.name)
        mesh = (getattr(collision_link, 'collision_mesh', None)
               if collision_link is not None else None)
        if mesh is not None:
            link.collision_mesh = mesh
            n_replaced += 1
        elif explicit_exclude:
            link.collision_mesh = None
            n_excluded += 1
    print('[collision-model] {} 個のリンクの干渉ジオメトリを ({}) から '
          '差し替えました{}。'.format(
              n_replaced, collision_urdf_path,
              ' ({} 個のリンクを干渉回避の対象から除外)'.format(n_excluded)
              if explicit_exclude else ''))


def collision_link_list_for_arm(robot, robot_arm):
    """干渉ジオメトリを持つロボットリンクの一覧を作る。解く腕・反対側の
    腕・台車を含め、ロボットの全身 (``robot.link_list``) を対象にする
    (``robot_arm`` 引数は将来腕ごとに調整したくなった場合のための
    プレースホルダで、現状はどちらの腕でも同じリストを返す)。

    ``collision_mesh`` を持たない (中間フレーム・mimic ジョイント用の
    ダミーリンクなど、実体のジオメトリが無い) リンクは、原理的に何とも
    干渉し得ないので除外する。``apply_collision_model`` がプリミティブ
    形状に差し替えるのも ``collision_mesh`` を持つリンクだけ
    (``[collision-model] N 個のリンクの干渉ジオメトリを...`` のログの数と
    一致する) なので、この除外をしても干渉回避の判定結果は変わらない。

    干渉回避付きバッチ IK (``batch_inverse_kinematics``) 自体はこの関数を
    使わない -- 最適化でチェックする組み合わせは常に ``collision_pairs``
    (``load_collision_pairs`` が返す明示的なペアのリスト) だけで決まり、
    ``batch_inverse_kinematics`` の ``collision_link_list`` (全リンクの
    集合を丸ごと渡す引数) 自体を渡さない (勾配降下法の 1 反復ごとのコストを
    抑えるため、組み合わせを事前に絞り込む必要がある)。一方
    ``build_collision_verification_pairs`` (IK が収束した**後**の事後検証
    用。勾配降下法を回さず姿勢確定後に厳密な距離を計算するだけなので
    組み合わせを絞る必要がない) と ``analyze_collision_pairs.py`` (``
    collision_pairs.json`` を作るための素材集め) は、この関数で全リンクを
    集めてから総当たりの組み合わせを作る。各リンクの干渉ジオメトリは
    ``apply_collision_model`` によりプリミティブ近似形状に差し替え済みで
    あることを前提とする。
    """
    return [link for link in robot.link_list
           if getattr(link, 'collision_mesh', None) is not None]


def load_collision_pairs(path, robot):
    """``--collision-pairs`` (JSON) を読み、``batch_inverse_kinematics``
    の ``collision_pairs`` にそのまま渡せる ``(Link, Link)`` /
    ``(Link, int)`` のタプルのリストに変換する。

    JSON の形式は 2 要素のリストのリスト (``[[名前A, 名前B], ...]``)。
    各ペアの 1 要素目は常にロボットのリンク名。2 要素目は:

    * ロボットのリンク名なら自己干渉ペア (``(Link, Link)``) として扱う。
    * ``human_obstacle_names`` が返す人体セグメント名 (``human_body_
      obstacles`` が同じ順序で作る障害物のリストに対応する) なら、その
      セグメントとの干渉ペア (``(Link, int)``、int はそのセグメントの
      ``collision_obstacles`` 中のインデックス) として扱う。

    ``build_collision_pairs.py`` が ``analyze_collision_pairs.py`` の出力
    (``collision_pair_analysis.json``) の ``self_collision_min_dist``/
    ``human_collision_min_dist`` の両方からこの形式で生成する (キー
    ``"{リンクA}|{リンクB または人体セグメント名}"`` を分解したもの)。

    ``robot`` (``apply_collision_model`` 適用済みを想定) のリンク名と
    突き合わせる。どちらの解釈にも当てはまらない名前が含まれていた場合は
    ``ValueError`` にする -- 黙って無視すると、そのペアだけ干渉チェックから
    漏れたことに気付けないため。
    """
    with open(path) as f:
        pair_names = json.load(f)
    links_by_name = {link.name: link for link in robot.link_list}
    obstacle_index_by_name = {
        name: idx for idx, name in enumerate(human_obstacle_names())}
    pairs = []
    for name_a, name_b in pair_names:
        if name_a not in links_by_name:
            raise ValueError(
                '{} (--collision-pairs) に含まれるリンク名 {!r} が ロボット'
                'に見つかりません。'.format(path, name_a))
        link_a = links_by_name[name_a]
        if name_b in links_by_name:
            pairs.append((link_a, links_by_name[name_b]))
        elif name_b in obstacle_index_by_name:
            pairs.append((link_a, obstacle_index_by_name[name_b]))
        else:
            raise ValueError(
                '{} (--collision-pairs) に含まれる名前 {!r} が、ロボットの '
                'リンク名にも human_obstacle_names() の人体セグメント名にも '
                '一致しません。'.format(path, name_b))
    return pairs


def build_collision_verification_pairs(robot, robot_arm):
    """事後検証 (``pick_verified_candidate``/``collision_pairs_min_
    distance``) で使う、干渉しうる組み合わせを総当たりで網羅したペアの
    リストを作る (``(Link, Link)``/``(Link, int)`` 混在、``load_collision_
    pairs`` が返す形式と同じ)。

    ``--collision-pairs`` (JSON) は勾配降下法の 1 反復あたりのコストを
    抑えるために事前に絞り込んだ組み合わせだが、事後検証は姿勢確定後に
    厳密な距離を 1 回計算するだけなので絞り込む必要がなく、むしろ絞り込む
    と (JSON に無い組み合わせで) 実際に貫通していても見逃してしまう。その
    ため検証はここで作る全組み合わせに対して行う。

    * 自己干渉: ``collision_link_list_for_arm`` (干渉ジオメトリを持つ
      ロボット全リンク) 同士の全組み合わせから、``create_self_collision_
      pairs`` (``ignore_adjacent=True``。``analyze_collision_pairs.py`` が
      干渉ペア候補を洗い出すのに使っているのと同じ) が除く「間に他の
      リンクを挟まず隣接する (共通の関節で直接つながっている) 組み合わせ」
      を除いたもの。隣接リンクは可動範囲によらず幾何学的に常に接した
      ままなので、そもそも「干渉」として検出する意味が無い (常に非常に
      近い/接触した距離になり、閾値判定が無意味になる)。
    * 人体との干渉: 同じ ``collision_link_list_for_arm`` の各リンクと
      ``human_obstacle_names`` の全人体セグメントの組み合わせ (除外なし --
      差し出している側の腕を含め、人体側はロボットのどのリンクとも幾何
      学的には干渉しうるため)。

    ロボットの構造 (干渉ジオメトリを持つリンクの集合・隣接関係) だけで
    決まり、人物ごとの骨格やその時点の姿勢には依存しないので、``main``
    から人物ループの外で 1 回だけ計算すればよい。
    """
    collision_link_list = collision_link_list_for_arm(robot, robot_arm)
    self_pairs = create_self_collision_pairs(
        collision_link_list, ignore_adjacent=True)
    pairs = [(collision_link_list[link_i], collision_link_list[link_j])
            for link_i, link_j in self_pairs]
    for link in collision_link_list:
        for obstacle_index in range(len(human_obstacle_names())):
            pairs.append((link, obstacle_index))
    return pairs


def solve_post_process(robot, robot_arm, palm, target_rot,
                       offset=POST_PROCESS_TARGET_HOVER_OFFSET,
                       stop=DEFAULT_POST_PROCESS_IK_STOP,
                       thre=DEFAULT_POST_PROCESS_IK_THRE,
                       rthre=DEFAULT_POST_PROCESS_IK_RTHRE,
                       gaze_ik_stop=DEFAULT_POST_PROCESS_GAZE_IK_STOP,
                       gaze_ik_rthre=DEFAULT_POST_PROCESS_GAZE_IK_RTHRE):
    """``pick_verified_candidate`` が干渉検証まで通した候補について、
    人間にロボットが実際に掌を押し付けられることを確認する後処理判定を
    行う (``POST_PROCESS_TARGET_HOVER_OFFSET`` 参照)。

    台車を動かさない (``inverse_kinematics`` に ``use_base`` を渡さない
    既定の挙動、``rarm_whole_body``/``larm_whole_body`` は台車を含まない)
    通常のヤコビアン法 IK (干渉回避なし) で、以下の 2 つを ``robot.
    inverse_kinematics`` 1 回の呼び出しで同時に解く。``inverse_
    kinematics`` は ``move_target``/``target_coords``/``link_list``/
    ``position_mask``/``rotation_mask`` 等をそれぞれリストで渡すと、複数の
    エンドエフェクタを 1 つのヤコビアンにまとめて同時に解ける。

    1. 腕: 掌の目標位置を ``TARGET_HOVER_OFFSET`` (掌の上空 0.04m) の
       代わりに ``offset`` (既定 ``POST_PROCESS_TARGET_HOVER_OFFSET``、
       掌へわずかにめり込む位置) にした位置・姿勢 (``target_rot`` を
       ``rotation_mask=True`` で厳密に使う)。
    2. 首 (``robot.head.link_list``, 3 関節): 人間の差し出している手
       (掌, ``palm['position']``) の方向をロボットの頭部エンドエフェクタ
       (``robot.head_end_coords``) の +Z が向くように (``rotation_mask=
       'xy'`` で視線軸周りの回転は見ない、``align_axis_to_direction``
       参照)。位置は制約しない (``position_mask=False``)。

    視線の目標が人間自身の掌の位置 (IK を解く前から分かっている固定点) で
    あって、ロボット自身の手先の収束後の位置ではないため、腕の収束を待って
    から視線目標を作る (逐次に 2 回解く) 必要が無く、1 回の呼び出しで同時に
    解ける。``inverse_kinematics`` は全タスクが収束したときのみ成功を返す
    ため、腕・視線のどちらも収束して初めて後処理判定は成功とする。

    収束閾値 ``thre``/``rthre`` (腕) は skrobot 既定 (位置 1mm・姿勢 1度)
    より緩めた値 (``DEFAULT_POST_PROCESS_IK_THRE``/``DEFAULT_POST_PROCESS_
    IK_RTHRE``) を明示的に渡す -- 事前段階の干渉回避付きバッチ IK 自体が
    それよりずっと緩い閾値 (``DEFAULT_COLLISION_IK_THRE``/``DEFAULT_
    COLLISION_IK_RTHRE``, 2cm/8度) で「収束」と判定した候補を初期値として
    渡ってくるため、既定の厳しい閾値では実用上問題ない解でも収束できず
    後処理が失敗しやすい。視線 (位置は制約しないため ``thre`` は使われず、
    ``rthre`` だけが効く) は ``gaze_ik_rthre`` (既定 ``DEFAULT_POST_
    PROCESS_GAZE_IK_RTHRE``) を渡す -- skrobot 既定の ``rthre`` (0.001rad
    ≈ 0.057度) では、人体との干渉回避で肩や台車が大きく振られた姿勢から
    必要な首振り角度 (数十度) に対してまず収束しない。反復回数は、腕・視線
    の両タスクが同じループで一緒に反復される (タスクごとに分けられない)
    ため、それぞれの既定 (``stop``/``gaze_ik_stop``) の大きい方を呼び出し
    全体の反復回数として使う -- 先に収束したタスクは誤差 0 のまま、もう
    一方の収束を待つだけなので安全。

    ``robot`` は呼び出し前の姿勢 (``pick_verified_candidate`` が反映した
    干渉検証済みの候補の姿勢、台車位置を含む) を初期値として直接書き換え
    て解く。``revert_if_fail=True`` で解くので、判定に失敗した場合
    ``robot`` は呼び出し前の姿勢に戻る。

    Returns
    -------
    dict or None
        腕 IK・視線 IK のどちらも収束したときは、後処理後の手先・台車・
        関節角を持つ結果 dict (``solved_result`` と同じキー構成の一部)。
        どちらか一方でも収束しなければ ``None``。
    """
    start_time = time.time()
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    target_pos = position + normal * offset
    target_coords = Coordinates(pos=target_pos.tolist(), rot=target_rot)

    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    head_move_target = robot.head_end_coords
    head_target = Coordinates(
        pos=position.tolist()).align_axis_to_direction(
            position - head_move_target.worldpos())

    try:
        # position_mask/rotation_mask はタスクごと (腕/首) に別のマスクを
        # 使いたいので、ここで normalize_mask により 3 要素配列へ解決済みの
        # リストにしてから渡す -- robot_model._resolve_mask_params は
        # 「既に解決済みの 3 要素配列のリスト」だけをタスクごとのマスクの
        # リストとして認識し、そうでない (bool/str が直接並んだ) リストは
        # 1 つのマスク指定そのものとして誤って解釈してしまうため。
        result = robot.inverse_kinematics(
            target_coords=[target_coords, head_target],
            move_target=[move_target, head_move_target],
            link_list=[whole_body.link_list, robot.head.link_list],
            position_mask=[normalize_mask(True), normalize_mask(False)],
            rotation_mask=[normalize_mask(True), normalize_mask('xy')],
            stop=max(stop, gaze_ik_stop),
            thre=[thre, thre], rthre=[rthre, gaze_ik_rthre],
            revert_if_fail=True)
    except Exception as e:
        print('  [post-process] 押し付け/視線 IK で例外が発生したため '
              '棄却します: {}'.format(e))
        return None
    if result is False:
        return None

    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    return dict(
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in move_target.worldpos()],
        hand_rot=[[float(v) for v in row] for row in move_target.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
        compute_time=time.time() - start_time,
    )


def pick_verified_candidate(robot, success_flags, angle_vectors, base_poses,
                            verification_pairs, joint_positions,
                            collision_verify_tolerance,
                            robot_arm, palm, rots,
                            attempts_per_pose=DEFAULT_ATTEMPTS_PER_POSE,
                            post_process_ik_stop=DEFAULT_POST_PROCESS_IK_STOP):
    """``batch_inverse_kinematics`` が返した候補群 (``success_flags``/
    ``angle_vectors``/``base_poses``。全て同じ添字で対応する) の中から、
    以下を全て満たす候補を、添字最小 (最優先) のものから探して返す。

    候補は「向き (``TURN_CANDIDATES_DEG``) × 初期値 (``attempts_per_
    pose``)」の全組み合わせで、添字は向き優先の並び (向き 0 の初期値
    0..N-1, 向き 1 の初期値 0..N-1, ...。``batch_inverse_kinematics`` に
    ``return_all_attempts=True`` を渡したときの並び) になっている
    -- つまり ``向きの添字 = 添字 // attempts_per_pose``。同じ向きでも
    初期値が違えば別の解になりうるので、初期値ごとの解を 1 つに集約せず
    全てここで検証する (モジュール docstring 参照)。優先順位は「向きが
    早いもの」→「その向きの中で初期値が早いもの (初期値 0 = 種の姿勢が
    最優先)」。

    1. IK が収束している (``success_flags``)。
    2. ``verification_pairs`` を実際には貫通していない
       (``collision_pairs_min_distance`` による事後検証、
       ``collision_verify_tolerance`` [m] まで許容, ``solve_person_ik`` の
       docstring 参照 -- IK の収束判定は干渉ペナルティの残差を見ないため、
       この事後検証が別途必要)。
    3. 後処理判定 (``solve_post_process``: 台車を動かさない干渉なしの腕
       IK で掌に押し付けられ、かつロボットが自分の手を見られるか) にも
       成功している。

    ``verification_pairs`` には最適化で使った (絞り込み済みの)
    ``collision_pairs`` ではなく、``build_collision_verification_pairs``
    が作る総当たりの組み合わせ (``solve_person_ik`` 側で ``collision_
    obstacles`` の有無に応じてフィルタ済みのもの) を渡す想定。

    1・2 を満たすが 3 (後処理判定) に失敗した候補は棄却し、次の添字
    (同じ向きの次の初期値の解、それも尽きれば次の向き) を試す -- 干渉
    回避付きバッチ IK (``solve_person_ik``) 自体は既に全ての向き × 全ての
    初期値を 1 回のバッチで解いてあるので、ここでやり直すのは事後の干渉
    検証・後処理判定だけで済む。
    1・2・3 を全て満たす候補が見つからなかった場合のみ、1・2 を満たした
    最初の候補 (``fallback``) を後処理前のまま (``post_process_result`` を
    ``None`` にして) 採用するフォールバックを行う -- ``TARGET_HOVER_
    OFFSET`` の目標に干渉なく届いている後処理前の解自体は後処理とは独立に
    有効な解であり、全滅させて ``unsolved`` にしてしまうより、viewer で
    「後処理前の姿勢」として表示できる解を残す方が有用なため。

    候補を検証するにはロボットにその候補の姿勢を反映する必要がある
    (``collision_pairs_min_distance``/``solve_post_process`` はロボットの
    現在の姿勢を見る/書き換える) ため、検証のたびに ``robot`` を書き換える
    -- 呼び出し後の ``robot`` は最後に調べた候補の姿勢のままになる点に
    注意 (後処理判定に成功して即採用した場合を除き、これは呼び出し側が
    最終的に採用する候補の姿勢と一致しない可能性がある -- ``solved_
    result``/``unsolved_result`` は ``angle_vector``/``base_pose`` を
    改めて反映し直すので、最終的な姿勢を保証するのはそちら)。

    Returns
    -------
    tuple or None
        ``(turn_index, angle_vector, base_pose, post_process_result)``。
        ``turn_index`` は採用した候補の**向き**の添字 (``TURN_CANDIDATES_
        DEG``/``rots`` の添字。候補そのものの添字ではない -- 呼び出し側が
        必要とするのは目標姿勢を決めた向きだけのため)。
        ``post_process_result`` は ``solve_post_process`` が返した後処理後
        の結果 dict、後処理判定に失敗した候補をフォールバックで採用した
        場合は ``None``。1・2 を満たす候補が 1 つも無ければ ``None`` を
        返す。
    """
    fallback = None
    for candidate_index, ok in enumerate(success_flags):
        if not ok:
            continue
        turn_index = candidate_index // attempts_per_pose
        attempt_index = candidate_index % attempts_per_pose
        label = 'turn={:.0f}deg/初期値 {}'.format(
            TURN_CANDIDATES_DEG[turn_index], attempt_index)
        robot.angle_vector(angle_vectors[candidate_index])
        robot.newcoords(base_poses[candidate_index])
        min_dist = collision_pairs_min_distance(
            robot, verification_pairs, joint_positions)
        if min_dist < -collision_verify_tolerance:
            print('  [collision-verify] {} の候補は IK は収束'
                  'したが、事後検証で {:.4f} m 貫通していたため棄却'
                  'します。'.format(label, min_dist))
            continue
        post_result = solve_post_process(
            robot, robot_arm, palm, rots[turn_index],
            stop=post_process_ik_stop)
        if post_result is not None:
            return (turn_index, angle_vectors[candidate_index],
                    base_poses[candidate_index], post_result)
        print('  [post-process] {} の候補は干渉検証を通過した '
              'が、後処理判定 (押し付け/視線 IK) には失敗したため、次の '
              '候補を試します。'.format(label))
        if fallback is None:
            fallback = (turn_index, angle_vectors[candidate_index],
                       base_poses[candidate_index], None)
    if fallback is not None:
        print('  [post-process] 全ての候補で後処理判定に失敗した '
              'ため、turn={:.0f}deg の候補を後処理前の解として採用 '
              'します。'.format(TURN_CANDIDATES_DEG[fallback[0]]))
    return fallback


def solve_person_ik(robot, palm, robot_arm, collision_obstacles,
                    attempts_per_pose=DEFAULT_ATTEMPTS_PER_POSE,
                    base_limits=None,
                    collision_weight=DEFAULT_COLLISION_WEIGHT,
                    collision_margin=DEFAULT_COLLISION_MARGIN,
                    self_collision=True,
                    collision_pairs=None,
                    self_collision_weight=None,
                    self_collision_margin=DEFAULT_SELF_COLLISION_MARGIN,
                    collision_ik_stop=DEFAULT_COLLISION_IK_STOP,
                    collision_ik_thre=DEFAULT_COLLISION_IK_THRE,
                    collision_ik_rthre=DEFAULT_COLLISION_IK_RTHRE,
                    collision_joint_limit_margin_ratio=(
                        DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO),
                    joint_positions=None,
                    verification_pairs=None,
                    collision_verify_tolerance=(
                        DEFAULT_COLLISION_VERIFY_TOLERANCE)):
    """1 人分について、``TURN_CANDIDATES_DEG`` の全ての向き × 全ての初期値
    (``attempts_per_pose`` 個) を、その人の身体
    (``collision_obstacles``) を障害物とした干渉回避付きバッチ IK で
    まとめて解く。``self_collision=True`` (既定) のときは、それに加えて
    ロボット自身のリンク同士の干渉 (例えば解いている腕が胴体・反対側の腕・
    台車にぶつかる) もソフトなペナルティとして回避する。

    チェックする組み合わせ (自己干渉のロボットリンク同士、人体との干渉の
    ロボットリンク×人体セグメント) は ``collision_pairs`` (``load_
    collision_pairs`` が返す ``(Link, Link)``/``(Link, int)`` 混在の
    リスト) で明示的に指定したものだけに限る -- ``collision_link_list``
    という「対象リンクの集合」は持たず (``batch_inverse_kinematics`` には
    渡さない)、``collision_pairs`` に現れるリンクだけが干渉ジオメトリの
    計算対象になる。``main`` は ``collision_pairs`` が ``None`` (``--
    collision-pairs`` に指定した JSON が無い) のときに ``self_collision``/
    ``collision_obstacles`` の両方を無効にしてこの関数を呼ぶので、ここで
    「全組み合わせにフォールバックする」ことはない -- 干渉回避を行うなら
    必ず ``collision_pairs`` を経由する。

    ``collision_ik_stop`` (最大反復回数) は、干渉回避付きバッチ IK
    (``backend='jax'`` の勾配降下法) が ``lax.fori_loop`` で固定回数
    律儀に反復する実装であるため、壁時計時間のタイムアウトを別途設けても
    「非現実的な姿勢だけを狙って弾く」効果は無く (収束の有無によらず
    1 人あたりの計算時間はほぼ一定になるため)、実質これを減らすのと
    同じ意味になる。反復回数を絞ることは、閾値ぎりぎりまで反復してようやく
    収束するような (=無理のある/不自然な配置になりがちな) 解を「収束しな
    かった」として弾く効果があるため、暫定的な「タイムアウト」としては
    こちらを調整する方が理にかなっている。ただし絞りすぎると自然な姿勢も
    収束前に弾かれてしまうので、``--attempts-per-pose`` とのトレードオフに
    なる。

    ``collision_obstacles`` はバッチ呼び出し全体で 1 つの集合しか渡せない
    (skrobot 側の制約) ため、人物をまたいでバッチをまとめることはできず、
    1 人ごとに 1 回呼ぶ。干渉回避は ``backend='jax'`` の勾配降下法でしか
    使えないので固定する。``robot`` の腕は ``seed_arm_pose`` で種の姿勢に
    してから呼ぶ (台車は常にワールド原点から開始する。``palm`` は
    ``main`` 側で ``translate_palm`` により、人物が Aero の前方
    ``HUMAN_FRONT_DISTANCE`` に来るよう平行移動済みのものを渡す想定)。
    バッチ IK 自体はロボットを動かさないので、戻り値は「解を反映
    するための材料」であり、``robot`` は呼び出し後も種の姿勢のまま。

    ``success_flags`` (収束判定) は位置・姿勢誤差だけを見ており、
    干渉ペナルティが実際に解消しているかは保証しない (``collision_pairs_
    min_distance`` の docstring 参照)。そのため収束した候補ごとに、採用
    する前に ``collision_pairs_min_distance`` で厳密な形状を使った事後
    検証を行い、``collision_verify_tolerance`` を超えて貫通している候補は
    棄却して次の候補を試す -- 「IK が収束したかどうか」と「実際に干渉して
    いないか」は独立な条件として両方満たす候補だけを採用する。この事後
    検証は最適化で使った ``collision_pairs`` (勾配降下法のコストを抑える
    ために絞り込んだ組み合わせ) ではなく、``verification_pairs``
    (``build_collision_verification_pairs`` が作る、隣接リンクの組み合わせ
    だけを除いた総当たりの組み合わせ) に対して行う -- ``collision_pairs``
    に無い組み合わせで実際には貫通している解を見逃さないため。
    ``joint_positions`` (``main`` 側で平行移動済みのものを渡す想定) は
    人体セグメントとの干渉検証に使う (``None`` なら人体との干渉は検証
    しない -- 骨格 JSON が無い場合。この場合 ``effective_verification_
    pairs`` はそもそも人体セグメント参照を含まない)。

    ``collision_joint_limit_margin_ratio`` (既定 ``DEFAULT_COLLISION_IK_
    JOINT_LIMIT_MARGIN_RATIO``) は、この関数内の干渉回避付きバッチ IK
    だけに適用する関節可動域の上下マージン比率 (``restrict_joint_range_
    margin`` 参照)。干渉回避のペナルティに引っ張られて複数の関節が可動域
    の端に張り付いた解になりやすく、それが後段の ``solve_post_process``
    の収束を妨げる一因になっているため、あらかじめ上下限それぞれこの
    比率だけ内側を使用禁止にして解かせる。バッチ IK 呼び出しの前後だけで
    適用・復元するので、``pick_verified_candidate`` 以降 (事後の干渉検証・
    後処理 IK) は本来の可動域のまま使われる。

    Returns
    -------
    tuple
        ``(picked, collision_ik_time)``。``picked`` は
        ``(turn_index, angle_vector, base_pose, post_process_result)``
        または ``None`` -- 「向き (``TURN_CANDIDATES_DEG``) × 初期値
        (``attempts_per_pose``)」の全候補の中で最初 (向きが早い順、同じ
        向きなら初期値が早い順) に、収束・事後の干渉検証・後処理判定の
        全てを満たしたものを返す (``pick_verified_candidate`` 参照。
        ``turn_index`` はその候補の向きの添字)。後処理判定に失敗した
        候補は棄却して次の候補を試す。どの候補も後処理判定に成功しなかった
        場合のみ、収束・事後の干渉検証を満たした最初の候補を後処理前の
        まま (``post_process_result`` を ``None`` にして) フォールバック
        採用する。収束・事後の干渉検証すら満たす候補が 1 つも無ければ
        ``None``。``collision_ik_time`` は干渉回避付きバッチ IK
        (``batch_inverse_kinematics`` の呼び出し) 自体の計算時間 [秒]。
    """
    seed_arm_pose(robot, robot_arm)
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    target_pos = palm_target_position(palm)
    rots = palm_to_target_rots(palm, robot_arm)
    target_coords = [Coordinates(pos=target_pos.tolist(), rot=rot)
                     for rot in rots]
    # collision_pairs 中の人体セグメントへの参照 (int) は
    # human_obstacle_names() の固定された並びを指しており、
    # collision_obstacles が空 (骨格 JSON が無い/--no-human-collision) の
    # ときは対応する障害物が存在しないので、そのまま渡すと
    # batch_inverse_kinematics 側でインデックス範囲外のエラーになる。
    # その場合は自己干渉ペア (Link 同士) だけを残す。
    effective_collision_pairs = collision_pairs
    if collision_pairs is not None and not collision_obstacles:
        effective_collision_pairs = [
            (link_a, other) for link_a, other in collision_pairs
            if not isinstance(other, int)]
    restore_joint_range = restrict_joint_range_margin(
        whole_body.link_list, collision_joint_limit_margin_ratio)
    collision_ik_start = time.time()
    try:
        # return_all_attempts=True: 既定では 1 目標につき初期値の中で誤差
        # 最小の 1 解しか返らないが、誤差最小の解が後処理判定 (solve_post_
        # process) にも通るとは限らず、初期値ごとの別の解なら通ることが
        # ある。そのため集約させず、(目標, 初期値) の全組み合わせの解を
        # そのまま受け取って全部候補にする (pick_verified_candidate 参照)。
        # 返る解は目標優先の並び (目標 0 の初期値 0..N-1, 目標 1 の初期値
        # 0..N-1, ...)。捨てられていた解も元々解かれてはいるので、この
        # 指定で計算時間は変わらない。
        angle_vectors, base_poses, success_flags, _ = \
            robot.batch_inverse_kinematics(
                target_coords=target_coords,
                move_target=move_target,
                link_list=whole_body.link_list,
                position_mask=True, rotation_mask=True,
                stop=collision_ik_stop,
                thre=collision_ik_thre,
                rthre=collision_ik_rthre,
                initial_angles='current',
                attempts_per_pose=attempts_per_pose,
                return_all_attempts=True,
                backend='jax',
                use_base='planar', base_limits=base_limits,
                collision_obstacles=collision_obstacles,
                collision_weight=collision_weight,
                collision_margin=collision_margin,
                self_collision=self_collision,
                collision_pairs=effective_collision_pairs,
                self_collision_weight=self_collision_weight,
                self_collision_margin=self_collision_margin)
    finally:
        restore_joint_range()
    collision_ik_time = time.time() - collision_ik_start
    # verification_pairs 中の人体セグメントへの参照 (int) も、
    # effective_collision_pairs と同じ理由 (collision_obstacles が空だと
    # 対応する障害物が存在しない) でここで除く。
    effective_verification_pairs = verification_pairs
    if verification_pairs is not None and not collision_obstacles:
        effective_verification_pairs = [
            (link_a, other) for link_a, other in verification_pairs
            if not isinstance(other, int)]
    picked = pick_verified_candidate(
        robot, success_flags, angle_vectors, base_poses,
        effective_verification_pairs, joint_positions,
        collision_verify_tolerance, robot_arm, palm, rots,
        attempts_per_pose=attempts_per_pose)
    return picked, collision_ik_time


def base_movable_region(base_limits):
    """バッチ IK に渡した ``base_limits`` (台車の IK 開始位置を原点とした
    [x, y, yaw] の (下限, 上限)) を、そのままワールド座標の範囲として
    dict にまとめる。

    台車は常にワールド原点から IK を開始する (``seed_arm_pose`` 参照) ので、
    ``base_limits`` はそのままワールド座標の範囲になる。
    ``view_handshake_poses.py`` がこの範囲を台車の可動域として可視化する。
    """
    x_range, y_range, yaw_range = base_limits
    return dict(
        x_range=[float(x_range[0]), float(x_range[1])],
        y_range=[float(y_range[0]), float(y_range[1])],
        yaw_range=[float(yaw_range[0]), float(yaw_range[1])],
    )


def solved_result(robot, robot_arm, target_pos, target_rot, turn_index,
                  angle_vector, base_pose, base_limits, post_process_result,
                  collision_ik_time):
    """採用した解をロボットに反映し、結果 dict を組む.

    バッチ IK はロボットを動かさないので、``angle_vector`` と
    ``base_pose`` を実際に反映してから手先・台車の姿勢を読み直す。
    Aero は ``root_link`` が ``base_link`` なので台車の姿勢は
    ``robot.newcoords`` で入る (``seed_arm_pose`` の注記を参照)。

    ``post_process_result`` (``solve_post_process`` が返した後処理後の
    結果 dict, ``pick_verified_candidate`` 参照) は、この関数がここまでで
    組んだ「後処理前」の位置姿勢と区別できるよう ``post_process`` キーに
    そのまま格納する。``view_handshake_poses.py`` はこの 2 つを切り替えて
    表示する。

    ``collision_ik_time`` (``solve_person_ik`` が計測した干渉回避付き
    バッチ IK 自体の計算時間 [秒]) はそのまま ``collision_ik_time`` キーに
    格納する (``run_pipeline_test.py`` が平均計算時間の集計に使う)。
    """
    robot.angle_vector(angle_vector)
    robot.newcoords(base_pose)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    result = dict(
        target=True,
        solved=True,
        turn_deg=TURN_CANDIDATES_DEG[turn_index],
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in hand_coords.worldpos()],
        hand_rot=[[float(v) for v in row] for row in hand_coords.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        base_movable_region=base_movable_region(base_limits),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
        collision_ik_time=collision_ik_time,
    )
    result['post_process'] = post_process_result
    return result


def unsolved_result(robot, robot_arm, target_pos, target_rot, base_limits):
    """どの候補も解けなかった人物のための結果 dict.

    逐次版が ``revert_if_fail=True`` で種の姿勢に戻してから結果を読んで
    いたのと同じになるよう、種の姿勢 (台車は ``solve_person_ik`` と同じく
    ワールド原点) を反映してから手先・台車の姿勢を読む。``turn_deg``/
    ``target_rot`` も逐次版と同じく最後に試した候補のものにする。
    """
    seed_arm_pose(robot, robot_arm)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    return dict(
        target=True,
        solved=False,
        turn_deg=TURN_CANDIDATES_DEG[-1],
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in hand_coords.worldpos()],
        hand_rot=[[float(v) for v in row] for row in hand_coords.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        base_movable_region=base_movable_region(base_limits),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
    )


def not_target_result(offered_hand, reason):
    """IK の対象外だった人物のための結果 dict.

    IK は解かないので関節角・台車位置は持たず、``solved`` は ``False``、
    ``target`` が ``False`` になる。``view_handshake_poses.py`` は
    ``target`` を見て「対象外」と表示する (通常の結果と同じファイル名で
    保存される)。

    Parameters
    ----------
    offered_hand : str or None
        掌 JSON の ``offered_hand`` (対象外なので通常は ``None``)。
    reason : str
        対象外にした理由 (``'no_offered_hand'`` / ``'no_palm'``)。
    """
    return dict(target=False, solved=False, offered_hand=offered_hand,
                robot_arm=None, not_target_reason=reason)


def load_palm_json(path):
    """``estimate_palm_poses.save_json`` が保存した 1 人分の JSON を読む."""
    with open(path) as f:
        return json.load(f)


def save_json(result, path):
    """IK の結果 dict を JSON として保存する."""
    json_io.save_json(path, result)


iter_palm_files = json_io.iter_json_files
load_skeleton_json = json_io.load_skeleton_json


def main():
    parser = argparse.ArgumentParser(
        description='掌の位置姿勢 JSON (estimate_palm_poses.py の出力) を '
                    '入力とし、ベース移動型ロボットが全身 IK をバッチで '
                    '解いて手を繋ぐ姿勢を求め、JSON として保存する。IK を '
                    '解くのは掌推定が「手を差し出している」と判定した '
                    '(offered_hand が L/R の) 人物だけで、null の人物には '
                    'IK の結果を持たない JSON (target: false) を書き出す。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'),
        help='掌の位置姿勢 JSON の入力ディレクトリ (既定は '
            'estimate_palm_poses.py の既定の出力先と同じ '
            'random_palm_poses/。test_generate_and_estimate_palm_poses.py '
            'が書き出した test_palm_pose_pipeline/palms/ を使う場合は '
            'このオプションで指定する)。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='IK の結果 JSON の保存先ディレクトリ (既定は '
            'random_handshake_poses/。入力と同じファイル名で保存するので、'
            'どの人物の結果かは入力ディレクトリの対応するファイルと '
            '突き合わせられる)。')
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='使うロボットの腕。既定 (auto) は人間の手の反対側 '
            '(人の左手ならロボットの右腕) -- 向かい合わず、人間と同じ '
            '方向を向いて反対側の手で繋ぐ想定のため。')
    parser.add_argument(
        '--attempts-per-pose', type=int,
        default=DEFAULT_ATTEMPTS_PER_POSE,
        help='1 つの目標姿勢に対して振る初期値の数。1 個目は肩を開いた '
            '種の姿勢 (台車はワールド原点)、残りは関節範囲・台車の可動 '
            '範囲の一様乱数。初期値ごとの解は 1 つに集約せず、全てが '
            '干渉検証・後処理判定に回される。増やすと解ける人物が増えるが '
            '遅くなる (既定 {})。'.format(DEFAULT_ATTEMPTS_PER_POSE))
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='人体の全身関節位置を持つ骨格 JSON (generate_random_human_'
            'poses.py の出力, --input-dir と同じファイル名で対応させる) '
            'のディレクトリ (既定 random_human_poses/。test_generate_and_'
            'estimate_palm_poses.py が書き出した test_palm_pose_pipeline/'
            'skeletons/ を使う場合はこのオプションで指定する)。干渉回避の '
            '障害物 (この人物の身体) を作るのに使う。')
    parser.add_argument(
        '--human-front-distance', type=float,
        default=HUMAN_FRONT_DISTANCE,
        help='Aero (常にワールド原点で IK を開始する) の前方どれだけの '
            '位置に人物を置くか [m] (``human_translation_offset`` 参照。'
            '既定 {:.1f})。骨格 JSON から人物の立ち位置が求まる場合は、'
            'その位置がちょうどこの距離になるよう人物側 (骨格全関節・掌 '
            '目標位置) を平行移動してから IK を解く。'.format(
                HUMAN_FRONT_DISTANCE))
    parser.add_argument(
        '--collision-weight', type=float,
        default=DEFAULT_COLLISION_WEIGHT,
        help='人体との干渉回避ペナルティの重み (既定 {})。'.format(
            DEFAULT_COLLISION_WEIGHT))
    parser.add_argument(
        '--collision-margin', type=float,
        default=DEFAULT_COLLISION_MARGIN,
        help='人体との干渉回避ペナルティが働き始める距離 [m] '
            '(既定 {})。'.format(DEFAULT_COLLISION_MARGIN))
    parser.add_argument(
        '--no-self-collision', dest='self_collision', action='store_false',
        help='ロボット自身のリンク同士の干渉 (解いている腕が胴体・反対側の '
            '腕・台車にぶつかる等) を回避するペナルティを無効にする '
            '(既定は有効。--collision-pairs が使えない場合は指定に '
            '関わらずどのみち無効になる)。人体との干渉回避 (--skeleton-'
            'dir) の有無に関わらず働く。')
    parser.add_argument(
        '--collision-pairs', type=str,
        default=os.path.join(_THIS_DIR, 'collision_pairs.json'),
        help='干渉回避で実際にチェックする組み合わせ (自己干渉のロボット '
            'リンク同士、および人体との干渉のロボットリンク×人体セグメント '
            '(human_obstacle_names 参照)) を指定する JSON (2 要素の名前の '
            'リストのリスト。既定 collision_pairs.json。build_collision_'
            'pairs.py が analyze_collision_pairs.py の出力 (collision_'
            'pair_analysis.json) から生成する)。干渉回避で対象にする組み '
            '合わせは常にこの JSON だけで決まり、collision_link_list の '
            'ような「対象リンク一覧への暗黙のフォールバック」は無い -- '
            '既定のパスにファイルが無ければ、自己干渉・人体との干渉の両方 '
            'を無効にして (通常のヤコビアン法の) IK を解く。片方の種類 '
            '(自己干渉/人体との干渉) のペアが JSON に 1 つも無い場合も '
            '同様に、その種類の干渉チェックだけが丸ごと働かなくなる '
            '(load_collision_pairs/solve_person_ik 参照)。')
    parser.add_argument(
        '--no-human-collision', action='store_true',
        help='人体を障害物とした干渉回避を無効にする (既定は --skeleton-dir '
            'に骨格 JSON があれば有効)。骨格 JSON 自体は読み込むので '
            '(--skeleton-dir が指す先に無ければ従来どおりエラーなく '
            'スキップされる)、人物の立ち位置の平行移動は従来どおり働く。 '
            'どのリンクの組み合わせが実際によく干渉するかを分析する '
            '目的で、干渉回避なしの (速いが干渉を考慮しない) IK 結果を '
            '大量に得たいときに使う (analyze_collision_pairs.py 参照)。')
    parser.add_argument(
        '--self-collision-weight', type=float, default=None,
        help='自己干渉回避ペナルティの重み (既定は --collision-weight と '
            '同じ値を使う, skrobot 側の self_collision_weight=None の '
            '既定挙動)。')
    parser.add_argument(
        '--self-collision-margin', type=float,
        default=DEFAULT_SELF_COLLISION_MARGIN,
        help='自己干渉回避ペナルティが働き始めるリンク間距離 [m] '
            '(既定 {})。'.format(DEFAULT_SELF_COLLISION_MARGIN))
    parser.add_argument(
        '--collision-ik-stop', type=int,
        default=DEFAULT_COLLISION_IK_STOP,
        help='干渉回避付きバッチ IK (backend=jax の勾配降下法) の最大反復 '
            '回数 (既定 {})。この勾配降下法は収束の有無によらず毎回この '
            '回数だけ律儀に反復する (壁時計時間のタイムアウトを別途設けても '
            '計算時間は変わらない)ため、閾値ぎりぎりまで反復してようやく '
            '収束するような無理のある姿勢を「収束しなかった」扱いにして '
            '弾きたい場合は、壁時計タイムアウトではなくこの値を減らす。 '
            '減らしすぎると自然な姿勢も収束前に弾かれるため、'
            '--attempts-per-pose とのトレードオフになる。'.format(
                DEFAULT_COLLISION_IK_STOP))
    parser.add_argument(
        '--collision-ik-thre', type=float,
        default=DEFAULT_COLLISION_IK_THRE,
        help='干渉回避付きバッチ IK の位置収束閾値 [m] (既定 {})。'.format(
            DEFAULT_COLLISION_IK_THRE))
    parser.add_argument(
        '--collision-ik-rthre', type=float,
        default=DEFAULT_COLLISION_IK_RTHRE,
        help='干渉回避付きバッチ IK の姿勢収束閾値 [rad] (既定 {:.4f})。'
            .format(DEFAULT_COLLISION_IK_RTHRE))
    parser.add_argument(
        '--collision-joint-limit-margin', type=float,
        default=DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO,
        help='干渉回避付きバッチ IK だけに適用する関節可動域の上下 '
            'マージン比率 (既定 {})。可動域の上限・下限それぞれ全域幅の '
            'この割合だけ内側に一時的に制限してから解く -- 干渉回避の '
            'ペナルティに引っ張られて関節が可動域の端に張り付いた解に '
            'なりやすく、それが後段の後処理 IK (solve_post_process) の '
            '収束を妨げる一因になっているため。0 を指定すると制限しない '
            '(従来と同じ挙動)。'.format(
                DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO))
    parser.add_argument(
        '--collision-verify-tolerance', type=float,
        default=DEFAULT_COLLISION_VERIFY_TOLERANCE,
        help='IK の収束判定 (success_flags) は位置・姿勢誤差だけを見ており '
            'collision_pairs の干渉ペナルティの残差を見ないため、収束した '
            '候補について採用前に厳密な形状で干渉を再計算し (collision_'
            'pairs_min_distance)、この距離 [m] を超えて貫通していれば '
            'その候補を棄却し次の候補を試す (どの候補も満たさなければ '
            'unsolved 扱いになる)。既定 {} は collision_mesh のポリゴン近似 '
            '誤差を吸収する程度の小さな値。'.format(
                DEFAULT_COLLISION_VERIFY_TOLERANCE))
    parser.add_argument(
        '--base-x-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_X_RANGE),
        help='台車の前後方向 (x) の移動範囲 [m]。IK 開始時の台車位置を '
            '原点とする (既定 {} {})。'.format(*DEFAULT_BASE_X_RANGE))
    parser.add_argument(
        '--base-y-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_Y_RANGE),
        help='台車の左右方向 (y) の移動範囲 [m] (既定 {} {})。'.format(
            *DEFAULT_BASE_Y_RANGE))
    parser.add_argument(
        '--base-yaw-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_YAW_RANGE),
        help='台車の向き (yaw) の範囲 [rad] (既定 {:.4f} {:.4f})。'.format(
            *DEFAULT_BASE_YAW_RANGE))
    parser.add_argument(
        '--seed', type=int, default=None,
        help='バッチ IK の乱数初期値に使う numpy の乱数シード。指定すると '
            '実行ごとに同じ解が得られる (既定は指定なし)。')
    parser.add_argument(
        '--collision-primitive-type', choices=['box', 'cylinder', 'sphere'],
        default=None,
        help='干渉回避に使うロボット自身のジオメトリを、指定した形状に '
            '全リンク強制変換する (view_aero_collision_model.py の '
            '--primitive-type と同じ。既定 (未指定) はリンクごとに '
            '自動選択)。')
    parser.add_argument(
        '--force-convert-collision-model', action='store_true',
        help='ロボット自身の干渉モデル (プリミティブ近似 URDF) のキャッシュ '
            'を使わず毎回作り直す (view_aero_collision_model.py の '
            '--force-convert と同じ)。')
    parser.add_argument(
        '--collision-urdf', type=str, default=None,
        help='干渉回避 (collision_link_list) のジオメトリ・対象リンクの '
            '読み込み元 URDF を明示的に指定する (既定は view_aero_'
            'collision_model.py と同じ方法で自動生成・キャッシュされる '
            'プリミティブ近似 URDF)。--collision-primitive-type/'
            '--force-convert-collision-model とは併用できない。IK を '
            '解くロボット本体のキネマティクスは変えず (常に Aero(use_'
            'hand=False))、干渉回避に使うリンクの集合・ジオメトリだけを '
            '(リンク名で対応づけて) このURDFのものに差し替える -- 干渉 '
            '計算を高速化する目的で手動で間引いた/単純化した独自の干渉用 '
            'URDF を使いたい場合などに使う。このURDFに対応するリンクが '
            '見つからない robot 側のリンクは、干渉回避の対象から明示的に '
            '除外される (apply_collision_model 参照)。')
    args = parser.parse_args()

    files = iter_palm_files(args.input_dir)
    if not files:
        print('{} に掌の位置姿勢 JSON が見つかりません。先に '
              'estimate_palm_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    # r/l_eef_grasp_link (IK が使う手先フレーム) は手あり/なし両方の URDF に
    # ある (Aero.__init__ 参照) ので、指の関節が要らないこのスクリプトでは
    # 手なしモデルを使う。
    robot = Aero(use_hand=False)
    restrict_elbow_range(robot)
    apply_collision_model(
        robot,
        primitive_type=args.collision_primitive_type,
        force_convert=args.force_convert_collision_model,
        collision_urdf_path=args.collision_urdf)

    # collision_link_list (対象リンクの集合を丸ごと指定する概念) は
    # 使わない -- 干渉回避で実際にチェックする組み合わせは常に
    # collision_pairs (--collision-pairs の JSON) だけで決める。JSON が
    # 無ければ、自己干渉・人体との干渉のどちらを対象にすればよいかを
    # 決める手段が無いので、干渉回避そのものを無効にして (通常のヤコビアン
    # 法の) IK にフォールバックする -- collision_link_list を使った
    # 「ロボット全身の全組み合わせ」への暗黙のフォールバックはしない。
    collision_pairs = None
    verification_pairs = None
    if os.path.exists(args.collision_pairs):
        collision_pairs = load_collision_pairs(args.collision_pairs, robot)
        print('[collision-pairs] {} 組の干渉ペアを {} から読み込みました。'
              .format(len(collision_pairs), args.collision_pairs))
        # IK の収束判定 (success_flags) は位置・姿勢誤差だけを見ており、
        # collision_pairs の干渉ペナルティが実際に解消したかは保証しない
        # ため、solve_person_ik は候補を採用する前に事後検証を行う
        # (pick_verified_candidate 参照)。この検証は collision_pairs
        # (最適化のコストを抑えるために絞り込んだ組み合わせ) ではなく、
        # 隣接リンクの組み合わせだけを除いた総当たりの組み合わせに対して
        # 行う -- collision_pairs に無い組み合わせで実際には貫通している
        # 解を見逃さないため。ロボットの構造だけで決まるので人物ループの
        # 外で 1 回だけ計算する (robot_arm 引数は collision_link_list_for_
        # arm と同じくプレースホルダで結果に影響しない)。
        verification_pairs = build_collision_verification_pairs(robot, 'r')
    else:
        print('[collision-pairs] {} が見つからないため、干渉回避 (自己干渉'
              '・人体との干渉の両方) を無効にして IK を解きます。'.format(
                  args.collision_pairs))

    base_limits = [tuple(args.base_x_range), tuple(args.base_y_range),
                   tuple(args.base_yaw_range)]

    n_solved = 0
    n_total = 0
    n_not_target = 0
    for i, path in enumerate(files):
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        palms = load_palm_json(path)
        human_hand = palms.get('offered_hand')
        palm = palms.get(human_hand) if human_hand in ('L', 'R') else None
        if palm is None:
            reason = ('no_palm' if human_hand in ('L', 'R')
                      else 'no_offered_hand')
            save_json(not_target_result(human_hand, reason), out_path)
            n_not_target += 1
            print('[{}/{}] {} -> {} (not target: {})'.format(
                i + 1, len(files), os.path.basename(path), out_path, reason))
            continue
        robot_arm = (DEFAULT_ROBOT_ARM[human_hand]
                     if args.robot_arm == 'auto' else args.robot_arm)

        # この人物の身体を干渉回避の障害物にする。骨格 JSON (全身の関節
        # 位置) が --skeleton-dir に無い、または collision_pairs が
        # 使えない (--collision-pairs の JSON が無い) ときは、人体との
        # 干渉回避なしのバッチ IK にフォールバックする。
        skeleton_path = os.path.join(args.skeleton_dir,
                                     os.path.basename(path))
        # Aero は常にワールド原点で IK を開始する (seed_arm_pose 参照) ので、
        # 骨格 JSON があれば、人物の立ち位置がちょうど Aero の前方
        # --human-front-distance になるよう、骨格の全関節位置と掌の目標
        # 位置を平行移動してから IK を解く (human_translation_offset 参照)。
        # この平行移動は干渉回避の有無に関わらず常に行う (台車と人物が
        # 重ならない開始位置にするため)。骨格 JSON が無ければ平行移動
        # できないので、掌の位置はそのまま使う。
        if os.path.exists(skeleton_path):
            joint_positions = load_skeleton_json(skeleton_path)
            offset = human_translation_offset(
                joint_positions, front_distance=args.human_front_distance)
            joint_positions = translate_joint_positions(
                joint_positions, offset)
            collision_obstacles = (
                [] if (args.no_human_collision or collision_pairs is None)
                else human_body_obstacles(joint_positions))
            palm = translate_palm(palm, offset)
        else:
            print('  {} に骨格 JSON が無いため、この人物は人体との干渉回避 '
                  'なしで解きます。'.format(skeleton_path))
            collision_obstacles = []
            joint_positions = None

        target_pos = palm_target_position(palm)
        rots = palm_to_target_rots(palm, robot_arm)
        picked, collision_ik_time = solve_person_ik(
            robot, palm, robot_arm, collision_obstacles,
            attempts_per_pose=args.attempts_per_pose,
            base_limits=base_limits,
            collision_weight=args.collision_weight,
            collision_margin=args.collision_margin,
            self_collision=(args.self_collision and collision_pairs is not None),
            collision_pairs=collision_pairs,
            self_collision_weight=args.self_collision_weight,
            self_collision_margin=args.self_collision_margin,
            collision_ik_stop=args.collision_ik_stop,
            collision_ik_thre=args.collision_ik_thre,
            collision_ik_rthre=args.collision_ik_rthre,
            collision_joint_limit_margin_ratio=(
                args.collision_joint_limit_margin),
            joint_positions=joint_positions,
            verification_pairs=verification_pairs,
            collision_verify_tolerance=args.collision_verify_tolerance)
        if picked is None:
            result = unsolved_result(robot, robot_arm, target_pos, rots[-1],
                                     base_limits)
        else:
            turn_index, angle_vector, base_pose, post_process_result \
                = picked
            result = solved_result(
                robot, robot_arm, target_pos, rots[turn_index],
                turn_index, angle_vector, base_pose,
                base_limits, post_process_result, collision_ik_time)
        result['offered_hand'] = human_hand
        result['robot_arm'] = robot_arm
        n_total += 1
        n_solved += int(result['solved'])
        save_json(result, out_path)
        print('[{}/{}] {} -> {} ({}Hand -> {}arm, {})'.format(
            i + 1, len(files), os.path.basename(path), out_path,
            human_hand, robot_arm,
            'solved' if result['solved'] else 'NOT solved'))

    print('{}/{} solved (対象外 {} 人: 掌推定の offered_hand が null 等)。'
          .format(n_solved, n_total, n_not_target))


if __name__ == '__main__':
    main()
