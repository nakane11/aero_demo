# 開発・デバッグ用ツール (`tools/`)

`scripts/` (本番: `run_pipeline_test.py`/`run_camera_pipeline_test.py` と
それらが直接 import・subprocess 呼び出しする範囲) と、それ以外の
grid search・中間結果の可視化・データ収集・ラベル付けなどの開発用
プログラムを分けるため、後者は `tools/` (ROS 依存のものは `tools/ros/`)
にまとめてある。本番パイプラインの説明は [README.md](../README.md) を
参照し、このページは `tools/`/`tools/ros/` 以下の各プログラムの使い方を
まとめる。

## 位置づけとディレクトリ構成

```
aero_demo/
    scripts/       本番。run_pipeline_test.py/run_camera_pipeline_test.py
                    とその依存 (generate_random_human_poses.py/
                    estimate_palm_poses.py/solve_palm_ik.py/
                    plan_handshake_motion.py/side_by_side_transition.py/
                    view_handshake_poses.py/
                    view_handshake_motion.py/handshake_viewer_common.py/
                    collision_pairs.json、ROS ノードは scripts/ros/ 以下)
    tools/         開発・デバッグ用 (このページの対象)
    tools/ros/     開発・デバッグ用のうち ROS (rospy 等) に依存するもの
    src/aero_demo/ ROS 非依存の共有ライブラリ (catkin_python_setup で
                    import 可能にしてある)
```

`tools/`/`tools/ros/` の各プログラムは `scripts/` 側のモジュール
(`solve_palm_ik`/`generate_random_human_poses`/`estimate_palm_poses` 等)
を直接 `import` する。ファイル冒頭で次のように `sys.path` へ
`scripts/`(と `src/`)を追加してから import しているので、
`rosrun`/PYTHONPATH の設定なしに `python3 tools/xxx.py` や
`python3 tools/ros/xxx.py` の形でそのまま実行できる (`rosrun` 経由での
実行はサポートしない)。

```python
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')  # tools/ros/ 以下は '..', '..', 'scripts'
...
```

`--input-dir`/`--skeleton-dir` 等の既定値も、`scripts/` 側のパイプラインが
既定で書き出す `data/synthetic/random_human_poses/`/`data/synthetic/random_palm_poses/`
等をそのまま指すようにしてある (指定を省略すれば `scripts/` 側で
`generate_random_human_poses.py`/`estimate_palm_poses.py` 等を実行した
結果をそのまま読める)。

## パラメータ探索・チューニング

### `tools/grid_search_collision_ik.py`

`solve_palm_ik.py` の速度・成功率トレードオフ (初期値の数・干渉回避ペア
設定・最大反復回数・干渉の重み/マージン・台車の可動域半幅・向き候補数・
後処理 IK の閾値等) を、各軸の全組み合わせで `solve_palm_ik.py` を
サブプロセス実行してグリッドサーチし、成功人数・1 人あたりの時間
(warmup 除く)・事後検証の棄却内訳を集計する。初回は jax の再コンパイルに
なるので各組を `--repeat` 回 (既定 2) 実行し最後の回を使う。
`--dataset` (必須、複数可) には `skeletons/`・`palms/` を持つディレクトリを
渡す。`--output` を指定すると結果を JSON Lines で追記する。

```bash
python3 tools/grid_search_collision_ik.py \
    --dataset /tmp/data/tune \
    --collision-pairs scripts/collision_pairs.json pairs16.json \
    --attempts-per-pose 512 256 --output results.json
```

### `tools/grid_search_handshake_motion.py`

`plan_handshake_motion.py` の軌道最適化のハイパーパラメータ
(`--motion-attempts`/`--n-waypoints`/`--max-iterations`) の成功率・
計算時間トレードオフを調べるグリッドサーチ。既定は OFAT (one-factor-at-
a-time)、`--mode full` で全軸総当たりにできる。結果は
`tools/grid_search_handshake_motion_results.csv` にも書き出す。

```bash
python3 tools/grid_search_handshake_motion.py --num-samples 100 \
    --force-optimize
```

### `tools/build_collision_pairs.py`

`scripts/collision_pairs.json` (本番の `solve_palm_ik.py
--collision-pairs` が読む既定ファイルそのもの) を、干渉回避無しで解いた
IK 結果から実際に干渉した頻度でランキングし直して作り直すツール。内部の
`analyze_handshake_dir` (旧 `analyze_collision_pairs.py` から移植) が
集計処理を担う。

```bash
python3 tools/build_collision_pairs.py --num-pairs 8
```

### `tools/handshake_reachability_map.py`

人が手をどこに・どの向きで差し出せば差し出しと判定され IK が解けるかを
格子状に調べる (reachability map)。体格を固定した骨格で掌の位置 (前方・
外側・高さ) と向きを振り、格子点ごとに差し出し判定 (`OfferedHandSelector`)
と IK (事後検証・後処理まで、軌道計画なし) を評価する。結果は JSON Lines で
`--output` に追記し、再実行すると続きから解く。

```bash
# 位置の格子 (既定の向き = 握手の向き: 指先は前、親指が上)
python3 tools/handshake_reachability_map.py --mode position \
    --hands R L --output /tmp/reach_position.jsonl
# 向きの格子 (位置は --orientation-position に固定)
python3 tools/handshake_reachability_map.py --mode orientation \
    --hands R L --output /tmp/reach_orientation.jsonl
```

### `tools/build_hand_offer_reachability.py`

`handshake_reachability_map.py` の結果 (JSON Lines) から、IK 失敗時の手の
出し方の助言 (`aero_demo.hand_offer_advice`、`run_camera_pipeline_test.py`
の `--hand-offer-table`) が使う `config/hand_offer_reachability.json` を作る。

```bash
python3 tools/build_hand_offer_reachability.py \
    /tmp/handshake_reachability/position*.jsonl \
    /tmp/handshake_reachability/orientation.jsonl
```

## 中間結果の可視化

### `tools/draw_random_human_poses.py`

`generate_random_human_poses.py`/`estimate_palm_poses.py` が出力した JSON
(SMPL の人モデル・骨格・掌の位置姿勢) を viser で表示する、パイプライン
本番の手順 1・2 を目視確認するためのツール。`--advance-mode auto` で
`--pause` 秒ごとに自動送り、`--output-dir` を指定すると表示した各姿勢の
画像を保存する。

```bash
python3 tools/draw_random_human_poses.py
```

### `tools/plot_handshake_motion_2d.py`

`run_pipeline_test.py --plan-motion` を実行 (または既存の作業ディレクトリ
を再利用) し、軌道計画の結果を人物ごとに真上から見た 2 次元の図 (PNG) に
する。台車の軌道・向き・人間の立ち位置/正面方向・差し出した手を描く。

```bash
python3 tools/plot_handshake_motion_2d.py 20 --seed 3 \
    --initial-base-pose 5 0 3.14
```

### `tools/render_pipeline_stages.py`

`run_pipeline_test.py --plan-motion --side-by-side-transition` の作業
ディレクトリから 1 人を選び、横並びまでの各段階 (差し出した手 → 第 1 段 IK →
hover の事後検証 → 押し込み → 接近軌道 → 横並び移動) で考慮する干渉ペア・
人体からの距離・台車の可動域・経路を、3 次元の描画 + 真上から見た図 + 説明の
PNG にする。`--viser` で同じシーンを viser にも出す (全段階のロボットの全体表示で
ロボットをクリックするとその段階の詳細、項目はチェックボックスで表示を切り替え。
`--no-images` で PNG を省く)。
出力例と説明は [`handshake_pipeline_stages.md`](handshake_pipeline_stages.md)。
WSL では EGL のソフトウェア描画を指定する。

```bash
export PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless JAX_PLATFORMS=cpu
python3 tools/render_pipeline_stages.py \
    --work-dir /tmp/aero_demo_pipeline_xxxx --name human_324 \
    --output-dir docs/images/pipeline_stages [--viser]
```

### `tools/summarize_final_base_placement.py`

`run_pipeline_test.py` の作業ディレクトリ (複数可) を読み、IK が決めた
最終の台車位置と人の位置関係 (方位・人の正面方向に沿った前方ずれ)・
成功率・IK の計算時間を集計する。

```bash
python3 tools/summarize_final_base_placement.py \
    /tmp/aero_demo_pipeline_xxxx /tmp/aero_demo_pipeline_yyyy
```

### `tools/view_aero_collision_model.py`

Aero の干渉 (コリジョン) モデル (`aero_demo.collision_model.
build_collision_model_urdf` が生成する box/cylinder/sphere のプリミティブ
近似 URDF、本番の `solve_palm_ik.py`/`handshake_viewer_common.py` も使う
のと同じもの) を、元の Aero に半透明で重ねて viser で表示するだけの
ツール。プリミティブ近似の生成処理自体は `aero_demo.collision_model` に
あり本番からも使われるが、この可視化そのものは開発用。

```bash
python3 tools/view_aero_collision_model.py
```

### `tools/ros/print_palm_positions.py`

`run_camera_pipeline_test.py` から ARM/RESET/IK/軌道計画を取り除き、
ボタン操作なしでカメラ画像から掌推定だけを常時実行し続け、検出できた
人物の掌の位置 (base_link 座標系) と IK 目標位置・実機の現在の手先位置
(TF) を標準出力に print し続けるだけの検証用スクリプト。骨格は
`visualization_msgs/MarkerArray` として `skeleton_markers` にも publish
するので、rviz からも確認できる (`launch/view_skeleton.launch` 参照)。

```bash
python3 tools/ros/print_palm_positions.py
```

### `tools/ros/compare_hand_detectors.py`

同じフレームに MediaPipe Holistic (計画時の掌推定) と Hands (押し込み直前の
再認識) をかけ、掌の位置・向きの差 (検出器の差) を測る。首を計画時と同じ
下向きにした後は台車・首とも動かさないので、人は肩〜手が映る距離
(0.8〜1.5m) で手を差し出して静止する。結果は `--out-dir` (既定
`data/palm_detector_compare/<日時>/`) に frames.jsonl / summary.json /
overlay_*.png (Holistic=赤、Hands=緑) として保存する。共通処理は
`tools/ros/palm_check_common.py` にある。

```bash
python3 tools/ros/compare_hand_detectors.py --side R
python3 tools/ros/compare_hand_detectors.py --side L --frames 100
```

### `tools/ros/replay_press_in_refine.py`

実機試行の bag で、押し込み直前の補正 (`_refine_press_in`) をやり直す。
hover で止まっている区間の画像で実機ノードと同じく Hands で手を検出し、
3 フレームずつ掌を選んで押し込み姿勢を解き直す。補正量の上限
(`REFINE_MAX_*`) は外して、必要だった補正量と IK が解けるかを出し、実機
ノードの上限で採用されるかに `*` を付ける。`--warmup approach` (既定) は
実機ノードと同じく接近の開始から Hands を走らせて追跡を温め、`--warmup 0`
は補正のときだけ Hands を呼ぶ (2026-10-16 までの実機ノード)。入力は試行の
`person_NN.log` (同じディレクトリに `palms/`・`handshakes/`・`motions/`) と
bag。roscore は不要。

```bash
python3 tools/ros/replay_press_in_refine.py \
    data/real/2026-10-16/trial6_side_by_side/person_06.log \
    data/real/2026-10-16/trial6_side_by_side/*.bag
```

## 実カメラデータの収集・ラベル付け

以下の 3 つは、実カメラでの `offered_hand` (差し出し手) 判定
(`estimate_palm_poses.OfferedHandSelector`) の精度を上げるための、
データ収集 → ラベル付け → チューニングの一連の流れで使う。

### 0. 連続録画

`tools/ros/record_palm_offer_clips.py` は今の判定器が認識できた差し出し
しか保存しないので、認識されなかった場面を集めるときは `rosbag record`
で判定器に依存せず連続録画する (トピック名は `run_camera_pipeline_test.py`
の既定値)。

```bash
mkdir -p data/real/2026-09-30_session1
rosbag record -O data/real/2026-09-30_session1/session1.bag \
    /camera/color/image_raw/decompressed \
    /camera/depth/image_raw/decompressed \
    /camera/color/camera_info /tf /tf_static
```

保存先は `data/real/<日付>_<セッション名>/` にする
([`docs/test_data.md`](test_data.md))。

`/tf_static` は latched なので、ロボット側が起動済みでも録画開始時に
受け取れる。

### 1. `tools/ros/extract_skeletons_from_bag.py`

判定器に依存しない生の rosbag (上の連続録画) を読み込み、
`--sample-interval` 秒 (既定 0.5) おきに骨格・掌位置姿勢・骨格重畳画像を
機械的にサンプリングして保存するオフライン抽出ツール (roscore 不要)。
`tools/ros/record_palm_offer_clips.py` が切り出した判定済みクリップを
`--single-sample` で混ぜることもできる。

```bash
python3 tools/ros/extract_skeletons_from_bag.py \
    --bag data/real/2026-09-30_session1/session1.bag \
    --output-dir data/real/2026-09-30_session1
```

保存先には `skeletons/`/`palms/`/`images/` の 3 サブディレクトリができる
(`skeletons/`/`palms/` はそれぞれ `estimate_palm_poses.py`/
`tune_offer_selector.py` の入力形式と互換)。掌 JSON には判定の基準にした
ロボット手先の位置 (`robot_position`, TF の `r_eef_grasp_link`) も保存し、
`tune_offer_selector.py` はそれを使ってスコアを計算し直す。抽出し直しても
付けた `human_label` は残る。

### 2. `tools/label_offer_images.py`

`extract_skeletons_from_bag.py` (または `record_palm_offer_clips.py`) が
保存した骨格重畳画像を 1 枚ずつ viser の GUI パネルに表示し、
Right/Left/Null ボタンで「実際にはどちらの手を差し出しているか」の
人手ラベル (`human_label`) を対応する掌 JSON に書き込む。

```bash
python3 tools/label_offer_images.py \
    --image-dir /tmp/offer_dataset/images \
    --palm-dir /tmp/offer_dataset/palms
```

### 3. `tools/tune_offer_selector.py`

人手ラベル付きサンプル (`human_label`) から `OfferedHandSelector` の
パラメータ (判定軸のブレンド・ランプ・高さ方向の重み・分離度の重み) を
ランダムサーチ + グリッド微調整で最適化する。結果は
`tools/tuned_offer_selector_params.json` に保存する。

```bash
python3 tools/tune_offer_selector.py \
    --skeleton-dir /path/to/skeletons --palm-dir /path/to/palms
```

## 実カメラパイプライン (`scripts/ros/run_camera_pipeline_test.py`) の掌クリップ収集と bag テスト

### `tools/ros/record_palm_offer_clips.py`: クリップの録画

実カメラ・実ロボットが動いている状態で実行する:

```bash
python3 tools/ros/record_palm_offer_clips.py
# 保存先を変えたい場合
python3 tools/ros/record_palm_offer_clips.py --save-dir /tmp/palm_offer_clips
```

`Ctrl-C` などで止めるまで無期限に動き続け、その間に検出した掌の差し出しを連番の
ファイル名で次々に保存する (判定基準は `run_camera_pipeline_test.py` の ARMED 中の
判定と同じ既定値に揃えてある)。掌の差し出しを検出すると、その時刻の `--pre-seconds`
秒前 (既定 2.0) から `--post-seconds` 秒後 (既定 2.0) までの
color/depth/camera_info/tf/tf_static を 1 本の `.bag` にまとめて `--save-dir`
(既定 `data/palm_offer_clips/`) に保存する。同じ差し出し動作を 2 回に分けて録らない
よう、1 クリップ保存後 `--cooldown-seconds` 秒 (既定 3.0) は次のトリガーを無視する。

このノードは今の判定器が認識できた差し出ししか保存しないので、「差し出したのに
認識されなかった」場面を集めるには使えない。その場合は `rosbag record` で連続録画して
から `tools/ros/extract_skeletons_from_bag.py` で抽出する (下の「実カメラデータの収集・
ラベル付け」参照)。

保存されるファイルはクリップごとに 3 つ:

```
data/palm_offer_clips/
    20260911_214500_R_000.bag   # color/depth/camera_info/tf/tf_static (4秒分)
    20260911_214500_R_000.json  # trigger_stamp/offered_hand/pre_seconds/post_seconds/topics/bag_path/snapshot_path
    20260911_214500_R_000.png   # 差し出しを検出した瞬間のカラー画像 + 骨格描画 (差し出し手は赤)
```

`~skeleton_image` には、購読者がいれば骨格・掌の有無や録画中かどうかに関わらず毎フレーム
同じ骨格描画画像を publish する (差し出し手は赤で描く)。クリップを保存し終えてからの
`--cooldown-seconds` の間だけ publish を止める。`rqt_image_view` 等で購読すれば、
常時の検出状況をリアルタイムに確認できる。

差し出し手判定の基準にするロボット手先の base_link 座標は、既定では `--robot-hand-frame`
(既定 `r_eef_grasp_link`、実機では `/aero_state_publisher` が配信する) を毎フレーム TF
で引いて使う (`estimate_palm_poses.OfferedHandSelector` 自身の既定動作 (未指定/`None`) は
合成骨格向けの「人物より world +x 側にロボットがいる」という前提で、実カメラ・base_link
座標系では食い違うため使わない)。ロボット未接続などでまだ TF が引けない間だけ概算値
`(0.32, -0.55, 0.93)` にフォールバックする。差し出しているのに判定が届かないときは
`--offer-score-min` (既定 0.65) を調整する。

### `run_camera_pipeline_test.py` の bag テスト用オプション

保存したクリップ (または [`test_data.md`](test_data.md) の切り出し bag) を `--bag` で渡せば、
実カメラ・実ロボットの TF 配信なしにパイプライン全体をそのままテストできる。

```bash
python3 scripts/ros/run_camera_pipeline_test.py \
    --bag data/palm_offer_clips/20260911_214500_R_000.bag \
    --auto-arm --no-wait-for-client
```

**注意 (未修正の既知の問題):** 現在の実装は `HandshakePipelineNode` の構築 (TF 待ちを含む)
が終わってから `rosbag play --clock` を起動するため、`/clock` が来ないまま起動処理で
止まることがある。その場合は `rosbag play <bag> --clock` を先に別プロセスで起動しておき、
ノードは `--bag` を付けずに起動する ([`jax_compilation_cache.md`](jax_compilation_cache.md)
6 節)。切り出し bag の `replay.sh` は、初期化中だけ TF を先頭 1 秒ぶんループ再生して
この問題を避けている ([`test_data.md`](test_data.md))。

- `--bag <path>`: 実カメラの代わりに指定した rosbag を再生する (内部で `rosbag play
  --clock` をサブプロセスとして起動し、`/use_sim_time` をあわせて有効にする)。
  `--bag-rate` で再生速度 (既定 1.0)、`--bag-loop` で繰り返し再生を指定できる。
- `--auto-arm`: viser 画面の ARM ボタンを押す代わりに起動直後から ARMED 状態にする
  (無人でのバッグ再生テスト用)。ARMED は差し出し手が決まるまで時間制限なく待つ。
  `--auto-execute` で IK・軌道計画に失敗して動かせなかったときは ARMED のまま次の差し出しを
  待つ (今の手を動かすか向きを変えるまで次の差し出しは受け付けない)。
- `--no-wait-for-client`: viser のブラウザクライアント接続を待たずに起動を続ける (既定では
  接続まで無期限に待つため、無人テストでは必須)。表示自体を見たいときは外して、普段どおり
  viser の URL をブラウザで開けばよい。
- `--no-robot-interface`: 実機 (`AeroROSRobotInterface`) に接続しない。
- `--force-optimize` (既定 False): 候補が検証に通っていても必ず jaxls の軌道最適化まで
  実行させる (`plan_handshake_motion.py` の同名オプションと同じ、計算時間の計測用。
  [`jax_compilation_cache.md`](jax_compilation_cache.md) 11 節)。通常運用では付けない。

`--auto-execute` を指定しなければ実機は一切動かさず viewer 上での確認のみになるので、
このテストに実ロボットは不要 (`--bag` のクリップに `tf`/`tf_static` も含めている
ため、実ロボットの TF 配信も不要)。

### `--auto-execute`: 実機での実行速度とログ

`--auto-execute` で実機を動かすとき、waypoint 間の所要時間は固定の刻み幅ではなく、
区間ごとに `run_camera_pipeline_test.py` 冒頭の定数から決める (`_limited_time_list`)。
軌道計画の `dt` (`plan_handshake_motion.DEFAULT_DT`) は最適化のコスト正規化に使うだけで、
実行速度には関係しない。

- 速度の上限: 台車は `BASE_MAX_VEL`/`BASE_MAX_ANGVEL` (並進 0.3 m/s・回頭 1.0 rad/s、
  ロボット本体側の `aero_base_link.yaml` の `max_velocity` と必ず一致させる)、
  腕・首・腰・リフターは URDF の `velocity`。指令ではこれに `VEL_LIMIT_RATIO`
  (既定 0.9) を掛けた値までしか使わない。
- 加速度の上限: 上記の速度まで `ACCEL_TIME` 秒 (既定 0.4) かけて加速する値。大きくする
  ほどゆるやかに加減速するが、全体の所要時間は延びる。
- 各区間の所要時間は、その区間で律速する軸 (台車の並進・回頭、各関節) の速度または
  加速度がちょうど上限になる長さにする。実機のコントローラ (腕は
  JointTrajectoryController、台車は pr2_base_trajectory_action) が各点の位置・速度から
  3 次補間する際の瞬間値で判定するので、上限を超えた指令を実機側で頭打ちにされることは
  ない。ほとんど動かない区間は `MIN_SEGMENT_TIME` (1/15 秒) を下限にする。
- 台車と腕には同じ時間列を送り、律速しない側はその区間だけ上限より遅く動く (タイミング
  がずれて干渉検証済みの経路から外れないようにするため)。
- 台車の各点の速度は腕と同じ規則 (始点・終点 0、途中は前後区間の平均速度の平均) で付け直して
  送る (`_send_base_trajectory`)。skrobot の `move_trajectory_sequence` のままだと、静止状態から
  最初の区間の速度へいきなり跳ぶため。

押し込みの後は、既定で横並び移動 (つないだ手を人の体の横へ下ろしながら横並びへ移る) を続けて
行う。計画が移動先の 3 割以上まで進めた (`verified`) 人だけで、解けていない試行は実機を動かさない。
押し込み終了 (「どうぞ」) から `--side-by-side-delay` 秒 (既定 4.0) 後に「一緒に横に並びますね」
(`--speech-transition-text`) と発話して動き出す。人の手を引いて動くので、各区間は
`--side-by-side-segment-time` 秒 (既定 0.2) 以上かける。押し込み後の画像の保存
(`--grasp-capture-duration`) は動き出すまでで打ち切る。`--no-side-by-side-transition` で行わない。
接近中と横並び移動中は首で人の顔を追い (`--no-head-tracking` で、ARMED 中の追従・見渡しとともに無効)、hover に着いたら首を
計画の角度 (掌を見る) に戻して押し込む。

ログは画面には重要なものだけ出し、詳細 (`[debug]` など) は
`/tmp/run_camera_pipeline_test_logs/<起動時刻>/` 以下に、起動時 (`startup.log`) と試行 (人)
ごと (`person_01.log`, ...) のファイルで保存する (画面に出る成功/失敗などのログも同じファイルに
時刻付きで入る。起動ごとに別のディレクトリで前回のものは消えない。最新は `latest` から
たどれる)。区間の送信時には次のようなログが出る:

```
[debug][segment] waypoint数=21 計画上のdyaw=-35.2deg (=[...]) 送信直前odom_yaw=0.3deg time_list合計=8.94s (=[0.35, 0.31, ...])
```

## 実機データでの回帰テスト

どのデータを使うかは [`docs/test_data.md`](test_data.md) にまとめてある。

### `tools/ros/clip_bag.py`

録画した全体 bag から再生テスト用の区間を切り出す。`/tf_static` は時刻を
区間先頭に付け替えて先頭に入れる。

```bash
python3 tools/ros/clip_bag.py <全体 bag> <出力 bag> \
    --start '2026-10-15 17:12:08.5' --end '2026-10-15 17:12:46.0'
```

### `tools/ros/resolve_saved_trials.py`

`run_camera_pipeline_test.py` が試行ごとに保存した skeletons/palms の
JSON を、ノードと同じ `HandshakePlanner` (IK -> 軌道計画 -> 横並び移動の
計画) で解き直し、到達段階と保存時から変わった試行を出す。骨格検出の
揺れが入らないので変更前後の比較に使う。

```bash
python3 tools/ros/resolve_saved_trials.py data/real/2026-10-13/logs
```

## 頭カメラの外部パラメータの較正

`launch/decompress.launch` の `head_to_camera_link` (と同じ値を持つ
`scripts/solve_palm_ik.py` の `HEAD_TO_CAMERA_LINK_*`) を、深度画像を床と
/scan に合わせて推定する。

### `tools/ros/record_head_calibration_sweep.py`

実機の首を格子状に動かして各姿勢で止め、較正用の bag (深度・camera_info・
TF・/scan・/joint_states・/odom、`--with-color` でカラーも) を録る。
腕・腰・脚には指令を送らない。首は head_controller に送り、ゴールが
SUCCEEDED でないか目標に届かなければ中止し、最後に元の角度へ戻す。
`--base-yaws` を付けると台車をその場で回して (開始時の向きから
`--max-base-yaw` 以内、/teleop/cmd_vel に低速の回転だけ) 各向きで同じ
首の動きを繰り返し、最後に元の向きへ戻す。台車中心から
`--obstacle-min`〜`--obstacle-radius` に /scan の点があれば回さない
(既定 0.40〜0.70m は狭い場所では止まるので、周りを確認して調整する)。
実機を動かすので、実行前に周囲を確認すること。`--dry-run` は動かさず
姿勢の列と最も近い障害物の距離だけ表示する。decompress.launch を先に
起動しておく。

```bash
rossetmaster 192.168.96.215; rossetip
python3 tools/ros/record_head_calibration_sweep.py \
    --output data/bags/head_calib.bag --dry-run
python3 tools/ros/record_head_calibration_sweep.py \
    --output data/bags/head_calib.bag            # 首 ±60deg x 15〜30deg の 28 姿勢
python3 tools/ros/record_head_calibration_sweep.py \
    --output data/bags/head_calib_rot.bag --base-yaws=-40,40
```

床と、レーザーの高さ (約 20cm) の壁・棚の面が 1〜3m に広く見える場所が
向いている (椅子の脚などが多い場所は /scan との対応が取りにくい)。首ピッチ
0deg は床が映らないので使わない。

### `tools/ros/calibrate_head_camera.py`

bag (複数可) から台車・首・腰・脚が止まっている時刻を選び、深度点群を
今の head_link -> camera_link に補正 (camera_link 系の回転・並進) を掛けて
base_link に置いたとき、床の点が z = `--floor-z` (ピッチ・ロール・高さ)、
/scan と同じ高さの断面の点が /scan の線上 (ヨー・x・y) になるよう、全時刻
まとめて最小二乗で解く (外れ値に強い soft_l1、並進には
`--translation-sigma` の事前分布、`--fix-translation` で回転のみ)。床が
映っていない時刻 (当てはめた平面が `--max-floor-tilt-deg` より傾く) は床に
使わない。時刻ごとの補正前後の残差 (床の高さ・傾き、/scan までの距離、
3cm 以内の割合、補正を base_link 系に直した回転) と、decompress.launch と
solve_palm_ik.py に書く値を表示する。`--no-optimize --head-to-camera ...` で
指定した値を別の bag で検証できる。roscore 不要。

```bash
python3 tools/ros/calibrate_head_camera.py \
    --bag data/bags/head_calib_a.bag data/bags/head_calib_b.bag
# 推定した値を別の bag で検証する
python3 tools/ros/calibrate_head_camera.py --bag other.bag --no-optimize \
    --head-to-camera 0.02918 0.00703 0.19434 -0.004963 -0.024023 0.012764 0.999618
```

## 実機の動作確認

### `tools/ros/test_base_velocity.py`

台車へ超低速の x 方向速度指令を一定時間送るだけの動作確認用スクリプト。
`estop_node.py` (非常停止ノード、本番として `scripts/ros/` に残してある)
の動作確認用に作成したもので、台車をゆっくり前進させながら AtomS3 の
ボタン (または estop への UDP STOP パケット) で実際に止まるかを確認する。

```bash
python3 tools/ros/test_base_velocity.py
python3 tools/ros/test_base_velocity.py --velocity 0.03 --duration 5.0
```
