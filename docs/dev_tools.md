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
                    plan_handshake_motion.py/view_handshake_poses.py/
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
既定で書き出す `scripts/random_human_poses/`/`scripts/random_palm_poses/`
等をそのまま指すようにしてある (指定を省略すれば `scripts/` 側で
`generate_random_human_poses.py`/`estimate_palm_poses.py` 等を実行した
結果をそのまま読める)。

## パラメータ探索・チューニング

### `tools/grid_search_collision_ik.py`

`solve_palm_ik.py` の速度・成功率トレードオフ (初期値の数・干渉回避ペア
設定・最大反復回数・台車の可動域半幅・向き候補数・後処理 IK の閾値等) を
グリッドサーチし、段階A (事後干渉検証まで)・段階B (押し込み・視線の IK
まで) の成功率・所要時間でランキング表示する (ファイル出力はしない)。

```bash
python3 tools/grid_search_collision_ik.py \
    --attempts-per-pose 16 64 \
    --collision-ik-stop 100 500 \
    --collision-pairs none-gd collision_pairs.json
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

## 実カメラデータの収集・ラベル付け

以下の 3 つは、実カメラでの `offered_hand` (差し出し手) 判定
(`estimate_palm_poses.OfferedHandSelector`) の精度を上げるための、
データ収集 → ラベル付け → チューニングの一連の流れで使う。

### 1. `tools/ros/extract_skeletons_from_bag.py`

判定器に依存しない生の rosbag (`rosbag record` で連続録画したもの) を
読み込み、`--sample-interval` 秒おきに骨格・掌位置姿勢・骨格重畳画像を
機械的にサンプリングして保存するオフライン抽出ツール (roscore 不要)。
`tools/ros/record_palm_offer_clips.py` が切り出した判定済みクリップを
`--single-sample` で混ぜることもできる。

```bash
python3 tools/ros/extract_skeletons_from_bag.py \
    --bag session1.bag --output-dir /tmp/offer_dataset
```

保存先には `skeletons/`/`palms/`/`images/` の 3 サブディレクトリができる
(`skeletons/`/`palms/` はそれぞれ `estimate_palm_poses.py`/
`tune_offer_selector.py` の入力形式と互換)。

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
