# aero_demo

SMPL の人体モデルからランダムな姿勢を生成し、MediaPipe 形式の骨格・掌の
位置姿勢を推定して、viser ビューアで可視化するためのツール群。

## パイプライン

1. **`scripts/generate_random_human_poses.py`**
   SMPL（https://smpl.is.tue.mpg.de/）の体型・姿勢をランダムに生成し、
   MediaPipe と同じ関節名の骨格を組み立てる。SMPLの人モデルと骨格 
   を 1 人分 1 ファイルの JSON として保存する。

2. **`scripts/estimate_palm_poses.py`**
   手順 1 の JSON（骨格）を入力とし、手のランドマークから左右それぞれの掌の位置
   姿勢を推定してJSON として保存する。
   あわせて、人がどちらの手を差し出しているかを `OfferedHandSelector`
   が判定し、`offered_hand` (`"R"` / `"L"` / `null`) として同じ JSON
   に入れる。

3. **`scripts/draw_random_human_poses.py`**
   手順 1・2 で生成した JSON を読み込み、SMPLの人体メッシュ・骨格・左右
   の掌の座標系をビューアで表示する。手繋ぎに使うと判定された手を赤で描く。

4. **`scripts/solve_palm_ik.py`**
   手順 2 の JSON (掌の位置姿勢) を入力とし、人間の手にロボットが触れる
   干渉回避付き全身 IK (台車移動 `use_base='planar'` を含む) を解いて、
   結果 (台車位置・全関節角・実際の手先姿勢) を JSON として保存する。
   ソフトな制約なので、干渉のない解が必ず得られるとは限らない。
   IK に使う腕は人間の手の反対側 (`--robot-arm r`/`l` で上書きできる)。
   IK は 1 人ずつ、複数初期値からの並列 IK で解く。
   1 目標あたりの初期値の数は `--attempts-per-pose` (既定 512、この数を
   増やしても計算時間はあまり増えない)。
   初期値ごとの解は全て (向き 3 通り × 初期値の数) が干渉検証・後処理判定に
   回され、最初に通ったものが採用される。干渉回避ペナルティの重み・
   マージンは `--collision-weight`/`--collision-margin`、台車の移動範囲は
   `--base-x-range`/`--base-y-range`/`--base-yaw-range`、乱数初期値の
   再現性は `--seed` で指定する。
   人体側の干渉回避ジオメトリ (体幹・頭部・四肢・掌・指、いずれも
   `Cylinder`) は `human_body_obstacles` が一手に作り、IK 最適化中の
   干渉コスト・候補採用前の事後検証 (`collision_pairs_min_distance`)・
   ビューアでの半透明表示 (`view_handshake_poses.py` 等) のすべてで
   同じ関数・同じ形状を使う (掌のような骨の線分では表せない部位も含めて
   一致させてあるので、表示と判定の間に見た目の食い違いは生じない)。

4.5. **`scripts/plan_handshake_motion.py`**
   手順 4 の握手姿勢 (最終姿勢 1 点) を目標として、そこへ至る「最後の接近」
   の軌道 (waypoint 列) を干渉回避付きで生成し JSON として保存する。
   遠方からの長距離走行はナビゲーションの担当として計画対象にせず、軌道の
   始点は「腕を下ろした姿勢 (`Aero.reset_pose`) + 最終台車位置から人間の
   反対方向へ `--approach-distance` (既定 1.0 m) 下がった位置」にする。
   終点の手前には掌の法線方向へ `--pretouch-standoff` (既定 0.25 m)
   引き戻した **pre-touch 姿勢** を挟み、最後の接近を法線方向の直線に
   することで手先が掌を通り抜けないようにする。この幾何的な構成だけで
   干渉が無ければ最適化は行わず (実測では多くの人物がこれで通る)、
   干渉が残った場合だけ scikit-robot の
   `skrobot.planner.trajectory_optimization.TrajectoryProblem`
   (台車 3 自由度 + 腕、`jaxls` バックエンド) で軌道最適化を行う。
   採用前には必ず、`solve_palm_ik.py` が最終姿勢の判定に使うのと同じ
   厳密な形状 (実メッシュ)・同じ人体ジオメトリ (`human_body_obstacles`)
   で全 waypoint を検証し、結果を `verified` フラグに入れる (経路上の
   許容貫通量は既定 1 cm)。軌道最適化のコスト関数に渡す人体障害物も
   同じ `human_body_obstacles` から作るので、最適化・事後検証・表示の
   3 者で人体側の形状が食い違うことはない。
   `jaxls` (PyPI に無い) が別途必要:
   `pip install "git+https://github.com/brentyi/jaxls.git"`。

5. **`scripts/view_handshake_poses.py`**
   手順 1 の骨格 JSONと、手順 4 の IK 結果 JSONを読み込み、SMPL の人体メッシュと
   ロボットモデルの 2 つを viser ビューアで表示する。IK の結果はテキストパネルに出す。
   干渉回避に使ったのと同じ近似ジオメトリ (ロボット・人体とも) を半透明で
   重ねて表示でき、テキストパネルの事後検証 (指先まで含めた貫通の再チェック)
   もこの表示中のメッシュをそのまま使って判定するので、見た目と判定結果が
   食い違うことはない (`scripts/ros/run_camera_pipeline_test.py` の骨格
   プレビュー画面も同じ仕組み)。

```
generate_random_human_poses.py  (既定の出力先: random_human_poses/)
        │  (skeleton.joint_positions + smpl.pose/betas/root_pos/gender)
        ▼
estimate_palm_poses.py  (既定の入力先: random_human_poses/, 出力先: random_palm_poses/)
        │  (左右の掌の position/rot (または None) + offered_hand)
        ├──▶ draw_random_human_poses.py  (既定の入力先: random_human_poses/, 掌: random_palm_poses/)
        │        (viserで表示)
        ▼
solve_palm_ik.py  (既定の入力先: random_palm_poses/, skeleton: random_human_poses/, 出力先: random_handshake_poses/)
        │  (IK 後の台車位置/全関節角/手先姿勢)
        ├──▶ plan_handshake_motion.py  (既定の入力先: random_handshake_poses/, skeleton: random_human_poses/, 出力先: random_motion_poses/)
        │        (接近開始姿勢 -> pre-touch -> 握手姿勢 の waypoint 列 + verified)
        ▼
view_handshake_poses.py  (骨格: random_human_poses/, IK 結果: random_handshake_poses/)
         (SMPL メッシュ + ロボットモデルを viserで表示)
```

`generate_random_human_poses.py`/`estimate_palm_poses.py`/`draw_random_
human_poses.py`/`solve_palm_ik.py`/`plan_handshake_motion.py`/
`view_handshake_poses.py` の
`--input-dir`/`--output-dir`/`--palm-dir`/`--skeleton-dir`/
`--handshake-dir` は、いずれも `scripts/` 直下の
`random_human_poses/`/`random_palm_poses/`/`random_handshake_poses/`/
`random_motion_poses/` が既定値になっているため、指定を省略すれば 1〜5
はそのままつながる。

## 環境構築 (IK を解くために必要なもの)

動作確認環境は Ubuntu 20.04 + Python 3.11。SMPL 経由の合成データパイプライン
(1〜5) とカメラ入力パイプライン (`scripts/ros/run_camera_pipeline_test.py`) の
依存関係はすべて [`pyproject.toml`](pyproject.toml) にまとまっており、
`uv sync` 一回で揃う (venv 内で個別に `pip install` する必要はない)。
バッチIKのバックエンドには jax を使う。

パッケージ管理には [uv](https://docs.astral.sh/uv/) を使う (`uv` は Python
本体のダウンロード・インストールも自動でやってくれるため、事前に
deadsnakes PPA 等で Python 3.11 を用意しなくてもよい)。
`pyproject.toml` の `[tool.uv.sources]` (scikit-robot fork の path 参照、
jaxls の git 参照) は uv 固有の機能なので、素の `pip install` では揃わない。

### 0. scikit-robot (fork の `aero` ブランチ) を隣に clone する

IK は skrobot の以下の機能に依存しており、これらは上流
(`iory/scikit-robot`) には入っていないため fork を使う:

* `skrobot.models.Aero` (`use_hand` 引数付き) と
  `skrobot.data.aero_urdfpath`
* `batch_inverse_kinematics` の `use_base='planar'` +
  `base_limits` (台車の平面移動を含む全身バッチ IK と、その移動範囲の
  指定 -- `--base-x-range`/`--base-y-range`/`--base-yaw-range` はこれを
  渡している)
* `batch_inverse_kinematics` の `collision_link_list`/
  `collision_obstacles` (干渉回避付きバッチ IK。
  `backend='jax'` の勾配降下法でしか使えない)

`pyproject.toml` が `../scikit-robot` を editable install するので、
`aero_demo` と同じワークスペースの `src/` 直下に clone しておく:

```bash
cd ~/ros/hand/src
git clone -b aero git@github.com:nakane11/scikit-robot.git
# すでに clone 済みなら: cd scikit-robot && git checkout aero
```

### 1. venv を作って依存関係を sync する

```bash
uv venv --python 3.11 ~/venv/aero-uv
cd ~/ros/hand/src/aero_demo
UV_PROJECT_ENVIRONMENT=~/venv/aero-uv uv sync
source ~/venv/aero-uv/bin/activate
```

これで scikit-robot (editable)・jax (`jax[cuda12]`)・numpy・opencv-python・
mediapipe・rospkg 等、SMPL パイプラインとカメラパイプライン両方の依存が
まとめて入る。GPU が無い環境では `pyproject.toml` の `dependencies` にある
`"jax[cuda12]>=0.10"` を `"jax>=0.10"` に変更してから sync し直す。

`plan_handshake_motion.py` の軌道最適化バックエンド (`jaxls`) は PyPI に
無いため既定では入らない。使う場合は `--extra motion` を付けて sync する:

```bash
UV_PROJECT_ENVIRONMENT=~/venv/aero-uv uv sync --extra motion
```

依存関係を追加・変更したくなったら `pyproject.toml` の `dependencies`
(または `optional-dependencies`) を編集して同じ `uv sync` を再実行すればよい。
venv ごと作り直したい場合も、上の2ステップ (`uv venv` → `uv sync`) を
やり直すだけで復元できる。

なお `rospy` 経由の ROS 本体 (`tf2_ros`/`message_filters`/
`sensor_msgs` 等) は ROS Noetic のシステムインストール由来なので、
`scripts/ros/*.py` を実行する前には venv の activate に加えて
`source /opt/ros/noetic/setup.bash` (ROS ワークスペースの setup.bash) も
必要。

GPU 版 jax は起動時にデバイスメモリの確保を試み、大きいサイズから
確保に失敗するたびに `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` の
警告を出しながら要求サイズを段階的に縮小していくことがある。気になる場合は
`XLA_PYTHON_CLIENT_PREALLOCATE=false` や `XLA_PYTHON_CLIENT_MEM_FRACTION` で
確保量を抑えられる。

### 2. Aero の URDF

`solve_palm_ik.py` は`Aero(use_hand=False)` =
`aero_nohand.urdf` を使う。これは初回実行時に自動ダウンロードして
`~/.skrobot/aero_description/typeJSK/urdf/` に展開するため、手作業は不要。

一方 `view_handshake_poses.py` が既定で使う`aero_with_feetech_hand.urdf` 
はこの tarball に含まれていないので、`feetech_hand` パッケージから持ってくる
必要がある。`aero_demo.
aero_urdf_setup.load_aero` (`view_handshake_poses.py`/`view_aero_
collision_model.py` が `Aero(...)` の代わりに使う) が初回呼び出し時に
自動で URDF・メッシュを `~/.skrobot/` 以下に配置するので、**catkin
ワークスペースの source や `ROS_PACKAGE_PATH` は不要**。`feetech_hand`
パッケージが `aero_demo` と同じワークスペースの `src/` 直下にない場合は、
`FEETECH_HAND_DIR` 環境変数でそのディレクトリを指定する。


### 3. jax の永続コンパイルキャッシュ

`solve_palm_ik.py` は起動時に jax の永続コンパイルキャッシュ (既定で
`~/.cache/jax_compilation_cache`, `JAX_COMPILATION_CACHE_DIR` 環境変数で
変更できる) を有効にする。干渉回避付きバッチ IK の JIT コンパイルは
計算グラフの形状 (`--collision-ik-stop`/`--attempts-per-pose`/
`--skeleton-dir` の有無/使う腕 (`--robot-arm`) などで決まる) ごとに
初回だけ必要な重い処理 (数分〜十数分かかることがある) で、以降は同じ
venv/jax バージョンで同じ形状の計算であればディスクキャッシュから
即座に読み込まれる。

## 使い方

SMPL のモデルファイル (`.pkl`) はライセンス上リポジトリに同梱されていない
ため、各スクリプトの `--model-path` / `--female-model-path` で指定する
(既定値は `~/SMPL_python_v.1.0.0/smpl/models/` 以下)。女性モデルが無ければ
男性モデルのみで続行する。

```bash
cd scripts

# 1. ランダムな人物姿勢を 100 体生成 (既定の保存先 scripts/random_human_poses/ に保存)
python3 generate_random_human_poses.py --num-samples 100

# 2. 骨格から左右の掌の位置姿勢を推定 (既定で 1. の出力を読み、scripts/random_palm_poses/ に保存)
python3 estimate_palm_poses.py

# 3. viser で表示 (既定で 1./2. の出力を読む)
python3 draw_random_human_poses.py

# 4. 人が差し出していると判定された手 (2. の offered_hand) に、その反対側の
#    ロボットの腕で触れる全身 IK を解く (offered_hand が null の人物は対象外)
python3 solve_palm_ik.py

# 4.5. 握手姿勢へ至る「最後の接近」の軌道を干渉回避付きで生成 (4. の結果を目標にする)
python3 plan_handshake_motion.py

# 5. SMPL メッシュ + ロボットモデルを viser で表示 (4. の結果と、1. の骨格を対応づける)
python3 view_handshake_poses.py
```

`scripts/run_pipeline_test.py` は 1/2/4 (と `--viewer` 指定時は 5) を
順に実行する回帰テストで、`--plan-motion` を付けると 4.5 も実行して
「経路上の干渉も含めて検証できた人数 (verified)」を集計に加える。この
とき `--viewer` も指定すると、5 は `view_handshake_poses.py` の代わりに
軌道を再生できる `view_handshake_motion.py` を開く。

保存先を変えたい場合は、各スクリプトの `--input-dir`/`--output-dir`/
`--palm-dir`/`--skeleton-dir`/`--handshake-dir` で明示的に指定できる
(例: `--output-dir /tmp/random_human_poses`)。`solve_palm_ik.py` が対象に
する人間の手は掌 JSON の `offered_hand` で決まる。
使うロボットの腕は `--robot-arm` で変更できる (既定のは人間の手の
反対側)。

`draw_random_human_poses.py` は加えて
`--advance-mode auto` にすると `--pause` 秒ごとに自動で次の人物へ進み、
`--output-dir` を指定すると表示した各姿勢の画像をその都度保存する

## 座標系

各 JSON の座標はロボット座標系 (x=前, y=左, z=上) で、両足が地面
(z=0) についた状態で保存される。掌のローカル座標系は
+x = 指先方向 (手首→指先)、+y = 手の甲→掌の方向 (掌の法線)、
+z = x × y。
