# aero_demo

MediaPipe 形式の骨格・掌の
位置姿勢を推定して、viser ビューアで可視化するためのツール群。

## パイプライン

1. **`scripts/generate_random_human_poses.py`**
   SMPL（https://smpl.is.tue.mpg.de/）の体型・姿勢をランダムに生成し、
   MediaPipe と同じ関節名の骨格を組み立てる。SMPLの人モデルと骨格 
   を 1 人分 1 ファイルの JSON として保存する。

2. **`scripts/estimate_palm_poses.py`**
   手順 1 の JSONを入力とし、手のランドマークから左右それぞれの掌の位置
   姿勢を推定してJSON として保存する。
   あわせて、人がどちらの手を差し出しているかを `OfferedHandSelector`
   が判定し、`offered_hand` (`"R"` / `"L"` / `null`) として同じ JSON
   に入れる。

3. (開発用、本番パイプラインの一部ではない) 手順 1・2 の JSON を目視確認
   したい場合は `tools/draw_random_human_poses.py` が使える。詳細は
   [`docs/dev_tools.md`](docs/dev_tools.md) 参照。

4. **`scripts/solve_palm_ik.py`**
   手順 2 の JSONを入力とし、人間の手にロボットが触れる干渉回避付き全身 
   IK (台車移動を含む) を解いて、結果を JSON として保存する。
   ソフトな制約なので、干渉のない解が必ず得られるとは限らない。
   IK に使う腕は人間の手の反対側 (`--robot-arm r`/`l` で上書きできる)。
   1 目標あたりの初期値の数は `--attempts-per-pose` (既定 512)。
   初期値ごとの解は全て (向き 3 通り × 初期値の数) が干渉検証・後処理判定に
   回され、最初に通ったものが採用される。向き 3 通り (ロボットの手首側が
   人間の親指側/回転なし/小指側に来る候補) を試す優先順序は、掌が上を
   向いていれば親指側、甲が上を向いていれば小指側、どちらとも言えない
   (掌がほぼ横向き) 場合は回転なしを最優先にする
   (`solve_palm_ik.turn_candidates_deg` 参照)。干渉回避ペナルティの重み・
   マージンは `--collision-weight`/`--collision-margin`、台車の移動範囲は
   `--base-x-range`/`--base-y-range`/`--base-yaw-range`、乱数初期値の
   再現性は `--seed` で指定する。台車の y 可動範囲は既定で差し出している
   手の側だけに、台車の向き (yaw) は既定で人間の正面方向 ±30°
   (`--base-yaw-facing-margin` で変更可) にそれぞれ人物ごとに制限される
   (`--no-hand-side-base-constraint`/`--no-facing-base-constraint` で
   無効化可)。
   人体側の干渉回避ジオメトリはIK 最適化中の干渉コスト・候補採用前の事後検証
   ・ビューアでの半透明表示のすべてで同じ形状(`Cylinder`)を使う。事後検証
   (候補ごとの `collision_pairs_min_distance` 呼び出し)はこの人体ジオメトリ
   を候補ループの前に 1 回だけ作って使い回す(人物の姿勢は候補間で変わらない
   ため)。

4.5. **`scripts/plan_handshake_motion.py`**
   手順 4 の握手姿勢を目標として、そこへ至る接近の軌道 (waypoints) を干渉回避
   付きで生成し JSON として保存する。

   軌道の始点 (腕を下ろした姿勢 + 台車位置姿勢) は「接近開始位置」(途中目標)。
   人間の手を中心とした円周上 (半径 = 手から最終台車位置までの距離 +
   `--approach-distance`、既定 0.2 m) に置き、そこから手を中心に公転と自転を
   同時に行って人の横に並ぶ経路 (半径が方位角に比例して縮むアルキメデス螺旋、
   終点で減速。公転は人体のある側を通らない向き) を台車の初期軌道にする。円周
   上の位置は、初期位置からの直進 (lead-in) がこの螺旋の出だしの接線になる
   ように探索し、向きは進行方向にする。lead-in は、その場回転で進行方向を
   向いてから直進する (止まらずにそのまま曲がり始める)。手繋ぎの最終姿勢は
   人と同じ方向を向くため、人と向き合った配置ではほぼ半回転が必要になるが、
   それを人から遠い lead-in ではなく人の手の周りでの回り込みの中で行う。
   初期位置から接近開始位置までの lead-in (干渉回避付きの計画・最適化の対象外)
   は、台車が人間の立ち位置から 1 m 以内に入る waypoint だけ干渉を検証し、
   結果 JSON の `lead_in_waypoints`/`lead_in_min_distances`/`lead_in_verified`
   に入れる (1 m より遠い区間は人体に届かないとみなして検証しない)。
   `verified`/`waypoint_min_distances` は従来通り接近開始位置から先の軌道
   (`waypoints`) だけの結果で、`lead_in_verified` には影響しない (経路全体の
   成否は両方を見る必要がある)。
   ロボットが人の背後・横にいると、この lead-in や接近開始位置からの経路が
   人体を横切ってしまうため、上記の位置 (角度 0) で干渉検証に通らなかった
   ときだけ、手を中心に置いた候補 (角度 0 の方向を ±30 度刻みで ±120 度まで
   回した位置) から、lead-in とその先の軌道の両方が干渉検証を通るもののうち
   台車の経路が最短のものを選ぶ (結果 JSON の `approach_angle`。角度 0 で
   通る通常の配置では他の候補は試さず、計算量も従来と同じ)。
   `--initial-base-pose X Y YAW` で初期台車位置を変えて試せる
   (`run_pipeline_test.py` にも同名のオプション)。
   終点の手前には掌の法線方向へ `--pretouch-standoff` (既定 0.25 m)
   引き戻した **pre-touch 姿勢** を挟み、最後の接近を法線方向の直線に
   することで手先が掌を通り抜けないようにする。この幾何的な構成だけで
   干渉が無ければ最適化は行わず、干渉が残った場合だけ scikit-robot の
   `skrobot.planner.trajectory_optimization.TrajectoryProblem`
   で軌道最適化を行う。
   採用前には必ず、`solve_palm_ik.py` が最終姿勢の判定に使うのと同じ
   厳密な形状 (実メッシュ)・同じ人体ジオメトリ (`human_body_obstacles`)
   で全 waypoint を検証し、結果を `verified` フラグに入れる (経路上の
   許容貫通量は既定 1 cm)。人体ジオメトリは人物の姿勢が waypoint 間で
   変わらないため、waypoint ごとに作り直さず検証ループの前に 1 回だけ
   構築して使い回す。

   `--force-optimize` (既定 False) を付けると、pre-touch/線形補間の
   候補が事後検証に通っていても early return せず、必ず jaxls の軌道
   最適化まで実行する。通常運用では最適化を経ずに済むケースがほとんど
   なので、jaxls の軌道最適化そのものの計算時間を単独で計測したいとき
   (ベンチマーク・回帰確認用) に使う。

5. **`scripts/view_handshake_poses.py`**
   手順 1 の骨格 JSONと、手順 4 の IK 結果 JSONを読み込み、SMPL の人体メッシュと
   ロボットモデルの 2 つを viser ビューアで表示する。IK の結果はテキストパネルに出す。
   干渉回避に使ったのと同じ近似ジオメトリを半透明で
   重ねて表示でき、テキストパネルの事後検証 (指先まで含めた貫通の再チェック)
   もこの表示中のメッシュをそのまま使って判定する。

```
generate_random_human_poses.py  (既定の出力先: random_human_poses/)
        │  (skeleton.joint_positions + smpl.pose/betas/root_pos/gender)
        ▼
estimate_palm_poses.py  (既定の入力先: random_human_poses/, 出力先: random_palm_poses/)
        │  (左右の掌の position/rot (または None) + offered_hand)
        ├──▶ (開発用) tools/draw_random_human_poses.py で viser 表示して確認できる
        ▼
solve_palm_ik.py  (既定の入力先: random_palm_poses/, skeleton: random_human_poses/, 出力先: random_handshake_poses/)
        │  (IK 後の台車位置/全関節角/手先姿勢)
        ├──▶ plan_handshake_motion.py  (既定の入力先: random_handshake_poses/, skeleton: random_human_poses/, 出力先: random_motion_poses/)
        │        (接近開始姿勢 -> pre-touch -> 握手姿勢 の waypoint 列 + verified)
        ▼
view_handshake_poses.py  (骨格: random_human_poses/, IK 結果: random_handshake_poses/)
         (SMPL メッシュ + ロボットモデルを viserで表示)
```

`generate_random_human_poses.py`/`estimate_palm_poses.py`/
`solve_palm_ik.py`/`plan_handshake_motion.py`/`view_handshake_poses.py`
の `--input-dir`/`--output-dir`/`--palm-dir`/`--skeleton-dir`/
`--handshake-dir` は、いずれも `scripts/` 直下の
`random_human_poses/`/`random_palm_poses/`/`random_handshake_poses/`/
`random_motion_poses/` が既定値になっているため、指定を省略すれば 1〜5
はそのままつながる ([`docs/dev_tools.md`](docs/dev_tools.md) の開発用
ツール群も、既定ではこれらと同じディレクトリを読み書きする)。

## 環境構築

動作確認環境は Ubuntu 20.04 + Python 3.11。SMPL 経由の合成データパイプライン
(1〜5) とカメラ入力パイプライン (`scripts/ros/run_camera_pipeline_test.py`) の
依存関係はすべて [`pyproject.toml`](pyproject.toml) にまとまっており、
`uv sync` 一回で揃う (venv 内で個別に `pip install` する必要はない)。
バッチIKのバックエンドには jax を使う (コンパイル・永続キャッシュの仕様は
[`docs/jax_compilation_cache.md`](docs/jax_compilation_cache.md) 参照)。

### 0. scikit-robot (fork の `base_limit` ブランチ) を隣に clone する

IK は skrobot の以下の機能に依存しており、これらは上流には入っていないため 
fork を使う:

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
git clone -b base_limit git@github.com:nakane11/scikit-robot.git
```

### 1. venv を作って依存関係を sync する

```bash
uv venv --python 3.11 ~/venv/aero-uv
cd ~/ros/hand/src/aero_demo
UV_PROJECT_ENVIRONMENT=~/venv/aero-uv uv sync
source ~/venv/aero-uv/bin/activate
```

GPU が無い環境では `pyproject.toml` の `dependencies` にある
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

GPU 版 jax は起動時にデバイスメモリの確保を試み、大きいサイズから
確保に失敗するたびに `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` の
警告を出しながら要求サイズを段階的に縮小していくことがある。気になる場合は
`XLA_PYTHON_CLIENT_PREALLOCATE=false` や `XLA_PYTHON_CLIENT_MEM_FRACTION` で
確保量を抑えられる。

### 2. Aero の URDF

`view_handshake_poses.py` が既定で使う`aero_with_feetech_hand.urdf` 
は`feetech_hand` パッケージから持ってくる必要がある。`aero_demo.
aero_urdf_setup.load_aero` (`view_handshake_poses.py`/`tools/view_aero_
collision_model.py` が `Aero(...)` の代わりに使う) が初回呼び出し時に
自動で URDF・メッシュを `~/.skrobot/` 以下に配置するので、**catkin
ワークスペースの source や `ROS_PACKAGE_PATH` は不要**。`feetech_hand`
パッケージが `aero_demo` と同じワークスペースの `src/` 直下にない場合は、
`FEETECH_HAND_DIR` 環境変数でそのディレクトリを指定する。


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

# 3. (開発用) viser で表示して 1./2. の結果を目視確認したい場合
python3 ../tools/draw_random_human_poses.py

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

上記の 3. を含め、grid search・データ収集・ラベル付けなど本番パイプライン
に必須ではない開発・デバッグ用のプログラムは `tools/` (ROS 依存のものは
`tools/ros/`) にまとめてある。一覧・使い方は
[`docs/dev_tools.md`](docs/dev_tools.md) を参照。

## 実カメラ入力: 掌差し出しクリップの収集とカメラ無しでのテスト

実カメラで動かすパイプライン (`scripts/ros/run_camera_pipeline_test.py`)
に加えて、(開発用) `tools/ros/record_palm_offer_clips.py` を常時起動して
おくと、掌の差し出しを検出するたびにその前後を rosbag クリップとして
自動で切り出して保存できる。保存したクリップは `run_camera_pipeline_
test.py` に `--bag` で渡せば、実カメラ・実ロボットの TF 配信なしに
パイプライン全体をそのままテストできる (このデータ収集・ラベル付け系の
開発用ツール一式は [`docs/dev_tools.md`](docs/dev_tools.md) にまとめて
ある)。

### 1. `record_palm_offer_clips.py`: クリップの録画

実カメラ・実ロボットが動いている状態で実行する:

```bash
python3 tools/ros/record_palm_offer_clips.py
# 保存先を変えたい場合
python3 tools/ros/record_palm_offer_clips.py --save-dir /tmp/palm_offer_clips
```

`Ctrl-C` などで止めるまで無期限に動き続け、その間に検出した掌の差し出しを連番のファイル名
で次々に保存し続ける(判定基準は `run_camera_pipeline_test.py` の ARMED 中の判定と同じ
既定値に揃えてある)。掌の差し出しを検出すると、その時刻の `--pre-seconds` 秒前 (既定 2.0) 
から `--post-seconds` 秒後(既定 2.0) までの color/depth/camera_info/tf/tf_static を 1 本の
`.bag` にまとめて `--save-dir` (既定 `palm_offer_clips/`) に保存する。
同じ差し出し動作を 2 回に分けて録らないよう、1 クリップ保存後
`--cooldown-seconds` 秒 (既定 3.0) は次のトリガーを無視する。

保存されるファイルはクリップごとに 3 つ:

```
palm_offer_clips/
    20260911_214500_R_000.bag   # color/depth/camera_info/tf/tf_static (4秒分)
    20260911_214500_R_000.json  # trigger_stamp/offered_hand/pre_seconds/post_seconds/topics/bag_path/snapshot_path
    20260911_214500_R_000.png   # 差し出しを検出した瞬間のカラー画像 + 骨格描画 (差し出し手は赤)
```

`~skeleton_image` には、購読者がいれば骨格・掌の有無や録画中かどうかに
関わらず毎フレーム同じ骨格描画画像を publish する (差し出し手は赤で描く)。
クリップを保存し終えてからの `--cooldown-seconds` の間だけ publish を
止める。`rqt_image_view` 等で購読すれば、常時の検出状況とどのフレームが
実際にクリップへ書き込まれているかをリアルタイムに確認できる。

差し出し手判定の基準にするロボット手先の base_link 座標は、既定では
`--robot-hand-frame` (既定 `r_eef_grasp_link`、skrobot Aero モデルの
`rarm_end_coords` に対応する実リンクで、実機では `/aero_state_publisher`
が配信する) を毎フレーム TF で引いて使う (`estimate_palm_poses.
OfferedHandSelector` 自身の既定動作 (未指定/`None`) は合成骨格向けの
「人物より world +x 側にロボットがいる」という前提で、実カメラ・
base_link 座標系ではロボット自身がおよそ原点付近 (=人物より -x 側)
にいることが多く食い違うため使わない)。ロボット未接続などでまだ TF が
引けない間だけ概算値 `(0.32, -0.55, 0.93)` にフォールバックする。固定値を
明示したい場合は `--robot-hand-position X Y Z` で上書きできる (指定すると
TF 解決より優先される)。ARM を押しても差し出し手が見つからないときと
同じ理由で判定が届かない場合は `--offer-score-min` (既定 0.65) を調整する。

### 2. `run_camera_pipeline_test.py --bag`: 保存したクリップで実カメラ無しテスト

```bash
python3 scripts/ros/run_camera_pipeline_test.py \
    --bag palm_offer_clips/20260911_214500_R_000.bag \
    --auto-arm --no-wait-for-client
```

- `--bag <path>`: 実カメラの代わりに指定した rosbag を再生する
  (内部で `rosbag play --clock` をサブプロセスとして起動し、
  `/use_sim_time` をあわせて有効にする)。`--bag-rate` で再生速度、
  `--bag-loop` で繰り返し再生を指定できる。
- `--auto-arm`: viser 画面の ARM ボタンを押す代わりに起動直後から
  ARMED 状態にする (無人でのバッグ再生テスト用)。
- `--no-wait-for-client`: viser のブラウザクライアント接続を待たずに
  起動を続ける (既定では接続まで無期限に待つため、無人テストでは必須)。
  表示自体は見たい場合はこのオプションを外して普段どおり viser の URL
  をブラウザで開けばよい。
- `--force-optimize` (既定 False): 実際の握手試行でも pre-touch/線形
  補間の候補が事後検証に通っていても early return せず、必ず jaxls の
  軌道最適化まで実行させる (`plan_handshake_motion.py` の同名オプション
  参照)。通常運用では付けない -- jaxls の計算時間そのものを単独で
  計測したいベンチマーク用。

`--auto-execute` を指定しなければ実機は一切動かさず viewer 上での
確認のみになるので、このテストに実ロボットは不要 (`--bag` のクリップに
`tf`/`tf_static` も含めているため、実ロボットの TF 配信も不要)。

### 3. `--auto-execute`: 実機での実行速度

`--auto-execute` で実機を動かすとき、waypoint 間の所要時間は固定の
刻み幅ではなく、区間ごとに `run_camera_pipeline_test.py` 冒頭の定数から
決める (`_limited_time_list`)。軌道計画の `dt`
(`plan_handshake_motion.DEFAULT_DT`) は最適化のコスト正規化に使うだけで、
実行速度には関係しない (以前あった `--dt` 引数は実際にはどこからも
使われていなかったため削除した)。

- 速度の上限: 台車は `BASE_MAX_VEL`/`BASE_MAX_ANGVEL` (並進 0.3 m/s・
  回頭 1.0 rad/s、ロボット本体側の `aero_base_link.yaml` の
  `max_velocity` と必ず一致させる)、腕・首・腰・リフターは URDF の
  `velocity`。指令ではこれに `VEL_LIMIT_RATIO` (既定 0.9) を掛けた値まで
  しか使わない。
- 加速度の上限: 上記の速度まで `ACCEL_TIME` 秒 (既定 0.4) かけて加速する
  値。大きくするほどゆるやかに加減速するが、全体の所要時間は延びる。
- 各区間の所要時間は、その区間で律速する軸 (台車の並進・回頭、各関節)
  の速度または加速度がちょうど上限になる長さにする。実機のコントローラ
  (腕は JointTrajectoryController、台車は pr2_base_trajectory_action) が
  各点の位置・速度から 3 次補間する際の瞬間値で判定するので、上限を超えた
  指令を実機側で頭打ちにされることはない。ほとんど動かない区間は
  `MIN_SEGMENT_TIME` (1/15 秒) を下限にする。
- 台車と腕には同じ時間列を送り、律速しない側はその区間だけ上限より遅く
  動く (タイミングがずれて干渉検証済みの経路から外れないようにするため)。
- 台車の各点の速度は腕と同じ規則 (始点・終点 0、途中は前後区間の平均
  速度の平均) で付け直して送る (`_send_base_trajectory`)。skrobot の
  `move_trajectory_sequence` のままだと、静止状態から最初の区間の速度へ
  いきなり跳ぶため。

実行時は区間ごとの所要時間と律速した軸が次のようにログに出る
(`〜の加速度` は加速度で、`下限` は `MIN_SEGMENT_TIME` で決まった区間):

```
[debug][segment] 速度上限 (×0.9, 加速 0.4s) で決めた所要時間 (合計 8.94s): 0:0.35s(base_yawの加速度), 1:0.31s(base_yawの加速度), ..., 19:0.13s(ankle_joint), ...
```

## 座標系

各 JSON の座標はロボット座標系 (x=前, y=左, z=上) で、両足が地面
(z=0) についた状態で保存される。掌のローカル座標系は
+x = 指先方向 (手首→指先)、+y = 手の甲→掌の方向 (掌の法線)、
+z = x × y。
