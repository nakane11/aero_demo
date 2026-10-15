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
   IK に使う腕は人間の手の反対側。
   干渉回避付きバッチ IK → 干渉の事後検証 → 掌へ押し込む後処理 IK の
   二段階で、制約・重み・解の優先順位・棄却条件は
   [`docs/ik_and_motion_constraints.md`](docs/ik_and_motion_constraints.md)
   にまとめてある。

4.5. **`scripts/plan_handshake_motion.py`**
   手順 4 の握手姿勢 (hover) を目標として、そこへ至る接近の軌道 (waypoints) を
   干渉回避付きで生成し JSON として保存する。台車は人の手を中心に公転+自転して
   人の横に並び、腕は pre-touch 姿勢を経由して寄る。幾何的に作った軌道で干渉が
   残ったときだけ jaxls で最適化し、全 waypoint を事後検証して `verified` に入れる
   (初期位置から接近開始位置までの直進は `lead_in_verified`)。
   `--initial-base-pose X Y YAW` で初期台車位置を変えて試せる
   (`run_pipeline_test.py` にも同名のオプション)。
   `--side-by-side-transition` を付けると、押し込み (掌を合わせた姿勢) の後に、
   つないだ手を人が肩から体の横へ下ろすのに合わせてロボットも人と横並びになる
   区間も計画し、軌道 JSON の `transition` に保存する
   (`scripts/side_by_side_transition.py`)。
   `--force-optimize` は jaxls の軌道最適化の計算時間を単独で計測するための
   ベンチマーク用オプションで、通常は付けない
   ([`docs/jax_compilation_cache.md`](docs/jax_compilation_cache.md) 11 節)。
   接近開始位置の候補・優先順位・最適化の重み・棄却条件・横並び移動の決め方は
   [`docs/ik_and_motion_constraints.md`](docs/ik_and_motion_constraints.md)、
   段階ごとの図と数値は
   [`docs/handshake_pipeline_stages.md`](docs/handshake_pipeline_stages.md) 参照。

5. **`scripts/view_handshake_poses.py`**
   手順 1 の骨格 JSONと、手順 4 の IK 結果 JSONを読み込み、SMPL の人体メッシュと
   ロボットモデルの 2 つを viser ビューアで表示する。IK の結果はテキストパネルに出す。
   干渉回避に使ったのと同じ近似ジオメトリを半透明で
   重ねて表示でき、テキストパネルの事後検証 (指先まで含めた貫通の再チェック)
   もこの表示中のメッシュをそのまま使って判定する。
   軌道 (手順 4.5) を再生するときは `scripts/view_handshake_motion.py` を使う。

```
generate_random_human_poses.py  (既定の出力先: data/synthetic/random_human_poses/)
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
の `--input-dir`/`--output-dir`/`--skeleton-dir`/`--handshake-dir` は、
いずれも `data/synthetic/` 直下の
`random_human_poses/`/`random_palm_poses/`/`random_handshake_poses/`/
`random_motion_poses/` が既定値になっているため、指定を省略すれば 1〜5
はそのままつながる ([`docs/dev_tools.md`](docs/dev_tools.md) の開発用
ツール群も、既定ではこれらと同じディレクトリを読み書きする)。
`data/` は git 管理外で、実機の bag などテストデータの置き場所と一覧は
[`docs/test_data.md`](docs/test_data.md) にまとめてある。

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
  指定 -- `run_camera_pipeline_test.py` の
  `--base-x-range`/`--base-y-range`/`--base-yaw-range` はこれを渡している)
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
`"jax[cuda12]==0.10.2; ..."`/`"jax[cuda12]==0.11.1; ..."` の 2 行の
`jax[cuda12]` を `jax` に変更してから sync し直す。

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

# 1. ランダムな人物姿勢を 100 体生成 (既定の保存先 data/synthetic/random_human_poses/ に保存)
python3 generate_random_human_poses.py --num-samples 100

# 2. 骨格から左右の掌の位置姿勢を推定 (既定で 1. の出力を読み、data/synthetic/random_palm_poses/ に保存)
python3 estimate_palm_poses.py

# 3. (開発用) viser で表示して 1./2. の結果を目視確認したい場合
python3 ../tools/draw_random_human_poses.py

# 4. 人が差し出していると判定された手 (2. の offered_hand) に、その反対側の
#    ロボットの腕で触れる全身 IK を解く (offered_hand が null の人物は対象外)
python3 solve_palm_ik.py

# 4.5. 握手姿勢へ至る「最後の接近」の軌道を干渉回避付きで生成 (4. の結果を目標にする)
#      押し込み後の横並び移動も計画するなら --side-by-side-transition を付ける
python3 plan_handshake_motion.py

# 5. SMPL メッシュ + ロボットモデルを viser で表示 (4. の結果と、1. の骨格を対応づける)
python3 view_handshake_poses.py
```

`scripts/run_pipeline_test.py <人数>` は 1/2/4 (と `--viewer` 指定時は 5) を
順に実行する回帰テストで、毎回 `/tmp/aero_demo_pipeline_*` に作業ディレクトリを
作る (`--seed` で再現できる)。`--plan-motion` を付けると 4.5 も実行して
「経路上の干渉も含めて検証できた人数 (verified)」を集計に加え、
`--side-by-side-transition` を併せて付けると横並び移動も計画する。
`--plan-motion --viewer` なら、5 は `view_handshake_poses.py` の代わりに
軌道を再生できる `view_handshake_motion.py` を開く (横並び移動は押し込みに続けて
再生され、人の腕もロボットの手に追従する)。
`--base-x-standing-margins`/`--front-offset-weight`/`--facing-yaw-weight` は
そのまま `solve_palm_ik.py` に渡される (最終台車位置の比較に使う)。
各段階の図と数値は
[`docs/handshake_pipeline_stages.md`](docs/handshake_pipeline_stages.md)。

保存先を変えたい場合は、各スクリプトの `--input-dir`/`--output-dir`/
`--skeleton-dir`/`--handshake-dir` で明示的に指定できる
(例: `--output-dir /tmp/random_human_poses`)。`solve_palm_ik.py` が対象に
する人間の手は掌 JSON の `offered_hand` で決まる。
使うロボットの腕は人間の手の反対側。

上記の 3. を含め、grid search・データ収集・ラベル付けなど本番パイプライン
に必須ではない開発・デバッグ用のプログラムは `tools/` (ROS 依存のものは
`tools/ros/`) にまとめてある。一覧・使い方は
[`docs/dev_tools.md`](docs/dev_tools.md) を参照。

## 実カメラ入力

実カメラ・実ロボットで動かすのは `scripts/ros/run_camera_pipeline_test.py`。
カメラ画像から骨格・掌を推定し、ARM を押す (または `--auto-arm`) と差し出し手の
判定 → IK → 軌道計画 → 横並び移動の計画までを行い、`--auto-execute` を付けたときだけ
実機 (台車・腕・首) を動かす (付けなければ viewer で確認するだけ)。
横並び移動をする設定 (既定) では、横並び移動が解けていない試行は実機を動かさない
(`--no-side-by-side-transition` で押し込みまでにできる)。

詳細ログ (`[debug]` など) は画面には出さず、
`/tmp/run_camera_pipeline_test_logs/<起動時刻>/` に起動時 (`startup.log`) と
試行 (人) ごと (`person_01.log`, ...) のファイルで保存する
(起動ごとに別のディレクトリで、最新は `latest` からたどれる)。

rosbag を再生すれば実カメラ・実ロボットなしで同じパイプラインを試せる:

```bash
python3 scripts/ros/run_camera_pipeline_test.py \
    --bag data/palm_offer_clips/20260911_214500_R_000.bag \
    --auto-arm --no-wait-for-client
```

ただし現在の実装は `--bag` を付けると初期化中に `/clock` が来ず止まることがある
(未修正、回避策は [`docs/jax_compilation_cache.md`](docs/jax_compilation_cache.md) 6 節
と各試行の `replay.sh`)。

次の詳細は [`docs/dev_tools.md`](docs/dev_tools.md) にまとめてある。

- 掌の差し出しを自動で切り出して保存する `tools/ros/record_palm_offer_clips.py` と、
  保存したクリップを使う `--bag`/`--auto-arm`/`--no-wait-for-client` の使い方
- `--auto-execute` で実機を動かすときの再生速度の決め方 (`BASE_MAX_VEL` など)・
  横並び移動の実行・ログの読み方

テストに使う実機データの一覧は [`docs/test_data.md`](docs/test_data.md)。

## 座標系

各 JSON の座標はロボット座標系 (x=前, y=左, z=上) で、両足が地面
(z=0) についた状態で保存される。掌のローカル座標系は
+x = 指先方向 (手首→指先)、+y = 手の甲→掌の方向 (掌の法線)、
+z = x × y。
