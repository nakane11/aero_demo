# aero_demo

SMPL の人体モデルからランダムな姿勢を生成し、MediaPipe 形式の骨格・掌の
位置姿勢を推定して、viser ビューアで可視化するためのツール群。

## パイプライン

1. **`scripts/generate_random_human_poses.py`**
   SMPL (Skinned Multi-Person Linear model) の体型・姿勢をランダムに生成し、
   その姿勢済み関節位置から MediaPipe と同じ関節名の骨格 (手のランドマーク
   `RHand0`..`RHand20` / `LHand0`..`LHand20` を含む) を組み立てる。SMPL の
   人モデル (`pose`/`betas`/`root_pos`/`gender`) と骨格 (`joint_positions`/
   `height`) の両方を 1 人分 1 ファイルの JSON として保存する。

2. **`scripts/estimate_palm_poses.py`**
   手順 1 の JSON (骨格の `joint_positions`) を入力とし、手首・知節 (MCP)
   のランドマークから左右それぞれの掌の位置姿勢 (位置と回転行列) を推定して
   JSON として保存する。手のランドマークが 3 点未満の側は `None` になる。
   骨格の生成元 (合成骨格でも実カメラの MediaPipe 推定でも) を問わず同じ
   ロジックで動く。
   あわせて、実際に手を繋ぐときにどちらの手を取るべきか (人がどちらの手を
   差し出しているか) を `OfferedHandSelector` が判定し、`offered_hand`
   (`"R"` / `"L"` / `null`) として同じ JSON に入れる。判定は「体の前への
   出方」「掌の向き」「腕の挙上」「肘の伸び」「手繋ぎとして無理のない
   高さ」の重み付き和で、どちらの手も基準に届かなければ `null`
   (差し出していない)。

3. **`scripts/draw_random_human_poses.py`**
   手順 1・2 で生成した JSON を読み込み、SMPL の人体メッシュ・骨格・
   (あれば) 左右の掌の座標系を viser ビューアで重ねて表示する。
   掌 JSON があれば `offered_hand` も読み、手繋ぎに使うと判定された手の
   骨格・ランドマークを赤、選ばれなかった手を白で描き分ける
   (`offered_hand` が `null` の人物は両手とも白、掌 JSON が無い人物だけ
   従来どおり右手=赤・左手=青)。

4. **`scripts/solve_palm_ik.py`**
   手順 2 の JSON (掌の位置姿勢) を入力とし、人間の手 (既定: 左手) に
   ロボット (Aero) の腕 (既定: 右腕) が触れる全身 IK (台車の平面移動
   `use_base='planar'` を含む) を解いて、結果 (台車位置・全関節角・
   実際の手先姿勢) を JSON として保存する。干渉回避や `rotation_axis`
   を段階的に緩める再試行など、実機を安全に動かすための機能は持たない
   簡易版 (詳細はスクリプト内の docstring 参照)。

5. **`scripts/view_handshake_poses.py`**
   手順 1 の骨格 JSON (SMPL の `pose`/`betas`/`root_pos`/`gender`) と、
   手順 4 の IK 結果 JSON (`joint_angle_vector`/`base_position`/
   `base_yaw`) をファイル名で対応づけて読み込み、SMPL の人体メッシュと
   ロボット (Aero) モデルの 2 つだけを viser ビューアで並べて表示する
   (`draw_random_human_poses.py` と違い、骨格線・掌 Axis・ランドマーク
   などは描かない)。

```
generate_random_human_poses.py  (既定の出力先: random_human_poses/)
        │  (skeleton.joint_positions + smpl.pose/betas/root_pos/gender)
        ▼
estimate_palm_poses.py  (既定の入力先: random_human_poses/, 出力先: random_palm_poses/)
        │  (左右の掌の position/rot (または None) + offered_hand)
        ├──▶ draw_random_human_poses.py  (既定の入力先: random_human_poses/, 掌: random_palm_poses/)
        │        (viser ブラウザビューアで表示)
        ▼
solve_palm_ik.py  (既定の入力先: test_palm_pose_pipeline/palms/, 出力先: test_palm_pose_pipeline/handshakes/)
        │  (IK 後の台車位置/全関節角/手先姿勢)
        ▼
view_handshake_poses.py  (骨格: test_palm_pose_pipeline/skeletons/, IK 結果: test_palm_pose_pipeline/handshakes/)
         (SMPL メッシュ + ロボットモデルを viser ブラウザビューアで表示)
```

`generate_random_human_poses.py`/`estimate_palm_poses.py`/`draw_random_
human_poses.py` の `--input-dir`/`--output-dir`/`--palm-dir` は
`scripts/` 直下の `random_human_poses/`/`random_palm_poses/` が既定値に
なっているため、指定を省略すればこの区間はそのままつながる。一方
`solve_palm_ik.py`/`view_handshake_poses.py` は既定で `scripts/
test_palm_pose_pipeline/` 以下の `palms/`/`skeletons/`/`handshakes/` を
読み書きする (`scripts/test_generate_and_estimate_palm_poses.py` が書き出す
固定サンプル用のディレクトリで、`random_palm_poses/`/`random_human_
poses/` とは別物)。1〜4 を通しで `random_*` ディレクトリに保存したまま
5 の `view_handshake_poses.py` に渡したい場合は、`--skeleton-dir
random_human_poses --handshake-dir random_handshake_poses` のように
明示的に指定する必要がある (詳細は下記使い方を参照)。

## 使い方

SMPL のモデルファイル (`.pkl`) はライセンス上リポジトリに同梱されていない
ため、各スクリプトの `--model-path` / `--female-model-path` で指定する
(既定値は `~/SMPL_python_v.1.0.0/smpl/models/` 以下)。女性モデルが無ければ
男性モデルのみで続行する。

```bash
# 1. ランダムな人物姿勢を 100 体生成 (既定の保存先 scripts/random_human_poses/ に保存)
rosrun aero_demo generate_random_human_poses.py --num-samples 100

# 2. 骨格から左右の掌の位置姿勢を推定 (既定で 1. の出力を読み、scripts/random_palm_poses/ に保存)
rosrun aero_demo estimate_palm_poses.py

# 3. viser で表示 (既定で 1./2. の出力を読む)
rosrun aero_demo draw_random_human_poses.py

# 4. 人間の左手 (既定) にロボットの右腕 (既定) で触れる全身 IK を解く
#    (既定の入力先・出力先は scripts/test_palm_pose_pipeline/palms/ と
#     .../handshakes/ -- 1.-3. で使う random_human_poses/ とは別のディレクトリ
#     なので、1.-3. の結果を渡したいときは下記のように明示的に指定する)
rosrun aero_demo solve_palm_ik.py \
    --input-dir scripts/random_palm_poses \
    --output-dir scripts/random_handshake_poses

# 5. SMPL メッシュ + ロボットモデルを viser で表示 (4. の結果と、1. の骨格を対応づける)
rosrun aero_demo view_handshake_poses.py \
    --skeleton-dir scripts/random_human_poses \
    --handshake-dir scripts/random_handshake_poses
```

保存先を変えたい場合は、各スクリプトの `--input-dir`/`--output-dir`/
`--palm-dir`/`--skeleton-dir`/`--handshake-dir` で明示的に指定できる
(例: `--output-dir /tmp/random_human_poses`)。`solve_palm_ik.py` の対象の
手は `--human-hand`/`--robot-arm` で変更できる (既定は人間の左手・ロボット
の右腕)。

`solve_palm_ik.py`/`view_handshake_poses.py` を `--input-dir`/
`--output-dir`/`--skeleton-dir`/`--handshake-dir` を省略してそのまま実行
すると、既定値である `scripts/test_palm_pose_pipeline/` 以下の
`palms/`/`skeletons/`/`handshakes/` (`scripts/test_generate_and_estimate_
palm_poses.py` が書き出す固定サンプル) を使う:

```bash
rosrun aero_demo solve_palm_ik.py          # test_palm_pose_pipeline/palms/ -> handshakes/
rosrun aero_demo view_handshake_poses.py   # test_palm_pose_pipeline/skeletons/ + handshakes/
```

`draw_random_human_poses.py`/`view_handshake_poses.py` は viser の
ブラウザビューアを起動する (WSLg 環境などでは自動でブラウザが開く。開かない
場合は標準出力の URL を手動で開く)。どちらも viser 画面の Back/Next
ボタンで人物を切り替える (`draw_random_human_poses.py` は加えて
`--advance-mode auto` にすると `--pause` 秒ごとに自動で次の人物へ進み、
`--output-dir` を指定すると表示した各姿勢の画像をその都度保存する)。
`view_handshake_poses.py` はビューアに SMPL メッシュとロボット (Aero)
モデルの 2 つだけを表示する (骨格線・掌 Axis・ランドマークなどは描かない)。
ロボットは既定で `solve_palm_ik.py` と同じ指関節なしの URDF を使うが、
`--use-hand` を付けると指関節ありの URDF で表示する。

## 座標系

各 JSON の座標はロボット座標系 (x=前, y=左, z=上) で、両足が地面
(z=0) についた状態で保存される。掌のローカル座標系は
+x = 指先方向 (手首→指先)、+y = 手の甲→掌の方向 (掌の法線)、
+z = x × y。
