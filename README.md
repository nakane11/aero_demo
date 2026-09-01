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

3. **`scripts/draw_random_human_poses.py`**
   手順 1・2 で生成した JSON を読み込み、SMPL の人体メッシュ・骨格・
   (あれば) 左右の掌の座標系を viser ビューアで重ねて表示する。

```
generate_random_human_poses.py  --output-dir random_human_poses/
        │  (skeleton.joint_positions + smpl.pose/betas/root_pos/gender)
        ▼
estimate_palm_poses.py  --input-dir random_human_poses/ --output-dir random_palm_poses/
        │  (左右の掌の position/rot、または None)
        ▼
draw_random_human_poses.py  --input-dir random_human_poses/ --palm-dir random_palm_poses/
        │  (viser ブラウザビューアで表示)
```

## 使い方

SMPL のモデルファイル (`.pkl`) はライセンス上リポジトリに同梱されていない
ため、各スクリプトの `--model-path` / `--female-model-path` で指定する
(既定値は `~/SMPL_python_v.1.0.0/smpl/models/` 以下)。女性モデルが無ければ
男性モデルのみで続行する。

```bash
# 1. ランダムな人物姿勢を 100 体生成
rosrun aero_demo generate_random_human_poses.py \
    --num-samples 100 --output-dir /tmp/random_human_poses

# 2. 骨格から左右の掌の位置姿勢を推定
rosrun aero_demo estimate_palm_poses.py \
    --input-dir /tmp/random_human_poses \
    --output-dir /tmp/random_palm_poses

# 3. viser で表示
rosrun aero_demo draw_random_human_poses.py \
    --input-dir /tmp/random_human_poses \
    --palm-dir /tmp/random_palm_poses
```

`draw_random_human_poses.py` は viser のブラウザビューアを起動する
(WSLg 環境などでは自動でブラウザが開く。開かない場合は標準出力の URL を
手動で開く)。既定 (`--advance-mode manual`) では viser 画面の Back/Next
ボタンで人物を切り替える。`--advance-mode auto` にすると `--pause` 秒
ごとに自動で次の人物へ進む。`--output-dir` を指定すると、表示した各姿勢の
画像をその都度保存する。

## 座標系

各 JSON の座標はロボット座標系 (x=前, y=左, z=上) で、両足が地面
(z=0) についた状態で保存される。掌のローカル座標系は
+x = 指先方向 (手首→指先)、+y = 手の甲→掌の方向 (掌の法線)、
+z = x × y。
