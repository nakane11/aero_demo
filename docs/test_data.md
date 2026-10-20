# テストデータの置き場所と一覧

テストデータ (実機の rosbag・抽出フレーム・実機ログ・合成データ) はすべて
リポジトリ直下の `data/` に置く。`data/` は `.gitignore` 済みで git には
入らない (bag は 1 本 10〜20 GB あるため)。このファイルが唯一の一覧なので、
データを足したり消したりしたらここも直す。

```
data/
  bags/                 録画した全体 bag の一時置き場 (record_camera_pipeline.launch の既定)。
                        頭カメラ較正用の head_calib_*.bag もここ (dev_tools.md)
  real/                 実機・実カメラのデータ (日付ごと)
    <YYYY-MM-DD>_<名前>/  連続録画のセッション (bag + 抽出フレーム)
    <YYYY-MM-DD>/         実機デモの日 (logs/、試行の切り出し bag、画像)
  synthetic/            合成データ (generate_random_human_poses.py 以降の既定の入出力先)
  palm_offer_clips/     record_palm_offer_clips.py の既定の保存先 (現在は空)
  base_slip/            台車スリップ計測の結果 (CSV)
  palm_detector_compare/ compare_hand_detectors.py の結果
```

## run_camera_pipeline_test.py のテストにどれを使うか

| 確かめたいこと | 使うデータ | やり方 |
|---|---|---|
| 変更後も骨格検出から横並び移動の計画まで通るか (動作確認) | `real/2026-10-15/trial1_side_by_side/` | `bash replay.sh`。再生で 3/3 成功している安定した試行なので、まずこれを回す |
| 左右どちらの腕でも通るか | 上に加えて `real/2026-10-13/trial22_side_by_side/` | trial1 は R手/l腕、trial22 は L手/r腕。trial22 は 0.5 倍速 (既定) で回す |
| 境界の試行で改善・悪化したか | `real/2026-10-15/trial2_side_by_side/` | 変更前でも 2/5 しか成功しない。何回か回して成功率で比べ、1 回の失敗を回帰とみなさない |
| IK・軌道計画・横並びの成功率を実機の人で前後比較 | `real/2026-10-13/logs/` の 22 試行 (+ `real/2026-10-15/logs/` の 2 試行) | `tools/ros/resolve_saved_trials.py` で保存した skeletons/palms の JSON を解き直す (下記) |
| 差し出し判定 (どちらの手か・見逃し・誤トリガー) | `real/2026-09-30_session1/` (ラベル 167 件)、`real/2026-09-30_session3/` (差し出し無し) | `tools/tune_offer_selector.py` などで抽出フレームを評価する ([dev_tools.md](dev_tools.md)) |

- 切り出し bag の再生では、骨格検出が毎回少しずつ違うフレームで差し出しを判定するので、掌の向きが実機と数度〜数十度ずれる。結果は実機との一致ではなく、ARMED になった手と IK・軌道計画の成否で見る。
- 再生は `--no-robot-interface` で回すので、scan による台車位置の補正、押し込み直前の解き直し、実機での実行は試されない。これらを確かめるには実機が要る。
- 保存した JSON からの解き直しは、検出の揺れが入らず毎回同じ結果になるので、変更前後の比較に向いている。`tools/ros/resolve_saved_trials.py` はノードと同じ `HandshakePlanner` で解き、試行ごとの到達段階 (ik_fail / no_press / motion_ng / motion_ok / transition_ok) と、保存時の段階から変わった試行を出す。ノードのオプション (`--torso-surface-offset` など) はそのまま渡せる。roscore は不要。

  ```bash
  python3 tools/ros/resolve_saved_trials.py \
      data/real/2026-10-13/logs data/real/2026-10-15/logs --out /tmp/resolve.json
  ```
- 連続録画のセッション bag (`real/2026-09-*`) には /odom・/scan・/joint_states が入っていない。差し出し判定や起動・warmup の確認には使えるが、試行の通し確認には使わない。
- 合成データでの回帰テストは `scripts/run_pipeline_test.py` で行う (`run_camera_pipeline_test.py` は使わない)。`--plan-motion` のときは `plan_handshake_motion.py` を `solve_palm_ik.py` と並行して先に起動し、入力に依存しない warmup (約40秒) を重ねる (400 人・`--side-by-side-transition` で 132 秒 → 95 秒、結果は同一。`--no-parallel-warmup` で従来どおり順に実行)。並行実行のため子プロセスには `XLA_PYTHON_CLIENT_PREALLOCATE=false` を渡している (GPU メモリを 75% ずつ確保すると後から起動した方が失敗するため)。

## 置き方の決まり

- 実機の全体 bag は `data/bags/` に録る。テストに使う区間は
  `tools/ros/clip_bag.py` で切り出して `data/real/<日付>/<名前>/` に置き、
  元の全体 bag は消してよい。
- 切り出した試行には、同じディレクトリに `replay.sh` (再生手順と期待結果)、
  実機の `person_NN.log`、その試行の handshakes/motions/palms/skeletons の
  JSON を一緒に置く。
- `run_camera_pipeline_test.py` のログは起動ごとに
  `/tmp/run_camera_pipeline_test_logs/<起動時刻>/` に残る (消えないが、/tmp なので
  再起動で失われる)。試行ごとの skeletons/palms/handshakes/motions の JSON も、
  `--save-dir` を指定しなければ同じディレクトリに保存される。押し込み後の画像
  (`/tmp/aero_demo_grasp_captures`) と手の再検出に失敗したときの画像
  (`/tmp/aero_demo_refine_failures`) は `--save-dir` を指定しないと別に /tmp に出る。
  実機デモの後に `data/real/<日付>/` の `logs/`・`grasp_captures/`・
  `refine_failures/` へ移す。

## 実機・実カメラ (`data/real/`)

### 連続録画のセッション (差し出し判定の評価用)

骨格・掌・骨格重畳画像は `tools/ros/extract_skeletons_from_bag.py` で抽出
したもの。`palms/*.json` の `human_label` は `tools/label_offer_images.py`
で人が付けた正解ラベル。どの bag にも /camera の画像・camera_info と /tf
(/tf_static) だけが入っている (/odom・/scan・/joint_states は無い)。

| ディレクトリ | bag | フレーム (ラベル) | 用途・旧名 |
|---|---|---|---|
| `2026-09-17_session1/` | session1.bag (53 秒・13 GB) | 19 (R 7 / L 4 / null 8) | jax キャッシュの実ノード検証 ([jax_compilation_cache.md](jax_compilation_cache.md))。旧 `data/` |
| `2026-09-30_session1/` | session1.bag (95 秒・24 GB) | 167 (R 62 / L 62 / null 43) | 差し出し見逃しの解析、人体の胴体の大きさの検証 ([ik_and_motion_constraints.md](ik_and_motion_constraints.md))。旧 `data2/` |
| `2026-09-30_session3/` | session3.bag (49 秒・12 GB) | 79 (全部 null) | 差し出していない場面 (誤トリガーの評価)。旧 `data3/` とリポジトリ直下の `session3.bag` |

### 実機デモの日

| ディレクトリ | 中身 |
|---|---|
| `2026-09-11/` | `toggle_grasp_*.jsonl` (グリッパの把持状態・温度・電流のログ) |
| `2026-10-02/`, `2026-10-05/` | `grasp_captures/` (押し込み後の画像)、`refine_failures/` (手の再検出に失敗したときの画像) |
| `2026-10-13/` | `logs/` (試行 1〜22 の実機ログと JSON。実機 22 人の IK・横並びの再現ベンチに使う)、`grasp_captures/`、`refine_failures/`、`trial22_side_by_side/` |
| `2026-10-15/` | `logs/` (試行 1・2)、`grasp_captures/`、`refine_failures/`、`trial1_side_by_side/`、`trial2_side_by_side/` |
| `2026-10-16/` | `trial3_side_by_side/`、`trial6_side_by_side/` (試行 3・6 の person_NN.log・JSON と、その試行の `grasp_captures/`・`refine_failures/` を中に置いた) |

### 試行の切り出し bag (再生テスト用)

`run_camera_pipeline_test.py` に入力される /camera の画像・camera_info、
/tf・/tf_static に加えて、実機の /odom・/scan・/joint_states も入っている。
それぞれの `replay.sh` を実行すると、専用の roscore で bag を再生して、
差し出し判定から横並び移動の計画までを実機なし (`--no-robot-interface`) で
確認できる。結果は `/tmp/run_camera_pipeline_test_logs/latest/person_01.log` に出る。
各ディレクトリの `replay_results/` には、過去に再生したときの JSON を置いてある。

`replay.sh` は最初の `[result]` (成功・失敗とも) が出た 3 秒後に残りの再生を打ち切る。
`--no-robot-interface` では ARMED 後の区間は使われないので、bag を最後まで再生して
待つだけだった約 1 分が不要になる (trial1 で warmup 済みのとき約 150 秒 → 80 秒)。
bag の最後まで再生したいときは `EARLY_STOP=0 bash replay.sh`。warmup は jax の
キャッシュが温まっていれば約 45 秒だが、solve_palm_ik.py などを変更した直後の
初回はキャッシュミスで約 100 秒かかる。何回も回して成功率を見るときは、初回を
捨ててから測る。

| ディレクトリ | 区間 (元 bag) | 実機の試行 | 再生の結果 |
|---|---|---|---|
| `2026-10-13/trial22_side_by_side/` | 22:08:46.0〜22:09:20.0 (34 秒) | L手/r腕、横並びまで成功 | 0.5 倍速 2/2、等速 4/5 成功 (等速は掌の向きが 6〜20° ずれる) |
| `2026-10-15/trial1_side_by_side/` | 17:12:08.5〜17:12:46.0 (37.5 秒) | R手/l腕、横並びまで成功 | 3/3 成功 |
| `2026-10-15/trial2_side_by_side/` | 17:13:50.2〜17:14:25.5 (35.3 秒) | R手/l腕、横並びまで成功 (scan 補正 16 cm、押し込みの解き直しは too_large) | 2/5 成功。掌の向き次第で解けるかが変わる境界のケース |
| `2026-10-16/trial3_side_by_side/` | 20:07:01.1〜20:07:35.8 (34.7 秒、元は 20-06-35.bag) | R手/l腕、横並びまで成功 (hover で手を再検出できず no_hand 2 回、scan 補正は 1 回目だけ採用) | 未確認 |
| `2026-10-16/trial6_side_by_side/` | 20:11:14.7〜20:11:52.3 (37.6 秒) | R手/l腕、横並びまで成功 (hover で手を再検出できず no_hand 2 回、scan 補正 2 回) | 未確認 |

再生でつまずく点 (詳しくは各 `replay.sh` の冒頭):

- WSL のループバックでは roscore がハングするので、eth0 のアドレスで専用の
  マスター (ポート 11411) を立てる。
- `run_camera_pipeline_test.py --bag` は、/clock が来ないまま初期化の TF 待ちで
  止まる。初期化中は TF だけを先頭 1 秒ぶんループ再生し、warmup が終わってから
  全体を再生する。
- 骨格検出は起動直後の約 1.5 秒は人を出さないので、区間は ARMED の約 3 秒前
  から始める。

切り出しの例:

```bash
python3 tools/ros/clip_bag.py \
    data/bags/camera_pipeline_2026-10-15-17-11-48.bag \
    data/real/2026-10-15/trial1_side_by_side/camera_pipeline_2026-10-15_trial1_side_by_side.bag \
    --start '2026-10-15 17:12:08.5' --end '2026-10-15 17:12:46.0'
```

区間は `person_NN.log` の `[ARMED]` の約 3 秒前から、`実行を終了しました`
の 2〜3 秒後 (台車が止まるまで) にする。bag の時刻は録画した PC の時刻で、
ログの時刻と同じ時計なので、そのまま使える。

## 合成データ (`data/synthetic/`)

| ディレクトリ | 中身 |
|---|---|
| `random_human_poses/` ほか 3 つ | 40 人ぶん。2026-09-08 の生成で、IK 結果は hand_y 固定より前のもので古い。最新のコードで評価するときは `scripts/run_pipeline_test.py` で作り直す (seed を指定すれば再現できる) |
| `label_backup_20260902/` | 2026-09-02 時点の掌 JSON (100 人) のバックアップ |
| `seed0_ik_targets/` | **IK・軌道計画・横並び移動のパラメータを変えたときの前後比較用 (日常はこれを使う)**。下記 |

### `seed0_ik_targets/` (固定の比較セット)

`run_pipeline_test.py 400 --seed 0` が作る 400 人のうち、掌を差し出していて IK の
対象になる 67 人 (R 36 / L 31) の `skeletons/` と `palms/` を保存したもの (2026-10-20 作成)。
残りの 333 人は掌を差し出していないので IK 以降を素通りし、成功数にも時間にも効かない。
そのため 400 人で回した結果と同じ数字が約 1/6 の人数で得られる (骨格生成の約 15 秒も省ける)。

```bash
cd scripts
python3 run_pipeline_test.py --input-set ../data/synthetic/seed0_ik_targets \
    --plan-motion --side-by-side-transition [--solve-args "..."] [--motion-args "..."]
```

- `--input-set` を付けると骨格生成・掌推定を省き、このセットを読み取り専用で使う (`num_people` は不要)。
  結果は従来どおり `/tmp/aero_demo_pipeline_*` に出る。
- **ファイル名 (`human_NNN.json`) を変えたり、番号を詰め直したりしない。** `solve_palm_ik.py` は
  人ごとの乱数シードをファイル名の crc32 で決めるので、名前が変わると IK の結果が変わる。
- 400 人 (seed 0) と同じ結果になることを確認済み: solved 54/67、軌道 verified 53/67。
  所要時間は 400 人の 115 秒 → 約 107 秒 (IK 約 38 秒、横並び移動約 39 秒が大半。warmup はキャッシュが温まっていれば 7 秒)。
  IK だけの変更なら `--plan-motion` を外すと solve のみ (約 50 秒) で済む。
- **横並び移動の成否は同じ入力でも回ごとに揺れる。** 同じセットを 3 回回して 49・52/53 (400 人では 51/53)。
  揺れるのは境界の 3 人 (human_150・175・358) で、IK・軌道計画は 3 回とも一致した。横並び移動の
  1〜2 人の差は変更の効果とみなさず、複数回回して比べるか、揺れる人を除いて見ること (原因は未調査)。
- seed 0 は調整に何度も使っているため、採用を決める前に別の seed (`run_pipeline_test.py 400 --seed 1` など)
  でも同じ向きに動くことを確認する。
- 作り直すとき: `run_pipeline_test.py 400 --seed 0 --plan-motion` を回し、`handshakes/` の JSON で
  `target` が true の人だけ、`skeletons/` と `palms/` の同名ファイルを新しい `skeletons/`・`palms/` へコピーする
  (名前はそのまま)。`data/` は git に入らないので、消えたらこの手順で復元する。

`scripts/run_pipeline_test.py` は毎回 `/tmp/aero_demo_pipeline_*` に生成するので
(消されずに溜まる)、ここのデータは使わない。`--seed 0` の 400 人や 42 人のベンチマークは、
保存したデータではなくこのスクリプトで毎回作り直している。
