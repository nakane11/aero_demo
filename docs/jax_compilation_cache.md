# jax / jaxls のコンパイル・キャッシュ仕様まとめ

`solve_palm_ik.py`(バッチ IK)と `plan_handshake_motion.py`(jaxls の軌道
最適化)は、どちらも jax の JIT コンパイルに依存しており、初回コンパイルは
数秒〜数十秒かかる。この文書は、そのコンパイル・永続キャッシュまわりの
仕様と対策をまとめる。README.md の「環境構築」からはリンクのみを残し、
詳細はこちらに集約する。

## 1. 永続コンパイルキャッシュの基本

`solve_palm_ik.py` は起動時に jax の永続コンパイルキャッシュ(既定で
`~/.cache/jax_compilation_cache`、`JAX_COMPILATION_CACHE_DIR` 環境変数で
変更できる)を有効にする。JIT コンパイルは、計算グラフの形状(
`--collision-ik-stop`/`--attempts-per-pose`/`--skeleton-dir` の有無/使う腕
(`--robot-arm`)などで決まる)ごとに初回だけ必要な重い処理で、以降は同じ
venv/jax バージョンで同じ形状の計算であればディスクキャッシュから即座に
読み込まれる。

キャッシュのキー(フィンガープリント)はコンパイル前 HLO から決まる。
ソースの行番号や空白はキャッシュキーに影響しないが、jaxls が計算グラフに
**定数として焼き込む値**(ロボットの姿勢由来の FK パラメータなど)が
1 bit でも違うと別計算とみなされ再コンパイルになる(対策は 4. 参照)。

## 1.5. 環境変数の設定順序 (`run_camera_pipeline_test.py`)

`JAX_COMPILATION_CACHE_DIR` 等は jax を import する前に設定する必要が
ある。`run_camera_pipeline_test.py` では `from aero_demo import json_io`
が `skrobot.pycompat` の `HAS_JAX` 判定経由で無条件に `import jax` して
しまうため、その import より前 (ファイル冒頭) でこれらの環境変数を
設定している。

## 2. warmup とキャッシュの単位(腕ごとに別キャッシュ)

起動時のウォームアップ(`run_camera_pipeline_test.py` の `_warmup_ik`)は、
左右それぞれの腕で IK(`solve_person_ik`)と `plan_person_motion`(jaxls の
軌道最適化)をダミー目標に対して 1 回ずつ解いておき、JAX/jaxls の初回
コンパイルを前倒しで済ませる。scikit-robot 側の `JaxlsSolver` はコンパイル
済み問題を腕ごと(`collision_link_list` が l/r で異なる)に別々のキャッシュ
として保持するため、事前ウォームアップで両腕分がキャッシュされ、以後
ARMED のたびに差し出し手の左右が入れ替わっても再コンパイルは起きない。

通常運用では pre-touch/線形補間だけで事後検証に通ることが多く、その場合は
jaxls 自体が呼ばれない(`plan_handshake_motion.py` の `--force-optimize` を
付けると常に jaxls まで実行させられる、ベンチマーク用オプション)。

## 3. コンパイル時間の内訳(実測)

軌道最適化 1 回分の内訳(`NonlinearSolver.solve` を lower/compile/execute
の 3 段階に分離して計測):

| フェーズ | コールド(ディスクキャッシュ無) | ヒット(ディスクキャッシュ有) |
|---|---|---|
| lower(トレース、計算グラフ構築) | 6.3〜7.7秒 | 6.3秒 |
| compile(XLAコンパイル) | 29.5秒(コード生成) | 3.0秒(キャッシュ読み込み) |
| execute(LM反復の実行) | 0.44秒 | 0.37秒 |
| 合計 | 37.7秒 | 9.7秒 |

重要な点:

- **実行(LM反復そのもの)は全体の 1〜4% しか占めない。** cold-hit の差
  (約28秒)の正体はほぼ全て XLA コンパイル自体であり、実行コストではない。
- **`lower`(トレース)はディスクキャッシュの対象外**で、ヒット時でも毎回
  6〜8秒かかる(ヒット時合計 9.7秒の6割以上を占める)。キャッシュが短縮
  するのは `compile` のみ。
- この特性のため `.lower().compile()` で execute のみ省略する AOT 化は
  効果薄(左腕で0.4秒程度の節約に留まる)と判断し、導入は見送った。

実ノード(roscore + rosbag再生)での実測もこれとよく一致する: hit 時
左腕 IK 2.3秒+軌道最適化12.9秒、右腕 IK 1.5秒+軌道最適化11.5秒、
miss 時の軌道最適化は36〜38秒。

## 4. FK定数の量子化(キャッシュミス対策、解決済み)

jaxls が計算グラフに焼き込む FK 由来の定数が実行ごとに微妙にブレると
別計算とみなされ再コンパイルが起きる問題があった。対策として
`jaxls_solver.py`(scikit-robot fork `base_limit` ブランチ)に
`_quantize_fk_constants` を追加し、FK 定数を小数9桁に丸めたうえで `+0.0`
して負ゼロ(`-0.0`)を正規化している(`np.round` は符号を保持するため、
丸めても `-0.0` が残ると別定数として扱われフィンガープリントが変わる)。
同じ丸め処理を `differentiable.py`(`extract_fk_parameters`)にも適用済み。

なお 2026-09-15 の検証で、これとは別に「ノード終了時の `SIGKILL` が
次回起動時の見かけ上のキャッシュミスを誘発する」現象が判明している
(7. 参照)。運用でのミスはこちらが主因で、量子化側の問題が現在も残って
いるとは言い切れない。

## 5. デバッグ手法

- **コンパイルが実際に走ったかは `~/.cache/jax_compilation_cache` の
  ファイル数の増減で判定する。** `JAX_LOG_COMPILES=1`/
  `JAX_EXPLAIN_CACHE_MISSES=1` は `jit_solve` については何も出力しない。
- 犯人(どの定数がブレているか)の特定には `XLA_FLAGS=--xla_dump_to=DIR`
  で `*.jit_solve.before_optimizations.txt`(コンパイル前 HLO)を2回分
  取得して diff する。`JAX_COMPILATION_CACHE_DIR` を実行ごとに空
  ディレクトリにすれば必ずコンパイルさせられる。
- HLO の diff よりも先に、jaxls へ渡す**入力側**(FK パラメータ)を diff
  する方が早い。`scratch/warmup_fk_dump.py`/`scratch/compare_fk_dump.py`
  で量子化後 FK 定数の最大絶対差・符号付き ULP 差を出せる。
- GPU 版 jax は起動時にデバイスメモリ確保に失敗するたびに
  `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` の警告を出しながら要求
  サイズを縮小することがある(正常動作、キャッシュとは無関係)。気になる
  場合は `XLA_PYTHON_CLIENT_PREALLOCATE=false`/
  `XLA_PYTHON_CLIENT_MEM_FRACTION` で確保量を抑えられる。

## 6. `--bag` 起動時のデッドロック(未修正)

`run_camera_pipeline_test.py --bag <file>` は、`rospy.set_param
('/use_sim_time', True)` → `rospy.init_node` → `HandshakePipelineNode(args)`
(実機接続を含む `__init__` 一式)が**完了してから** `rosbag play --clock`
をサブプロセスとして起動する実装になっている。そのため `/clock` が一切
配信されていない状態で `__init__` 内の TF 待ち等に入り、**起動が永久に
デッドロックする**(実機コントローラの無い開発環境で発生を確認)。

回避策: `rosbag play <bag> --clock` を**先に別プロセスとして起動して
`/clock` を配信させておいてから**、ノードは `--bag` を付けずに起動する
(記録済みクリップの color/depth/camera_info/tf/tf_static は既定のトピック
名と一致するため、`--bag` なしでもそのまま subscribe される)。

```bash
rosparam set /use_sim_time true
rosbag play data/session1.bag --clock &
python3 scripts/ros/run_camera_pipeline_test.py --auto-arm --no-wait-for-client
```

本体コード側の恒久修正(rosbag play の起動を `rospy.init_node` の直後・
`HandshakePipelineNode` 構築の前に前倒しすれば解消するはず)はまだ行って
いない。

## 7. 実運用でのキャッシュミス: SIGKILL が原因(解決済み)

以前は実ノードで低頻度(8回中3回程度)のキャッシュミスが観測されていた
が、原因は FK 定数のブレではなく**ベンチの終了方法**だった。

`SIGINT` → 2秒待ち → `SIGKILL` で強制終了すると、jax の永続コンパイル
キャッシュへの非同期書き込みが完了する前にプロセスが死に、次回起動時に
不完全なエントリを読んで見かけ上の miss を起こしていた。`SIGKILL` を
撤廃し、`SIGINT` 送信後にプロセスが自発的に終了するまで待つようにした
ところ、16回連続起動で miss 0回・2回目以降のキャッシュ更新も0件だった。

**運用上の結論: ノード終了時に `SIGKILL` を使わない
(`SIGINT` を送った後はプロセスが自発的に終了するまで待つ)。** これだけで
安定してキャッシュヒットする。本体コード側の対策は不要と判断した。

## 8. `solve_palm_ik.py` へのウォームアップ導入(最終状態)

`run_camera_pipeline_test.py` の `_warmup_ik` のうち、バッチIK
(`solve_person_ik` → `batch_inverse_kinematics`, `backend='jax'`)の部分を
`solve_palm_ik.py` にも `_warmup_batch_ik` として移植した(`main()` が
人物ループに入る前に実行、`--no-warmup` で無効化可能、既定で有効)。
`--robot-arm auto`(既定)では対象人物がどちらの手を差し出すか事前に
分からないため両腕分ウォームアップし(`--robot-arm` で固定している場合は
使う方だけ)、ダミー目標には `human_body_obstacles({})` を使い実際の対象者
と同じ shape で jit をトレース/コンパイルさせる。

ただし入力バッチに IK 対象(`offered_hand` が L/R)が1人もいない場合は
バッチIK自体が呼ばれずウォームアップが無駄になるため、`main()` は事前に
対象の有無を確認し、**0人ならウォームアップ自体をスキップする**
(`--no-warmup` の指定に関わらず)。

`plan_handshake_motion.py`(jaxls の軌道最適化)へは移植しなかった。この
スクリプトは1回の起動につき1回だけ実行される使い捨てプロセスで、
pre-touch/線形補間の幾何的な軌道が事後検証を通ればそこで早期returnし
jaxls 自体を呼ばないことが多いため、一律ウォームアップすると無条件に
数秒〜十数秒(lower 6〜8秒+compile 数秒)を追加するだけで、典型的な使い方
(変更のたびに素早く回す回帰テスト)にとってはむしろ悪化になる。

### 効果測定(`run_pipeline_test.py --plan-motion`、`--seed 42`、20人生成・
3人が対象)

| 段階 | 導入前 | 導入後 |
|---|---|---|
| IK 1段階目(バッチIK) | 2.673 秒/人 | **0.142 秒/人**(約19倍) |
| IK 2段階目(事後検証+後処理判定) | 0.338 秒/人 | 0.485 秒/人(誤差範囲) |
| 掌推定込み全体 | 3.015 秒/人 | 0.632 秒/人 |

IK 2段階目のばらつきはウォームアップとは無関係(`solve_palm_ik.py` に
`--seed` が渡らずバッチIKの初期値サンプリングが実行のたびに変わるため)。

### 実ノード(`run_camera_pipeline_test.py --auto-arm`、rosbag 1人分)での
最終確認

warmup(両腕とも hit): 左腕 IK 2.7秒+軌道最適化11.9秒、右腕 IK 1.7秒+
軌道最適化10.9秒。

本番(warmup後、pre-touch経由で事後検証に通りjaxls本体は未使用):
IK 合計約0.26秒(`collision_ik_time` 0.19秒+`candidate_selection_time`
0.06秒)、軌道計画 `compute_time` 0.86秒(`kind: pretouch`)。
