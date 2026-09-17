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

## 9. 台車可動域の動的制限による人物ごとの再コンパイル(解決済み、2026-09-17)

2026-09-16 の調査で、`run_pipeline_test.py` (`--seed 0`、複数人まとめて
1 プロセスで処理) で warmup 後も IK 1段階目 (バッチIK) が 6〜8秒/人と
`_warmup_batch_ik` 導入時の実測(8節、0.142秒/人) より大幅に遅い現象が
再発した。原因は warmup とは別の箇所にある。

`solve_palm_ik.py` の `restrict_base_yaw_range_to_human_facing`/
`restrict_base_y_range_to_hand_side` (`solve_person_ik` 呼び出し前、
`person_base_limits` を作る箇所) が、台車の可動域 (`base_limits`) を
**人物ごとに異なる連続値** (特に yaw は人物の向き `human_yaw` に依存) へ
制限している。`base_limits` は `batch_inverse_kinematics` 内部で仮想
台車関節 (`_attach_batch_virtual_base_chain`) の `min_angle`/`max_angle`
になり、最終的に `jnp.array(...)` として `solve_batched` (jax.jit)
のクロージャに焼き込まれる (4節の FK 定数と同じ機構)。そのため人物ごとに
HLO フィンガープリントが変わり、`_warmup_batch_ik` が固定の
`base_limits` で先に払ったコンパイルの恩恵を本番の各人物が受けられず、
事実上「人物 1 人につき 1 回」再コンパイルが起きる。

検証 (同じ 10 人・3 人が IK 対象、`~/.cache/jax_compilation_cache` の
ファイル数増分で判定):

| 条件 | 3人分のIK所要時間(warmup除く) | キャッシュファイル増分 |
|---|---|---|
| 既定 (動的制約あり) | 約26.5秒 (≒8.8秒/人) | +4 |
| `--no-facing-base-constraint --no-hand-side-base-constraint` | 約3.7秒 (≒1.2秒/人) | +1 |

約7倍の差があり、再コンパイルが支配的であることを確認した。

**対策 (案B、2026-09-16 実装・2026-09-17 検証):** scikit-robot フォークの
`differentiable.py`(`create_batch_ik_solver`)で `joint_limits_lower`/
`joint_limits_upper` を `_create_solver_fn` のクロージャ定数からやめ、
既存の `obstacle_values` と同じパターンで `solve_single`/`solve_batch` の
通常引数 (trace 対象) として渡すよう変更した。`robot_model.py` 側も
`_batch_ik_collision_solver_cache` のキャッシュキーから
`base_joint_limits_sig` (min_angle/max_angle のタプル) を削除し、
`wants_collision_avoidance` 時は毎回 `solver_kwargs` にその呼び出しの
実際の範囲を明示的に渡すようにした。値がクロージャに焼き込まれなくなる
ため、人物ごとに `base_limits` が変わっても再コンパイルが起きなくなる
(scikit-robot 側の変更、`aero_demo` 側は無修正)。

検証 (`python3 scripts/run_pipeline_test.py 100 --seed 0 --plan-motion`、
IK 対象 18 人中 18 人 solved、2 回連続実行で `~/.cache/jax_compilation_
cache` のファイル数増分 0 = 追加コンパイルなしを確認):

| 段階 | 対策前 (8節時点の想定、8.8秒/人) | 対策後 |
|---|---|---|
| IK 1段階目 (干渉回避バッチIK) | 約 8.8 秒/人 | **0.142〜0.143 秒/人**(約62倍、8節のwarmup単体導入時とほぼ同水準まで復帰) |
| IK 2段階目 (事後の干渉検証+後処理判定) | 0.338 秒/人 (参考) | 0.328〜0.336 秒/人 |
| 軌道計画 (verified 18/18) | - | 0.9 秒/人 |

18人の IK 対象・solved 人数・verified 人数は対策前後で変わらず、精度面の
劣化は確認されなかった。

**`run_camera_pipeline_test.py` (実ノード) への当てはまりについて:**
コードを確認したところ、`run_camera_pipeline_test.py` の
`_solve_handshake` は `restrict_base_yaw_range_to_human_facing`/
`restrict_base_y_range_to_hand_side` を一切呼んでおらず、`base_limits` は
`__init__` で作った起動時の固定値 (`--base-x/y/yaw-range` 由来) を全人物・
全 ARM イベントで使い回している。つまりこの人物ごとの動的
`base_limits` によるキャッシュミス問題は、そもそも `solve_palm_ik.py`
(オフラインの複数人まとめ処理パイプライン) 固有の問題であり、
`run_camera_pipeline_test.py` は対策前から (静的な `base_limits` しか
使わないため) 影響を受けていなかったことが判明した。8節末尾に書いていた
「実運用でも同様に当てはまる可能性がある (未検証)」という懸念は誤りで、
実際には該当しない。

rosbag (`data/session1.bag`、実カメラ・実オペレータ1人分) を使った実ノード
での確認 (`--auto-arm --no-wait-for-client`、rosbag を `/use_sim_time`
込みで先行再生): warmup 左腕 IK 2.1秒+軌道最適化12.1秒、右腕 IK 1.6秒+
軌道最適化11.0秒 (いずれも 3節の「ヒット」実測値と同水準)。本番 (warmup
後、実際に手を差し出した1人): IK 0.33秒 (`collision_ik_time` 0.17秒+
`candidate_selection_time` 0.16秒)、軌道計画は線形補間のみで事後検証に
通り (`compute_time` 1.5秒)、想定通り安定していた。

**追記 (2026-09-17): `run_camera_pipeline_test.py` にも動的制限を追加。**
上記の通りキャッシュミス対策としては `run_camera_pipeline_test.py` に
手を入れる必要はなかったが、`solve_palm_ik.py` の
`restrict_base_yaw_range_to_human_facing`/
`restrict_base_y_range_to_hand_side` (差し出し手の側・人物の正面向きに
合わせて台車可動域を絞る) を `run_camera_pipeline_test.py` は呼んでおらず
両スクリプトで挙動が食い違っていたため、案Bの対策により人物ごとに
`base_limits` を変えても再コンパイルが起きないことが確認できたのを機に
`_solve_handshake` (旧963〜1010行付近) にも同じ制限を追加し、
`solve_palm_ik.py` の `main()` と挙動を揃えた。

rosbag (`data/session1.bag`) でこの変更後を再検証
(`--auto-arm --no-wait-for-client`、手順は上記と同一): warmup 左腕 IK
2.1秒+軌道最適化12.1秒、右腕 IK 1.6秒+軌道最適化10.7秒。本番 (実際に
手を差し出した1人、`offered_hand: "R"`、`robot_arm: "l"`): IK 合計0.20秒
(`collision_ik_time` 0.146秒+`candidate_selection_time` 0.053秒)、軌道
計画は線形補間のみで事後検証に通り (`kind: pretouch`、`compute_time`
0.87秒、`verified: true`)、`solved: true`。ノードログに
`recompil`/`Traceback` 等の異常は見られず、変更前と同水準の所要時間で
安定して動作することを確認した。1人分の検証のため統計的な比較はできない
が、案Bの効果によりこの変更でキャッシュミスが再発しないことは実ノードでも
裏付けられた。

### 合成骨格での「長寿命プロセス・人物ごとの逐次呼び出し」再現ベンチマーク

`run_camera_pipeline_test.py` と全く同じ呼び出しパターン (`_warmup_ik` を
1回 → `estimate_palm_poses.PalmPoseEstimator`/`solve_palm_ik.
solve_person_ik`/`plan_handshake_motion.plan_person_motion` を人物ごとに
逐次呼ぶ) を、実カメラ/ROS 抜きで検証する一時ベンチマークスクリプトを
作成し、`generate_random_human_poses.py` と同じ `RandomSkeletonGenerator`
で合成した骨格 (--seed 0 で60人、--seed 1で100人、offered_hand が決まった
人だけ IK・軌道計画まで進める) に対して実行した。

| | --seed 0 (60人生成) | --seed 1 (100人生成) |
|---|---|---|
| offered_hand が決まった人数 | 7 / 60 | 8 / 100 |
| IK solved | 7 / 7 | 8 / 8 |
| 軌道 verified | 7 / 7 | 7 / 8 (1件は線形補間が事後検証で貫通、`--force-optimize` 相当の追加最適化なしのため) |
| warmup 合計 | 25.0 秒 (左13.4秒+右11.6秒) | 25.3 秒 (左13.7秒+右11.6秒) |
| IK 1段階目 (collision_ik_time) | 0.145 秒/人 | 0.146 秒/人 |
| IK 2段階目 (candidate_selection_time) | 0.268 秒/人 | 0.454 秒/人 |
| 軌道計画 | 0.770 秒/人 | 1.121 秒/人 |

2回の実行を合わせた計 15 人 (offered_hand が決まった人) では IK solved
15/15、軌道 verified 14/15 で、いずれも run_pipeline_test.py のオフライン
計測 (0.142〜0.143秒/人) と一致する水準の IK 時間になった。これは、
`run_camera_pipeline_test.py` が (前述の通り) 人物ごとに `base_limits` を
変えない静的な設計のため、案Bの修正の有無に関わらずそもそも1人ずつの
逐次呼び出しパターンでも再コンパイルが起きないことを裏付けている。

## 10. `human_body_obstacles` の重複再構築(解決済み、2026-09-17)

jax のコンパイル・キャッシュとは別の話だが、上記の cProfile 調査
(8〜9節) の過程で見つかった軌道計画・IK 2段階目の無駄な再計算を
まとめてここに記録する。

`plan_handshake_motion.verify_waypoints`(1軌道あたり waypoint 数だけ
ループ、既定 20)と `solve_palm_ik.pick_verified_candidate`(IK 2段階目、
収束した候補の数だけループ、数百件になりうる)は、いずれも同じ
`joint_positions`(その人物の骨格、ループ中ずっと不変)に対して
`solve_palm_ik.collision_pairs_min_distance` を繰り返し呼ぶ。この関数は
呼ばれるたびに内部で `human_body_obstacles(joint_positions)` を呼んで
人体側の干渉回避用 `Cylinder`(骨・掌・指、既定で44個)を毎回作り直して
おり、姿勢が変わらない間は完全に無駄な再計算だった。

**対策:** `collision_pairs_min_distance` に `obstacle_links` 引数
(既定 `None`、未指定なら従来通り内部で構築)を追加し、`verify_waypoints`
はループの外側で、`pick_verified_candidate` は候補ループに入る前に、
それぞれ `human_body_obstacles(joint_positions)` を 1 回だけ呼んで
使い回すようにした(`scripts/plan_handshake_motion.py`/`scripts/
solve_palm_ik.py`)。呼び出し側を変えない既存呼び出し (`grid_search_
collision_ik.py` など) は挙動を変えない。

**効果測定で踏んだ落とし穴:** 最初にロボット本体をフル装備 (`Aero()`、
手指込み 84 リンク) でスタンドアロン計測したところ、検証ペア数が
5587 組と非常に多く、`collision_pairs_min_distance` 自体が重い
(0.353秒/回) ため `human_body_obstacles` の再構築コスト (0.009秒/回)
は全体の 2.5% にしかならず、「ほぼ効果なし」という誤った結論になった。

しかし `plan_handshake_motion.py`/`solve_palm_ik.py` が実際に使う
ロボットは `Aero(use_hand=False)`(手指抜き 32 リンク)で、検証ペア数は
大幅に少ない。ペア数が減ると `collision_pairs_min_distance` 自体は
軽くなる一方 `human_body_obstacles` の再構築コスト(ロボットのリンク数に
依存しない、人体側だけの処理)は変わらないため、後者の相対的な比重が
逆に大きくなる。`run_pipeline_test.py 10 --plan-motion --seed 1` で
実際に生成された骨格・掌姿勢を使い、`plan_handshake_motion.py`/
`solve_palm_ik.py` をそのまま cProfile で計測し直した結果:

| | 対策前 | 対策後 | 短縮率 |
|---|---|---|---|
| `verify_waypoints` (3人, 20 waypoint/人) | 3.302秒 (`human_body_obstacles` 1.116秒) | 2.235秒 (同 0.055秒) | 約32% |
| `pick_verified_candidate` (3人分, IK 2段階目) | 1.782秒 (`collision_pairs_min_distance` 1.358秒中 `human_body_obstacles` 0.619秒) | 1.284秒 (同 0.802秒) | 約28%(`collision_pairs_min_distance` 単体では約41%) |

対策前後で `verified`/`min_dist`/`solved`/採用した姿勢はすべて完全一致
しており、この変更による結果面への影響は無い。スタンドアロン計測で
過小評価してしまった原因は、`human_body_obstacles` の再構築コストは
ロボットのリンク数に依存せず一定であるのに対し、検証対象のロボットの
リンク数(=ペア数)を実際の運用より大幅に多くしてしまい、相対的な
比重を見誤ったため(典型的なアムダールの法則的な誤り)。ベンチマークは
必ず実際に使われるロボット構成 (`use_hand=False` かどうか) と検証ペア数
を揃えて行う必要がある。

## 11. `--force-optimize` あり/なしでの軌道計画時間の実測(2026-09-17)

`python3 scripts/run_pipeline_test.py 200 --seed 1 --plan-motion` を
`--force-optimize` あり/なしでそれぞれ実行し、軌道計画
(`plan_handshake_motion.py`) 1人あたりの計算時間 (`compute_time`) を
比較した。200人生成中 IK 対象 24人、solved 23人、軌道計画対象 23人・
verified 22人という結果はあり/なしで完全に一致した(採用した軌道の
作り方だけが変わる: なし=`linear=5, pretouch=18`、あり=
`linear=1, optimized=20, pretouch=2`)。

| | `--force-optimize` なし | `--force-optimize` あり |
|---|---|---|
| 平均 (23人、`run_pipeline_test.py` 表示値) | 0.609 秒/人 | 1.410 秒/人 |
| **最大** (motion JSON の `compute_time` から集計) | **3.052 秒/人**(`human_147`, kind=`linear`) | **2.865 秒/人**(`human_147`, kind=`linear`) |
| 最大を除いた上位 | 1.137秒(`human_111`, linear, verified)、0.861秒(`human_140`)、0.859秒(`human_028`) | 1.667秒(`human_028`, optimized)、1.527秒(`human_140`, optimized)、1.511秒(`human_112`, optimized) |

**最大値がほぼ同じ理由(`--force-optimize` の有無に関わらない外れ値):**
最大となった `human_147` は pre-touch も線形補間も厳密検証(事後の干渉
チェック)を通らない(`waypoint_min_distances` に `-0.0277` の貫通が残る)
ケースで、520〜588行目のロジックにより **`--force-optimize` を指定しな
くても** `candidate['verified']` が偽のまま jaxls の最適化ループ
(`--motion-attempts` 回の warm start リトライ)まで進む。この人物は
最適化を試みても `best` (貫通が最も浅い候補、たまたま線形補間) を上回れ
ず、採用される `kind` は `linear` のまま `optimized: false` で記録される
(採用軌道の見た目は変わらないが、実際には jaxls の最適化を一通り試行
している)。そのため「1人あたりの最大軌道計画時間」は `--force-optimize`
の値に関わらずこの検証失敗ケースの jaxls リトライコストで頭打ちになり、
あり/なしで意味のある差にならない。

**平均には意味のある差が出る:** それ以外の 22人は pre-touch/線形補間が
厳密検証を通るため、`--force-optimize` なしなら 0.3〜0.9秒程度で早期
returnするのに対し、`--force-optimize` ありだと全員が jaxls の最適化を
1回強制される(採用軌道の `kind` は `optimized` に変わるが、最適化前の
候補の方が貫通が浅ければ最終的な採用軌道自体は変わらない)ため、平均が
約2.3倍(0.609秒→1.410秒)に増える。`--force-optimize` は「全員分の
軌道最適化そのものの計算時間を計測する」ためのベンチマーク用オプション
であり、通常運用(既定、`--force-optimize` なし)の 1人あたり計算時間の
実態は平均 0.609秒/人・最大 3.052秒/人(ただし最大は上記の検証失敗ケース
由来で `--force-optimize` の効果とは無関係)である。
