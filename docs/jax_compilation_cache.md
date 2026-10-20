# jax / jaxls のコンパイル・キャッシュ仕様まとめ

`solve_palm_ik.py`(バッチ IK)と `plan_handshake_motion.py`(jaxls の軌道
最適化)は、どちらも jax の JIT コンパイルに依存しており、初回コンパイルは
数秒〜数十秒かかる。この文書は、そのコンパイル・永続キャッシュまわりの
仕様と対策をまとめる。README.md の「環境構築」からはリンクのみを残し、
詳細はこちらに集約する。

## 1. 永続コンパイルキャッシュの基本

`solve_palm_ik.py`・`plan_handshake_motion.py`・`run_camera_pipeline_test.py`
は、いずれも冒頭(jax の import より前)で次の環境変数を `setdefault` し、jax
の永続コンパイルキャッシュを有効にする。

| 環境変数 | 既定値 |
|---|---|
| `JAX_COMPILATION_CACHE_DIR` | `~/.cache/jax_compilation_cache` |
| `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` | `0` (短いコンパイルも保存) |
| `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES` | `0` |

事前に環境変数を設定しておけば、そちらが優先される。JIT コンパイルは、計算
グラフの形状(`--collision-ik-stop`/`--attempts-per-pose`/`--skeleton-dir`
の有無/使う腕などで決まる)ごとに初回だけ必要な重い処理で、以降は同じ
venv/jax バージョンで同じ形状の計算であればディスクキャッシュから読み込まれる。

キャッシュのキー(フィンガープリント)はコンパイル前 HLO から決まる。
ソースの行番号や空白はキャッシュキーに影響しないが、jaxls が計算グラフに
**定数として焼き込む値**(ロボットの姿勢由来の FK パラメータなど)が
1 bit でも違うと別計算とみなされ再コンパイルになる(対策は 4. 参照)。

## 1.5. 環境変数の設定順序

`JAX_COMPILATION_CACHE_DIR` 等は jax を import する前に設定する必要が
ある。skrobot を import すると jax も import されるため、各スクリプトとも
ファイル冒頭(skrobot・`aero_demo` の import より前)でこれらを設定している
(`run_camera_pipeline_test.py` では `XLA_PYTHON_CLIENT_PREALLOCATE=false`
も同じ場所で設定する)。

## 2. warmup とキャッシュの単位(腕ごとに別キャッシュ)

`run_camera_pipeline_test.py` は起動時に `HandshakePlanner.warmup()`
(`_warmup_body`)を実行する。左右それぞれの腕で IK(`solve_person_ik`)と
`plan_person_motion`(`force_optimize=True` で jaxls の軌道最適化まで必ず
実行)をダミー目標に対して 1 回ずつ解き、`--side-by-side-transition`(この
スクリプトでは常に有効)のときは
横並び移動のバッチ IK(`side_by_side_transition.warmup_batch_ik`)も解く。
乱数は warmup 中だけ固定シードにし、終了後に元へ戻す。

scikit-robot 側の `JaxlsSolver` はコンパイル済み問題を腕ごと
(`collision_link_list` が l/r で異なる)に別々のキャッシュとして保持する。
両腕を事前に warmup するので、以後 ARMED のたびに差し出し手の左右が
入れ替わっても再コンパイルは起きない。warmup は本番の `plan` と引数・形状を
完全に一致させる必要があり、1 つでも違うと別関数としてトレースし直される。
そのため次の 2 点に注意している。

- **軌道問題に焼き込まれる、軌道で動かさない関節(差し出さない腕・頭など)の
  角度は固定値にする**(`_pin_warmup_fixed_joints`)。IK の出力で少し変わる
  だけで再コンパイル(左右で約 60 秒)になるため。
- **jax の x64 は warmup の前に有効にする**(`plan_handshake_motion.py` の
  `main()` は `jax.config.update('jax_enable_x64', True)` の後に
  `_warmup_solver` を呼ぶ)。skrobot の `JaxBackend` は最初に作られたときに
  x64 を有効にするので、warmup が先だと float32 でコンパイルされ、本番の
  float64 の solve で各腕の初回に約 9 秒の再コンパイルが起きる。

通常運用では pre-touch/線形補間だけで事後検証に通ることが多く、その場合は
jaxls 自体が呼ばれない(`plan_handshake_motion.py` の `--force-optimize` を
付けると常に jaxls まで実行させられる、ベンチマーク用オプション)。

## 3. コンパイル時間の内訳(軌道最適化 1 回分、過去の実測)

`NonlinearSolver.solve` を lower/compile/execute の 3 段階に分離して計測した
(当時の問題規模での値。現行 warmup の秒数は 13. を参照)。

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
  数秒かかる。キャッシュが短縮するのは `compile` のみ。
- この特性のため `.lower().compile()` で execute のみ省略する AOT 化は
  効果薄(左腕で 0.4 秒程度の節約)と判断し、導入していない。

## 4. FK定数の量子化(キャッシュミス対策、解決済み)

jaxls が計算グラフに焼き込む FK 由来の定数が実行ごとに微妙にブレると
別計算とみなされ再コンパイルが起きる問題があった。対策として
`jaxls_solver.py`(scikit-robot fork `base_limit` ブランチ)に
`_quantize_fk_constants` を追加し、FK 定数を小数9桁に丸めたうえで `+0.0`
して負ゼロ(`-0.0`)を正規化している(`np.round` は符号を保持するため、
丸めても `-0.0` が残ると別定数として扱われフィンガープリントが変わる)。
同じ丸め処理を `kinematics/differentiable.py`(`extract_fk_parameters`)にも
適用している。

なお、これとは別に「ノード終了時の `SIGKILL` が次回起動時の見かけ上の
キャッシュミスを誘発する」現象がある(7. 参照)。運用でのミスはこちらが
主因だった。

## 5. デバッグ手法

- **コンパイルが実際に走ったかは `~/.cache/jax_compilation_cache` の
  ファイル数の増減で判定する。** `JAX_LOG_COMPILES=1`/
  `JAX_EXPLAIN_CACHE_MISSES=1` は `jit_solve` については何も出力しない。
- 犯人(どの定数がブレているか)の特定には `XLA_FLAGS=--xla_dump_to=DIR`
  で `*.jit_solve.before_optimizations.txt`(コンパイル前 HLO)を2回分
  取得して diff する。`JAX_COMPILATION_CACHE_DIR` を実行ごとに空
  ディレクトリにすれば必ずコンパイルさせられる。
- HLO の diff よりも先に、jaxls へ渡す**入力側**(FK パラメータ)を diff
  する方が早い(量子化後 FK 定数の最大絶対差・符号付き ULP 差を見る)。
- GPU 版 jax は起動時にデバイスメモリ確保に失敗するたびに
  `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` の警告を出しながら要求
  サイズを縮小することがある(正常動作、キャッシュとは無関係)。
  `run_camera_pipeline_test.py` は `XLA_PYTHON_CLIENT_PREALLOCATE=false` を
  既定で設定している。他のスクリプトで抑えたい場合は同じ環境変数や
  `XLA_PYTHON_CLIENT_MEM_FRACTION` を使う。

## 6. `--bag` 起動時のデッドロック(未修正)

`run_camera_pipeline_test.py --bag <file>` は、`rospy.set_param
('/use_sim_time', True)` → `rospy.init_node` → `HandshakePipelineNode(args)`
(warmup・実機接続を含む `__init__` 一式)が**完了してから** `rosbag play
--clock` をサブプロセスとして起動する実装になっている。そのため `/clock` が
一切配信されていない状態で `__init__` 内の TF 待ち等に入り、**起動が永久に
デッドロックする**(実機コントローラの無い開発環境で発生を確認)。

回避策: `rosbag play <bag> --clock` を**先に別プロセスとして起動して
`/clock` を配信させておいてから**、ノードは `--bag` を付けずに起動する
(記録済みクリップの color/depth/camera_info/tf/tf_static は既定のトピック
名と一致するため、`--bag` なしでもそのまま subscribe される)。
切り出し bag の `replay.sh` は、初期化中だけ TF をループ再生して同じ問題を避けている
([test_data.md](test_data.md))。

```bash
rosparam set /use_sim_time true
rosbag play data/real/2026-09-17_session1/session1.bag --clock &
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

## 8. `solve_palm_ik.py` と `plan_handshake_motion.py` の warmup

- `solve_palm_ik.py` は `main()` が人物ループに入る前に `_warmup_batch_ik`
  (バッチ IK、`solve_person_ik` を両腕分)を実行する。ダミー目標には
  `human_body_obstacles({})` を使い、実際の対象者と同じ shape で jit を
  トレース/コンパイルさせる。`ik_kwargs` は人物ループと同じものを渡す
  (違うと別のコンパイル)。入力に IK 対象(`offered_hand` が L/R)が 1 人も
  いなければバッチ IK 自体が呼ばれないので、warmup はスキップする。
- `plan_handshake_motion.py` は `main()` が人物ループの前に `_warmup_solver`
  (左右の腕の軌道最適化、空の骨格で足りる)を実行し、
  `--side-by-side-transition` 指定時は横並び移動のバッチ IK の warmup も
  行う。いずれもスキップするオプションは無い。
- 結果が処理順に依存しないよう、`solve_palm_ik.py` は `main()` で
  `np.random.seed(0)` し、人物ごとにファイル名の `zlib.crc32` で乱数シードを
  決め直す。

## 9. 台車可動域の人物ごとの変更は再コンパイルを起こさない(解決済み)

`solve_palm_ik.py` の `restrict_base_yaw_range_to_human_facing`/
`restrict_base_y_range_to_hand_side`(`main()` が人物ごとの
`person_base_limits` を作る箇所)は、台車の可動域 (`base_limits`) を人物ごとに
異なる連続値へ絞る。`base_limits` は `batch_inverse_kinematics` 内部で仮想
台車関節 (`_attach_batch_virtual_base_chain`) の `min_angle`/`max_angle`
になるが、以前はこれが `solve_batched` (jax.jit) のクロージャに定数として
焼き込まれ、人物ごとに HLO が変わって「1 人につき 1 回」再コンパイル
(約 9 秒/人)していた。

**対策 (scikit-robot フォーク):** `differentiable.py` の
`create_batch_ik_solver` で `joint_limits_lower`/`joint_limits_upper` を
クロージャ定数ではなく `solve_single`/`solve_batch` の通常引数 (trace 対象)
として渡すようにした。`robot_model.py` 側も `_batch_ik_collision_solver_cache`
のキャッシュキーから台車の可動域を外し、呼び出しごとの実際の範囲を渡す。
これにより人物ごとに `base_limits` が変わっても再コンパイルは起きない
(18 人処理で 2 回連続実行してもキャッシュ増分 0 を確認済み)。

`run_camera_pipeline_test.py` の `HandshakePlanner.plan` も同じ制限
(`restrict_base_y_range_to_hand_side`/`restrict_base_yaw_range_to_human_facing`)
を呼んでおり、両スクリプトで挙動が揃っている。

## 10. `human_body_obstacles` の使い回し

軌道の事後検証(`plan_handshake_motion.verify_waypoints`、waypoint 数だけ
ループ)や IK 2 段階目の `solve_palm_ik.pick_verified_candidate`(収束した
候補の数だけループ)は、同じ骨格に対して `collision_pairs_min_distance` を
繰り返し呼ぶ。人体側の障害物(`Cylinder` 群)と表面サンプルは骨格が同じ
間は不変なので、`collision_pairs_min_distance` の `obstacle_links`/
`obstacle_samples` 引数と `plan_handshake_motion.build_obstacle_cache` で
1 人あたり 1 回だけ作って使い回している(結果は不変)。

ベンチマーク(cProfile など)でこの種の再構築コストを評価するときは、
実際に使うロボット構成(`Aero(use_hand=False)`、手指抜き)と検証ペア数を
揃えること。手指込みの `Aero()` だとペア数が桁違いに多く、人体側の再構築の
比重を過小評価する。

## 11. `--force-optimize` の意味

`plan_handshake_motion.py`(`run_pipeline_test.py` 経由でも指定可)の
`--force-optimize` は、pre-touch/線形補間が事後検証に通っても jaxls の
軌道最適化を 1 回実行させる、**軌道最適化そのものの計算時間を測るための
ベンチマーク用オプション**(通常運用では指定しない)。付けると平均計算時間は
約 2 倍以上になる。

一方、pre-touch も線形補間も検証を通らない人は `--force-optimize` が無くても
jaxls の最適化ループ(`--motion-attempts`、既定 3 回の warm start リトライ)
まで進む。そのため「1 人あたりの最大計算時間」はこの検証失敗ケースの
リトライコストで決まり、`--force-optimize` の有無ではほぼ変わらない。

## 12. 干渉ペアが無いときのバッチ IK

干渉ペアの JSON (`scripts/collision_pairs.json`) が無い、または 0 組のとき、
`solve_person_ik` は干渉回避を切る代わりに「自己干渉だけ有効・リンク 1 つ
(`body_link`)」で `batch_inverse_kinematics` を呼ぶ(組が作れず干渉コストは
0)。干渉回避を切るとヤコビアン法になり、台車を動かすと人ごとに再コンパイル
して 1 人約 9 秒かかるため(`tools/build_collision_pairs.py` の手順 3)。
今はペアありと同じ勾配降下法なので、腕ごとに 1 回のコンパイルで済む。

## 13. 現行コードでの warmup と 1 人あたり計算時間の実測(2026-10-15)

テストデータの置き場所は docs/test_data.md。WSL (aero-uv venv、jax キャッシュあり、実機なし) で測った値。

### `scripts/run_pipeline_test.py 400 --seed 0 --plan-motion --side-by-side-transition` (合成 400 人)

IK 対象 (差し出した人) は 62 人で、IK 成功 60・軌道 verified 59・横並び 59/59。
42 人 (`--seed 0`) だと対象が 6 人しかなく、平均が揺れるので 400 人で測った。

| 段階 | warmup | 1 人あたり平均 | 最大 |
|---|---|---|---|
| IK (`solve_palm_ik.py`、1+2 段階目) | 13.4 秒 (l 7.5 + r 5.9) | 0.63 秒 (1 段階目 0.35 + 2 段階目 0.28) | 4.63 秒 |
| 軌道計画 (`plan_handshake_motion.py`、横並び除く) | 32.2 秒 (l 18.4 + r 13.8) | 0.15 秒 | 0.27 秒 |
| 横並び移動 (transition) | 上の軌道計画の warmup に含む | 1.38 秒 | 11.22 秒 |
| 掌推定 | なし | 0.001 秒 | - |

- warmup 合計は約 46 秒 (IK 13.4 + 軌道 32.2)。2 つのスクリプトが別プロセスなので足し算になる。
- 1 人あたりの合計 (IK + 軌道 + 横並び) は平均約 2.2 秒。最大は人ごとにばらつくので、各段階の最大の単純な合計 (約 16 秒) より小さい。
- 最大は IK 4.6 秒・横並び 11.2 秒とも外れ値で、平均の 7〜8 倍。
- 最大は各人の JSON (`handshakes/*.json` の `collision_ik_time`+`candidate_selection_time`、`motions/*.json` の `compute_time`・`transition.compute_time`) から集計した (スクリプトの表示は平均のみ)。

### `scripts/ros/run_camera_pipeline_test.py` (切り出し bag の再生、0.5 倍速)

| 試行 | 結果 | IK | 軌道 | 横並び |
|---|---|---|---|---|
| `2026-10-15/trial1_side_by_side/` | 成功 | 1.5 秒 | 0.1 秒 | 4.00 秒 |
| `2026-10-15/trial2_side_by_side/` | 成功 | 1.7 秒 | 0.2 秒 | 0.72 秒 |
| `2026-10-13/trial22_side_by_side/` | 押し込み姿勢が解けず軌道計画なし | - | - | - |

- warmup は起動から「[warmup] 完了」まで 60〜62 秒 (計 3 回: 61.7 / 60.2 / 60.2)。run_pipeline_test.py の IK+軌道 warmup の合計 (約 46 秒) より長い。
- 1 人あたりの IK+軌道は 1.6〜1.9 秒 (横並び込みで 2.3〜5.9 秒)。ただし成功したのは 2 試行 (各 1 人) だけなので平均・最大としては参考値。run_camera_pipeline_test.py は再生ごとに 1 人しか処理しないので、数を稼ぐには何度も再生する必要がある。
- 同じ bag でも掌の向きが毎回ずれるので成否・時間が変わる (上記の境界ケース)。trial22 は今回失敗。
- `RATE=1.0` で再生すると bag が先頭 1 秒で止まり人が検出されなかった。時間を測るときは既定の 0.5 倍速で回す。
