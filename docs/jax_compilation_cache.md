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
軌道最適化 (jaxls) の warmup は、既定では別プロセスで行い、最適化が要る人の
計画だけその完了を待つ(`--no-background-optimizer-warmup` で同一プロセスの
同期実行)。

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

通常運用では線形補間だけで事後検証に通ることが多く、その場合は
jaxls 自体が呼ばれない(`plan_handshake_motion.py` の `--force-optimize` を
付けると常に jaxls まで実行させられる、ベンチマーク用オプション)。

## 3. コンパイル時間の内訳(軌道最適化 1 回分)

`NonlinearSolver.solve` を lower/compile/execute の 3 段階に分離して計測した例
(問題規模によって秒数は変わるが、比率の傾向は同じ)。

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
- 実行は短いので、`.lower().compile()` で execute だけを省く AOT 化はしていない。

## 4. FK定数の量子化(キャッシュミス対策)

jaxls が計算グラフに焼き込む FK 由来の定数が実行ごとに微妙にブレると
別計算とみなされ再コンパイルが起きる。これを避けるため
`jaxls_solver.py`(scikit-robot fork `base_limit` ブランチ)の
`_quantize_fk_constants` で、FK 定数を小数9桁に丸めたうえで `+0.0`
して負ゼロ(`-0.0`)を正規化している(`np.round` は符号を保持するため、
丸めても `-0.0` が残ると別定数として扱われフィンガープリントが変わる)。
同じ丸め処理を `kinematics/differentiable.py`(`extract_fk_parameters`)にも
適用している。

これとは別に、ノード終了時の `SIGKILL` も次回起動時の見かけ上の
キャッシュミスを起こす(7. 参照)。

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

## 6. `--bag` 起動時のデッドロック(既知の問題)

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

rosbag play の起動を `rospy.init_node` の直後・`HandshakePipelineNode` 構築の
前に前倒しすれば解消するはずだが、コードはまだそうなっていない。

## 7. ノードを `SIGKILL` で止めない

`SIGKILL` で強制終了すると、jax の永続コンパイルキャッシュへの非同期書き込みが
完了する前にプロセスが死に、次回起動時に不完全なエントリを読んで見かけ上の
キャッシュミス(再コンパイル)が起きる。

**ノードを止めるときは `SIGINT` を送り、プロセスが自発的に終了するまで待つ。**
これで起動を繰り返してもキャッシュミスは起きない。

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

## 9. 台車可動域の人物ごとの変更は再コンパイルを起こさない

`solve_palm_ik.py` の `restrict_base_yaw_range_to_human_facing`/
`restrict_base_y_range_to_hand_side`(`main()` が人物ごとの
`person_base_limits` を作る箇所)は、台車の可動域 (`base_limits`) を人物ごとに
異なる連続値へ絞る。`base_limits` は `batch_inverse_kinematics` 内部で仮想
台車関節 (`_attach_batch_virtual_base_chain`) の `min_angle`/`max_angle`
になる。これが `solve_batched` (jax.jit) のクロージャに定数として焼き込まれると、
人物ごとに HLO が変わって 1 人につき 1 回 (約 9 秒) 再コンパイルする。

そのため scikit-robot フォークの `differentiable.py` の
`create_batch_ik_solver` は、`joint_limits_lower`/`joint_limits_upper` を
クロージャ定数ではなく `solve_single`/`solve_batch` の通常引数 (trace 対象)
として受け取る。`robot_model.py` 側も `_batch_ik_collision_solver_cache`
のキャッシュキーに台車の可動域を含めず、呼び出しごとの実際の範囲を渡す。
これにより人物ごとに `base_limits` が変わっても再コンパイルは起きない。

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
`--force-optimize` は、線形補間が事後検証に通っても jaxls の
軌道最適化を 1 回実行させる、**軌道最適化そのものの計算時間を測るための
ベンチマーク用オプション**(通常運用では指定しない)。付けると平均計算時間は
約 2 倍以上になる。

一方、線形補間が検証を通らない人は `--force-optimize` が無くても
jaxls の最適化まで進む。そのため「1 人あたりの最大計算時間」はこの検証失敗
ケースの最適化コストで決まり、`--force-optimize` の有無ではほぼ変わらない。

## 12. 干渉ペアが無いときのバッチ IK

干渉ペアの JSON (`scripts/collision_pairs.json`) が無い、または 0 組のとき、
`solve_person_ik` は干渉回避を切る代わりに「自己干渉だけ有効・リンク 1 つ
(`body_link`)」で `batch_inverse_kinematics` を呼ぶ(組が作れず干渉コストは
0)。干渉回避を切るとヤコビアン法になり、台車を動かすと人ごとに再コンパイル
して 1 人約 9 秒かかるため(`tools/build_collision_pairs.py` の手順 3)。
ペアありと同じ勾配降下法なので、腕ごとに 1 回のコンパイルで済む。

## 13. warmup と 1 人あたり計算時間の測り方

テストデータの置き場所は [test_data.md](test_data.md)。計算時間は jax キャッシュが
温まった状態 (同じコードで 2 回目以降) で測る。

### `scripts/run_pipeline_test.py` (合成データ)

```bash
cd scripts
python3 run_pipeline_test.py --input-set ../data/synthetic/seed0_ik_targets \
    --plan-motion --side-by-side-transition
```

- `seed0_ik_targets` (IK 対象 67 人) か `400 --seed 0` で測る。42 人 (`--seed 0`) だと対象が
  数人しかなく、平均が揺れる。
- warmup は `[warmup]` 行と、各段階の「warmup 合計」の表示で見る。`plan_handshake_motion.py`
  の warmup は `solve_palm_ik.py` と並行して走るので (test_data.md)、全体の待ち時間は
  2 つの warmup の和より短い。
- スクリプトの表示は平均だけなので、最大は各人の JSON (`handshakes/*.json` の
  `collision_ik_time`+`candidate_selection_time`、`motions/*.json` の `compute_time`・
  `transition.compute_time`) から集計する。最大は少数の外れ値で決まり、段階ごとの最大の
  単純な合計より小さい。

### `scripts/ros/run_camera_pipeline_test.py` (切り出し bag の再生)

- warmup は起動から「[warmup] 完了」までの時間で見る。
- 再生ごとに 1 人しか処理せず、同じ bag でも掌の向きが毎回ずれるので成否・時間が変わる。
  数を稼ぐには何度も再生する。
- `replay.sh` の既定の 0.5 倍速で回す (`RATE=1.0` では bag が先頭 1 秒で止まり人が検出
  されないことがある)。
