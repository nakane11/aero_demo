# jax / jaxls のコンパイル・キャッシュ仕様まとめ

`solve_palm_ik.py`(バッチ IK)と `plan_handshake_motion.py`(jaxls の軌道
最適化)は、どちらも jax の JIT コンパイルに依存しており、初回コンパイルは
数秒〜数十秒かかる。この文書は、そのコンパイル・永続キャッシュまわりで
これまでの調査から分かった仕様・既知の問題・対策をまとめる。README.md の
「環境構築」からはリンクのみを残し、詳細はこちらに集約する。

## 1. 永続コンパイルキャッシュの基本

`solve_palm_ik.py` は起動時に jax の永続コンパイルキャッシュ(既定で
`~/.cache/jax_compilation_cache`、`JAX_COMPILATION_CACHE_DIR` 環境変数で
変更できる)を有効にする。干渉回避付きバッチ IK / jaxls の軌道最適化の
JIT コンパイルは、計算グラフの形状(`--collision-ik-stop`/
`--attempts-per-pose`/`--skeleton-dir` の有無/使う腕(`--robot-arm`)などで
決まる)ごとに初回だけ必要な重い処理(数分〜十数十秒かかることがある)で、
以降は同じ venv/jax バージョンで同じ形状の計算であればディスクキャッシュ
から即座に読み込まれる。

キャッシュのキー(フィンガープリント)はコンパイル前 HLO から決まる。
**ソースの行番号や空白などは HLO に含まれるがキャッシュキーには影響しない**
(スクリプトを編集してもキャッシュは効き続けることを確認済み)。一方、
jaxls が計算グラフに**定数として焼き込む値**(ロボットの姿勢由来の FK
パラメータなど)が 1 bit でも違うと別計算とみなされ再コンパイルになる。
詳細は「4. 既知だったキャッシュミスの原因と対策」を参照。

## 2. warmup とキャッシュの単位(腕ごとに別キャッシュ)

起動時のウォームアップ(`run_camera_pipeline_test.py` の `_warmup_ik`)は、
左右それぞれの腕で IK(`solve_person_ik`)だけでなく `plan_person_motion`
(jaxls の軌道最適化)もダミー目標に対して 1 回ずつ解いておき、JAX/jaxls
の初回コンパイルを前倒しで済ませる。scikit-robot 側の `JaxlsSolver` は
コンパイル済み問題を腕ごと(`collision_link_list` が変わるため l/r で
構造が異なる)に別々のキャッシュとして保持するため、この事前ウォームアップ
で両腕分がキャッシュされた状態になり、以後 ARMED のたびに差し出し手の
左右が入れ替わっても再コンパイルは起きない。

ただし通常運用では pre-touch/線形補間だけで事後検証に通ることが多く、
その場合は jaxls 自体が呼ばれない(`plan_handshake_motion.py` の
`--force-optimize` を付けると常に jaxls まで実行させられる。単独で
jaxls の計算時間を計測したいベンチマーク用のオプション)。

## 3. コンパイル時間の内訳(トレース/XLAコンパイル/実行)

2026-09-14 の調査(`scratch/verify_aot_split.py`。jaxls 内部の
`NonlinearSolver.solve`(`@jdc.jit` = `jax.jit` のラッパー)を
`.lower()`/`.compile()`/実行の 3 段階に分離して計測)で、軌道最適化 1 回
分(左腕)の内訳が判明した:

| フェーズ | コールド(ディスクキャッシュ無) | ヒット(ディスクキャッシュ有) |
|---|---|---|
| lower(Python側のトレース、計算グラフ構築) | 6.3〜7.7秒 | 6.3秒 |
| compile(XLAコンパイル) | 29.5秒(コード生成) | 3.0秒(キャッシュからの読み込み) |
| execute(LM反復の実行、初回呼び出し) | 0.44秒 | 0.37秒 |
| 合計 | 37.7秒 | 9.7秒 |

実測合計は実ノード(roscore + rosbag 再生、9 回連続起動)の実測値
(cold 34.9秒/36.9秒、hit 平均 左12.81秒/右11.20秒)とほぼ一致する
(IK フェーズ分を加えると `plan_person_motion` 全体で左腕 12.29 秒程度)。

重要な点:

- **「実行(LM反復そのもの)」は全体の 1〜4% しか占めない。** cold-hit の
  差(約 28 秒)の正体はほぼ全て XLA コンパイル自体(コード生成 vs
  キャッシュ読み込みの差)であり、実行コストではない。
- **`lower`(トレース)はディスクキャッシュの対象外。** `.lower()` は
  XLA に渡す前段階で Python 側の計算グラフ(jaxpr)を構築するステップで、
  jax_compilation_cache のキー生成より前の処理のため、ディスクキャッシュ
  がどれだけ効いていても毎回 6〜8 秒かかる。ヒット時の合計 9.7 秒のうち
  6 割以上をこの `lower` が占める。
- したがって、ヒット時でもコンパイルまわりだけで(`lower` 6.3秒 +
  `compile` 3.0秒 =)9.3 秒程度かかるのは仕様であり、「キャッシュが効いて
  いれば速いはず」という直感には反する。キャッシュが短縮するのは
  `compile` のみ。

## 4. 既知だったキャッシュミスの原因と対策(解決済み)

過去に 2 種類のキャッシュミス問題があり、いずれも「jaxls が計算グラフに
焼き込む FK 由来の定数が実行ごとに微妙にブレる」ことが原因だった
(scikit-robot fork `base_limit` ブランチ)。

1. **軌道最適化(`jit_solve`)側**: `_warmup_ik` の乱数シード未固定により
   IK の解自体が毎回わずかに違い、さらに skrobot が関節角を差分回転で
   適用するため下位ビットのブレが後続の腕に伝播していた。対策は
   (a) warmup 中だけ `np.random.seed` を固定、
   (b) `jaxls_solver.py` に `_quantize_fk_constants` を追加し FK 定数を
   小数 9 桁に丸めたうえで `+0.0` して負ゼロ(`-0.0`)を正規化
   (`np.round` は符号を保持するため、丸めても `-0.0` が残ると別定数として
   扱われフィンガープリントが変わってしまう)。
   → warmup 全体 47〜75秒 → 約23秒(commit `92bcc65` ほか)。
2. **バッチ IK(`jit_solve_batch`)側**: 上記の丸め処理が
   `differentiable.py`(`extract_fk_parameters`/干渉球中心座標の計算)には
   適用されていなかった。同じロジックを追加。
   → 右腕 IK 6.9〜7.2秒(常時ミス)→ 2回目以降 1.8秒(ヒット)
   (commit `3ed6358`)。

「l/r 腕の非対称性」に見えていたものはどちらも手の左右差ではなく
**「2 本目に処理される方の腕」で起きる現象**だった(`--arm rl` で順序を
入れ替えると再現先も入れ替わることを確認済み)。

## 5. 未解決の残課題

左腕(1 本目)の**軌道最適化**が、実ノードで**低頻度(8〜9 回に 1 回程度)
の intermittent cache miss** を起こすことがある(2026-09-14 時点、原因
未確定・優先度低で保留)。入力姿勢(`angle_vector`/`worldpos`/`worldrot`)
は miss した回とヒットした回で bit 完全一致することを確認済みなので、
原因は FK 計算そのものの中(numpy 演算のスレッド非決定性など、量子化 9 桁
丸めの境界にたまたま乗る値がある、等)にあると推測されるが未確定。次に
調べるなら `scratch/warmup_fk_dump.py` + `scratch/compare_fk_dump.py` で
miss 時とヒット時の量子化後 FK 定数そのものを比較するのが次の一手。

2026-09-15 の実ノード再検証(8. 参照)では、8 回中 3 回で miss が発生し
(発生率としては 8〜9 回に 1 回よりかなり高い)、しかも miss したのは
右腕(2 本目)単独が 2 回、左右同時が 1 回で、**左腕単独の miss は一度も
発生しなかった**。「1 本目固定」という前提とは一致しない結果であり、
サンプル数が 8 回と少なく偶然の可能性もある一方、その再検証では毎回
`SIGINT` → 2 秒待ち → `SIGKILL` でノードを強制終了しており、jax の
ディスクキャッシュへの書き込みが完了する前にプロセスが死ぬと次回の
当該エントリが不完全になり見かけ上の miss を誘発する可能性がある
(強制終了に伴う人為的なノイズの可能性を排除できていない)。実運用に
近い「自然な起動・終了」での頻度は未確認のまま。

## 6. AOT (`jax.jit(...).lower().compile()`) について分かったこと

「warmup が実行結果を使い捨てているなら、`.lower().compile()` で
コンパイルだけ済ませて実行(execute)を省略すれば速くなるのでは」という
仮説を検証したが、**「3. コンパイル時間の内訳」の通り execute は全体の
1〜4% しかないため、効果はほぼ無い**(左腕で 0.4 秒程度の節約に留まる)。
一方で warmup が持っていた「ダミー目標が本当に解けるか」の実行時検証を
失うデメリットの方が大きいため、本番コードへの導入は見送った。

技術的な副次知見(将来 jaxls 内部を直接いじる際の参考):

- このリポジトリの venv は Python 3.11 のため、jaxls は PEP 695 の
  generics 構文(`class Foo[*Args]:`)を使わない互換モジュール
  `jaxls._py310._problem`/`jaxls._py310._solvers` を実際には使っている
  (`jaxls/__init__.py` が `sys.version_info` で分岐している)。
  `jaxls._problem` を直接 import すると `SyntaxError` になるので注意。
- `@jdc.jit`(= `jax.jit` の薄いラッパー)でデコレートしたインスタンス
  メソッドは、通常の呼び出し(`instance.method(...)`)は自動で `self` が
  bind されるが、**`.lower()`/`.compile()` にアクセスすると binding が
  外れる**。`instance.method.lower(instance, ...)` のように `self` を
  明示的に渡す必要がある(渡さないと `missing 1 required positional
  argument: 'self' 相当` のエラーになる)。
- `jdc.Static[...]` でマークした static 引数(例: `return_summary`)は
  `.lower()` 時には渡すが、コンパイル済み関数 `compiled(...)` を呼ぶ際に
  **再度渡してはいけない**(渡すと `Function compiled with input pytree
  does not match the input pytree it was called with` エラーになる。
  static 引数は既に compile 時点で計算グラフに焼き込まれているため)。
- compile-only 経路(`.compile()` のみ、実行なし)と通常の `solve()`
  実行経路は、**同じキャッシュキーに書き込まれる**ことを実測で確認済み
  (別プロセスでプロセス A が compile-only、プロセス B が通常経路を実行し、
  B 側で新規キャッシュ書き込みがゼロであることを確認)。

検証に使ったスクリプトは `scratch/verify_aot_split.py`(untracked)。

## 7. デバッグ手法

- **コンパイルが実際に走ったかどうかはログの秒数ではなく
  `~/.cache/jax_compilation_cache` のファイル数の増減で判定する。**
  `JAX_LOG_COMPILES=1`/`JAX_EXPLAIN_CACHE_MISSES=1` は `jit_solve` に
  ついては何も出力しないため当てにならない。
- 犯人(どの定数がブレているか)を特定するには `XLA_FLAGS=--xla_dump_to=DIR`
  を付けて `*.jit_solve.before_optimizations.txt`(コンパイルした時だけ
  出力される、コンパイル前 HLO)を 2 回分取得して diff する。
  `JAX_COMPILATION_CACHE_DIR` を実行ごとに空ディレクトリにすれば必ず
  コンパイルさせられる。
- HLO の diff よりも先に、jaxls へ渡す**入力側**(FK パラメータそのもの)
  を diff する方が早い。`scratch/warmup_fk_dump.py` が
  `phm.build_problem` をラップして焼き込まれる定数を npz に保存し、
  `scratch/compare_fk_dump.py` で 2 つの npz の最大絶対差・符号付き ULP
  差を出せる。
- GPU 版 jax は起動時にデバイスメモリの確保を試み、大きいサイズから
  確保に失敗するたびに `RESOURCE_EXHAUSTED: CUDA_ERROR_OUT_OF_MEMORY` の
  警告を出しながら要求サイズを段階的に縮小していくことがある(これ自体は
  正常動作で、キャッシュとは無関係)。気になる場合は
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` や `XLA_PYTHON_CLIENT_MEM_FRACTION`
  で確保量を抑えられる。

## 8. `--bag` 起動時のデッドロック(未修正、2026-09-15 判明)

`run_camera_pipeline_test.py --bag <file>` は、`rospy.set_param
('/use_sim_time', True)` → `rospy.init_node` → `HandshakePipelineNode(args)`
(実機接続 `AeroROSRobotInterface` を含む `__init__` 一式)が**完了してから**
`rosbag play --clock` をサブプロセスとして起動する実装になっている。その
ため `use_sim_time=True` かつ `/clock` が一切配信されていない状態で
`__init__` 内の TF 待ち等に入り、`/clock` の最初の 1 メッセージが来ないため
**起動が永久にデッドロックする**(実機コントローラの無い開発環境で発生を
確認)。

回避策: `rosbag play <bag> --clock`(必要なら `-r`/`--loop` も)を**先に別
プロセスとして起動して `/clock` を配信させておいてから**、ノードは
`--bag` を付けずに起動する(記録済みクリップの color/depth/camera_info/
tf/tf_static は既定のトピック名と一致するため、`--bag` なしでもそのまま
subscribe される)。

```bash
rosparam set /use_sim_time true
rosbag play data/session1.bag --clock &
python3 scripts/ros/run_camera_pipeline_test.py --auto-arm --no-wait-for-client
```

本体コード(`--bag` 実装)の恒久修正はまだ行っていない(rosbag play の
起動を `rospy.init_node` の直後・`HandshakePipelineNode` 構築の前に
前倒しすれば解消するはず)。

## 9. 実ノードでの warmup 再検証(2026-09-15)

過去の実測値(hit: 左 12.81 秒/右 11.20 秒、cold: 左 34.9 秒/右 36.9 秒、
「3. コンパイル時間の内訳」のオフライン推定)が、本番用の永続キャッシュ
(`~/.cache/jax_compilation_cache`、削除・変更せず読み取り専用として使用)
を使った実ノード起動でも成り立つかを確認する目的で、8. の回避策を使って
`roscore` + `rosbag play data/session1.bag --clock`(ループなし、毎回
新規起動しなおす)+ `run_camera_pipeline_test.py --auto-arm
--no-wait-for-client` を 8 回連続起動し、`_warmup_ik_body` の `print` ログ
から左右各腕の IK(`solve_person_ik`)/軌道最適化(`plan_person_motion`)
時間を集計した。

| 回 | 左IK | 左軌道最適化 | 右IK | 右軌道最適化 | 備考 |
|---|---|---|---|---|---|
| 1 | 2.0s | 13.1s | 1.7s | 12.0s | hit/hit |
| 2 | 2.1s | 13.2s | 1.4s | 37.2s | hit/miss |
| 3 | 2.8s | 12.4s | 1.6s | 11.1s | hit/hit |
| 4 | 3.1s | 13.1s | 1.4s | 36.1s | hit/miss |
| 5 | 2.1s | 11.9s | 1.6s | 11.4s | hit/hit |
| 6 | 2.1s | 13.5s | 1.5s | 12.3s | hit/hit |
| 7 | 2.0s | 38.3s | 1.5s | 37.6s | miss/miss |
| 8 | 2.1s | 13.5s | 1.6s | 10.7s | hit/hit |

hit 時のみ(1,3,5,6,8 の 5 回)の平均:

- 左腕: IK 2.3 秒 + 軌道最適化 12.9 秒(範囲 11.9〜13.5 秒)
- 右腕: IK 1.5 秒 + 軌道最適化 11.5 秒(範囲 10.7〜12.3 秒)

「3. コンパイル時間の内訳」の過去実測(hit 左 12.81 秒/右 11.20 秒)・
オフライン推定(左腕 12.29 秒程度)とよく一致した(いずれも ±5% 程度)。
miss 時の軌道最適化は 36〜38 秒で、これも過去の cold 実測(左 34.9 秒/
右 36.9 秒)とほぼ一致する。

miss の発生率・偏りについては「5. 未解決の残課題」に追記した通り、
過去に記録されていた「左腕(1 本目)が 8〜9 回に 1 回」という頻度・対象と
一致しない結果(8 回中 3 回、右腕優位)になっており、今回の検証方法
(毎回 `SIGKILL` で強制終了)自体が追加の miss を誘発した可能性を含めて
未解明のまま残っている。
