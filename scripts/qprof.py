# -*- coding:utf-8 -*-
"""計算時間の内訳の計測 (環境変数 ``MFIX_QPROF`` = 出力ディレクトリ のときだけ有効)。

develop の Q0 (``scripts/qprof.py``、3352223・d4c9f88・7bdf134) の legacy の部品だけの版。develop の
goal_ik (PoseGoalSolver・goal_e・M3 など) と qcheck を包む所は master に無いので外し、legacy の
横並び移動・寄せの中の段 (``side_by_side_transition``) と人の腕の判定を足した。

``install()`` を各スクリプトの main の先頭 (計測したいモジュールの import の後) で呼ぶ。無効なら何もしない
(関数を包まないので、結果も時間も変わらない)。関数を包んで、段階 (stage) ごとに次の分類の排他的な時間
(入れ子の内側の分を引いた時間) を足し込む:

- ``solve``: 数値の解く処理の本体 (skrobot のバッチ IK・ヤコビアン IK、軌道最適化の ``JaxlsSolver.solve``、
  横並びの ``_batch_ik``・``_press_ik_once``)。中の FK は solve に含める。
- ``compile``: jax のトレース・lowering・バックエンドのコンパイル (永続キャッシュの読み込みを含む、
  jax.monitoring のイベント)。
- ``fk``: skrobot の FK (``CascadedCoords.update`` の再計算・``angle_vector`` の設定) と台車ごと置く
  ``sbs._place``。
- ``verify``: 事後検証 (干渉・人体との距離、``TransitionChecker``、軌道の点の検証、人体の障害物の作り直し
  ``human_body_obstacles``)。
- ``human``: 人の腕の計算・判定 (``HumanArm`` の下ろし方・姿勢・目安の判定、人の手首の判定、経路の掌の道のり)。
- ``py``: 段階の時間のうち上のどれでもない分 (Python の処理)。

段階は包んだ関数の名前の入れ子 (``refine_transition/plan_tr/stage1/path`` など)。スレッドごとに別に数え、
終了時に ``<出力ディレクトリ>/<スクリプト名>_<pid>.json`` に書く。``MFIX_QCPROFILE=1`` ならメインスレッドの
cProfile も ``<スクリプト名>_<pid>.pstats`` に書く。集計は ``tools/q_profile_summary.py``・
``tools/q_profile_report.py``。
"""

import atexit
import collections
import functools
import json
import os
import sys
import threading
import time

ENV = 'MFIX_QPROF'
CPROFILE_ENV = 'MFIX_QCPROFILE'
_perf = time.perf_counter
_tls = threading.local()
_lock = threading.Lock()
_threads = []          # 全スレッドの状態 (終了時にまとめる)
_installed = False
_wrapped_modules = set()
_T_INSTALL = None
_COMPILE_EVENTS = (
    '/jax/core/compile/jaxpr_trace_duration',
    '/jax/core/compile/jaxpr_to_mlir_module_duration',
    '/jax/core/compile/backend_compile_duration',
)


def enabled():
    return bool(os.environ.get(ENV))


class _State(object):
    def __init__(self):
        self.stack = []                                   # [stage, cat, t0, child, 関数名]
        self.acc = collections.defaultdict(float)         # (stage, cat) -> 秒
        self.fn_acc = collections.defaultdict(float)      # (stage, cat, 関数名) -> 秒 (排他的)
        self.count = collections.Counter()                # (stage, 関数名) -> 回数
        self.calls = []                                   # solve の 1 回ごと: (stage, 名前, 候補数, 秒, compile 秒)
        self.episodes = []                                # 包んだ一番外の関数 1 回ごとの内訳
        self.compile_depth = collections.Counter()
        self.compile_events = collections.defaultdict(float)
        self.thread = threading.current_thread().name


def _state():
    st = getattr(_tls, 'st', None)
    if st is None:
        st = _State()
        _tls.st = st
        with _lock:
            _threads.append(st)
    return st


def _stage(st):
    return st.stack[-1][0] if st.stack else '(none)'


def _push(st, stage, cat, fn=None):
    st.stack.append([stage, cat, _perf(), 0.0, fn])


def _pop(st):
    stage, cat, t0, child, fn = st.stack.pop()
    el = _perf() - t0
    st.acc[(stage, cat)] += el - child
    if fn is not None:
        st.fn_acc[(stage, cat, fn)] += el - child
    if st.stack:
        st.stack[-1][3] += el
    return el


def _snapshot(st):
    return dict(st.acc), len(st.calls)


def _episode_record(st, name, stage, wall, before, extra):
    acc0, ncalls0 = before
    delta = {}
    for key, v in st.acc.items():
        d = v - acc0.get(key, 0.0)
        if d > 1e-9:
            delta['{}|{}'.format(*key)] = d
    rec = dict(name=name, stage=stage, wall=wall, t_end=time.time(), delta=delta,
               solves=list(st.calls[ncalls0:]))
    rec.update(extra)
    st.episodes.append(rec)


def _is_static(owner, attr):
    return isinstance(owner, type) and isinstance(owner.__dict__.get(attr), staticmethod)


def wrap_stage(owner, attr, label, episode=False):
    """``owner.attr`` を段階 ``label`` として包む (段階の入れ子に足す)。``episode`` なら 1 回ごとの内訳を残す。"""
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, '_qprof', False):
        return
    is_static = _is_static(owner, attr)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        st = _state()
        parent = _stage(st)
        stage = label if parent == '(none)' else parent + '/' + label
        top = episode and not any(f[1] == 'py' and f[0].split('/')[-1] == label for f in st.stack)
        before = _snapshot(st) if top else None
        _push(st, stage, 'py')
        st.count[(stage, attr)] += 1
        ok = False
        try:
            out = fn(*args, **kwargs)
            ok = True
            return out
        finally:
            el = _pop(st)
            if top:
                _episode_record(st, label, stage, el, before, dict(ok=ok))
    wrapper._qprof = True
    setattr(owner, attr, staticmethod(wrapper) if is_static else wrapper)


def wrap_cat(owner, attr, cat, record_calls=False, size_fn=None, name=None):
    """``owner.attr`` を分類 ``cat`` として包む (段階は呼び出し元のまま)。関数名は ``name`` (既定 attr)。"""
    fn = getattr(owner, attr, None)
    if fn is None or getattr(fn, '_qprof', False):
        return
    is_static = _is_static(owner, attr)
    fname = name or attr

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        st = _state()
        # solve の中の FK・検証は solve に含める (legacy の IK は中で FK を回す)。
        if st.stack and st.stack[-1][1] == 'solve':
            return fn(*args, **kwargs)
        stage = _stage(st)
        st.count[(stage, fname)] += 1
        comp0 = st.acc.get((stage, 'compile'), 0.0)
        _push(st, stage, cat, fname)
        try:
            return fn(*args, **kwargs)
        finally:
            el = _pop(st)
            if record_calls:
                n = None
                if size_fn is not None:
                    try:
                        n = size_fn(args, kwargs)
                    except Exception:  # noqa: BLE001
                        n = None
                st.calls.append((stage, fname, n, el, st.acc.get((stage, 'compile'), 0.0) - comp0))
    wrapper._qprof = True
    setattr(owner, attr, staticmethod(wrapper) if is_static else wrapper)


def _wrap_fk_update(cls):
    orig = cls.update
    if getattr(orig, '_qprof', False):
        return

    def update(self, force=False):
        if not force and not self._changed:
            return
        st = _state()
        if st.stack and st.stack[-1][1] in ('fk', 'solve'):
            return orig(self, force)
        _push(st, _stage(st), 'fk')
        try:
            return orig(self, force)
        finally:
            _pop(st)
    update._qprof = True
    cls.update = update


def _wrap_angle_vector(cls):
    orig = cls.angle_vector
    if getattr(orig, '_qprof', False):
        return

    def angle_vector(self, av=None, return_av=None):
        if av is None:
            return orig(self, av, return_av)
        st = _state()
        if st.stack and st.stack[-1][1] in ('fk', 'solve'):
            return orig(self, av, return_av)
        _push(st, _stage(st), 'fk')
        try:
            return orig(self, av, return_av)
        finally:
            _pop(st)
    angle_vector._qprof = True
    cls.angle_vector = angle_vector


def _on_scalar(event, value, **kwargs):
    if event in _COMPILE_EVENTS:
        _state().compile_depth[event] += 1


def _on_duration(event, duration, **kwargs):
    if event not in _COMPILE_EVENTS:
        if event.startswith('/jax/compilation_cache/'):
            _state().compile_events[event] += duration
        return
    st = _state()
    st.compile_depth[event] -= 1
    if st.compile_depth[event] > 0:
        return  # 入れ子のトレース (外側で数える)
    st.compile_events[event] += duration
    stage = _stage(st)
    st.acc[(stage, 'compile')] += duration
    if st.stack:
        st.stack[-1][3] += duration


def _install_jax_listeners():
    try:
        from jax._src import monitoring
    except Exception:  # noqa: BLE001
        return
    monitoring.register_scalar_listener(_on_scalar)
    monitoring.register_event_duration_secs_listener(_on_duration)


def _batch_size(args, kwargs):
    """``batch_inverse_kinematics`` の解く数 (目標の数 x 初期値の数)。"""
    targets = kwargs.get('target_coords', args[1] if len(args) > 1 else None)
    attempts = kwargs.get('attempts_per_pose', 1)
    return len(targets) * int(attempts)


def _sbs_batch_size(args, kwargs):
    """``sbs._batch_ik`` の解く数 (目標の数 x 初期値の数)。"""
    targets = args[3] if len(args) > 3 else kwargs['targets']
    attempts = args[8] if len(args) > 8 else kwargs['attempts_per_pose']
    return len(targets) * int(attempts)


def _modules():
    mods = {}
    for name in ('solve_palm_ik', 'side_by_side_transition', 'plan_handshake_motion',
                 'run_camera_pipeline_test'):
        m = sys.modules.get(name)
        if m is not None:
            mods[name] = m
    main = sys.modules.get('__main__')
    if main is not None:
        base = os.path.splitext(os.path.basename(getattr(main, '__file__', '') or ''))[0]
        if base in ('solve_palm_ik', 'plan_handshake_motion', 'run_camera_pipeline_test'):
            mods[base + '@main'] = main
    return mods


def _wrap_solve_palm_ik(m):
    wrap_stage(m, 'solve_person_ik_side_by_side', 'IK', episode=True)
    wrap_stage(m, 'solve_person_ik', 'hover_ik')
    wrap_stage(m, 'pick_verified_candidate', 'pick')
    wrap_stage(m, 'solve_post_process', 'post_process')
    wrap_stage(m, 'solve_press_in', 'press_in')
    wrap_stage(m, 'refine_post_process', 'refine_post', episode=True)
    wrap_stage(m, '_warmup_batch_ik', 'warmup_batch_ik', episode=True)
    for v in ('collision_pairs_min_distance', 'collision_pair_distances',
              'grouped_clearance_shortfall', 'human_obstacle_clearances',
              'other_hand_base_clearance', 'select_other_arm_posture',
              'human_body_obstacles', 'human_body_obstacles_reused'):
        wrap_cat(m, v, 'verify')


def _wrap_sbs(m):
    wrap_stage(m, 'plan_transition', 'transition', episode=True)
    wrap_stage(m, 'refine_transition', 'refine_transition', episode=True)
    wrap_stage(m, 'warmup_batch_ik', 'warmup_sbs_batch_ik', episode=True)
    wrap_stage(m, '_plan_transition', 'plan_tr')
    wrap_stage(m, '_plan_stage1', 'stage1')
    wrap_stage(m, 'solve_goal_candidates', 'goal_candidates')
    wrap_stage(m, '_plan_path_impl', 'path')
    wrap_stage(m, 'plan_approach', 'approach')
    wrap_stage(m, '_add_approach', 'add_approach')
    wrap_stage(m, '_direct_to_approach_end', 'direct_end')
    wrap_stage(m, '_add_alt_arm_approach', 'add_alt_arm')
    wrap_stage(m, '_approach_alt_arms', 'alt_arms')
    wrap_stage(m, 'fix_goal_wrist', 'fix_wrist')
    wrap_stage(m, '_aim_head', 'aim_head')
    wrap_cat(m, '_batch_ik', 'solve', record_calls=True, size_fn=_sbs_batch_size)
    wrap_cat(m, '_press_ik_once', 'solve', record_calls=True)
    wrap_cat(m, '_place', 'fk')
    TC = m.TransitionChecker
    for v in ('check', 'check_base_path', 'forearm_clearances', 'set_press'):
        wrap_cat(TC, v, 'verify', name='TransitionChecker.' + v)
    wrap_cat(TC, '__init__', 'verify', name='TransitionChecker.__init__')
    HA = m.HumanArm
    for v in ('__init__', 'lowering_problem', 'lowering', 'lowering_pose', 'pose',
              'skeleton', 'robot_hand_wrist', 'batch_lowering_palms'):
        wrap_cat(HA, v, 'human', name='HumanArm.' + v)
    for v in ('_lowering_path_steps', 'wrist_problem', 'wrist_ok_turn',
              'wrist_angles_for_turns'):
        wrap_cat(m, v, 'human')


def _wrap_phm(m):
    wrap_stage(m, 'plan_person_motion', 'traj', episode=True)
    wrap_stage(m, '_warmup_solver', 'warmup_traj', episode=True)
    for v in ('verify_waypoints', 'verify_human_clearance', 'verify_lead_in',
              'other_hand_min_clearance', 'build_obstacle_cache'):
        wrap_cat(m, v, 'verify')


def _wrap_node(m):
    HP = getattr(m, 'HandshakePlanner', None)
    if HP is not None:
        wrap_stage(HP, 'plan', 'node_plan', episode=True)
        wrap_stage(HP, '_warmup_body', 'node_warmup', episode=True)
        wrap_stage(HP, '_optimizer_warmup_job', 'warmup_traj_job', episode=True)


def install():
    """有効なら関数を包み、終了時の書き出しを登録する (何度呼んでもよい)。無効なら何もしない。"""
    global _installed, _T_INSTALL
    if not enabled():
        return
    _install_cprofile()
    _state()
    if not _installed:
        _T_INSTALL = _perf()
        _installed = True
        _install_jax_listeners()
        atexit.register(dump)
        from skrobot.coordinates.base import CascadedCoords
        from skrobot.model.robot_model import RobotModel
        _wrap_fk_update(CascadedCoords)
        _wrap_angle_vector(RobotModel)
        wrap_cat(RobotModel, 'inverse_kinematics', 'solve', record_calls=True)
        wrap_cat(RobotModel, 'batch_inverse_kinematics', 'solve', record_calls=True,
                 size_fn=_batch_size)
        try:
            from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver
            wrap_cat(JaxlsSolver, 'solve', 'solve', record_calls=True,
                     name='JaxlsSolver.solve')
        except Exception:  # noqa: BLE001
            pass
    for key, m in _modules().items():
        if id(m) in _wrapped_modules:
            continue
        _wrapped_modules.add(id(m))
        name = key.split('@')[0]
        if name == 'solve_palm_ik':
            _wrap_solve_palm_ik(m)
        elif name == 'side_by_side_transition':
            _wrap_sbs(m)
        elif name == 'plan_handshake_motion':
            _wrap_phm(m)
        elif name == 'run_camera_pipeline_test':
            _wrap_node(m)


_CPROFILE = None


def _install_cprofile():
    """``MFIX_QCPROFILE=1`` なら、メインスレッドを cProfile で測り、終了時に
    ``<出力ディレクトリ>/<スクリプト名>_<pid>.pstats`` に書く (関数ごとの Python の時間を見るため)。"""
    global _CPROFILE
    if _CPROFILE is not None or os.environ.get(CPROFILE_ENV, '0') != '1':
        return
    import cProfile
    _CPROFILE = cProfile.Profile()
    _CPROFILE.enable()

    def _dump_cprofile():
        _CPROFILE.disable()
        out_dir = os.environ[ENV]
        os.makedirs(out_dir, exist_ok=True)
        script = os.path.splitext(os.path.basename(sys.argv[0] or 'python'))[0]
        _CPROFILE.dump_stats(os.path.join(out_dir, '{}_{}.pstats'.format(script, os.getpid())))
    atexit.register(_dump_cprofile)


def dump(path=None):
    """集計を JSON に書く (終了時に自動で呼ぶ)。"""
    if not enabled():
        return None
    out_dir = os.environ[ENV]
    os.makedirs(out_dir, exist_ok=True)
    if path is None:
        script = os.path.splitext(os.path.basename(sys.argv[0] or 'python'))[0]
        path = os.path.join(out_dir, '{}_{}.json'.format(script, os.getpid()))
    with _lock:
        threads = list(_threads)
    data = dict(argv=sys.argv, pid=os.getpid(), time=time.time(), threads=[],
                wall=_perf() - _T_INSTALL if _T_INSTALL is not None else None)
    for st in threads:
        data['threads'].append(dict(
            thread=st.thread,
            acc=[[k[0], k[1], v] for k, v in st.acc.items()],
            fn_acc=[[k[0], k[1], k[2], v] for k, v in st.fn_acc.items()],
            count=[[k[0], k[1], v] for k, v in st.count.items()],
            calls=st.calls, episodes=st.episodes,
            compile_events=dict(st.compile_events)))
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, default=float)
    os.replace(tmp, path)
    return path
