#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""study/sbs-comfort: 横並び移動に develop の人の項を入れた案を、保存した実機の
試行で比べる。

IK・軌道の揺れを除くため、押し込みまでの結果 (横並び移動の job) を 1 度だけ
解いて pickle に置き、案ごとに横並び移動だけを解き直す。

  collect DIRS... --jobs J.pkl
      保存した試行 (skeletons/・palms/) を resolve_saved_trials と同じ手順
      (ノードの既定の引数) で解き、横並び移動の job を保存する。
  collect-press --jobs J.pkl
      11/18・11/19 の実機の押し込み後の姿勢 (bag の /joint_states) から解き直す
      job (ノードの --transition-resolve と同じ形、resolve_timing.py と同じ)。
  run --jobs J.pkl --out OUT.json
      job ごとに、early: 確保 (寄せなし) -> 裏の計画 (寄せ、refine_transition)、
      resolve: 押し込み後の解き直し (寄せまで、掌の向きは全部) を解き、人の項と
      一緒に記録する。案は環境変数 (SBS_COMFORT_*・SBS_ULNAR_MAX など) で選ぶ。
"""

import argparse
import copy
import datetime
import glob
import json
import os
import pickle
import re
import sys
import time
import zlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _path in (os.path.join(_REPO_ROOT, 'scripts', 'ros'),
              os.path.join(_REPO_ROOT, 'scripts'),
              os.path.join(_THIS_DIR, 'ros')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import run_camera_pipeline_test as rcpt  # noqa: E402
import sbs_human_term as ht  # noqa: E402

spik, sbs = rcpt.spik, rcpt.sbs


def make_planner(node_argv=()):
    args = rcpt.build_arg_parser().parse_args(list(node_argv))
    rcpt.phm.LEAD_IN_BLEND = bool(args.lead_in_blend)
    planner = rcpt.HandshakePlanner(args)
    spik.attach_camera_optical_coords(planner.robot)
    if planner.refine_robot is not None:
        spik.attach_camera_optical_coords(planner.refine_robot)
    rcpt.phm.arms_down_angles(planner.robot, planner.robot.joint_list)
    planner.warmup()
    return args, planner


# --- collect -----------------------------------------------------------------

def collect(dirs, out):
    args, planner = make_planner()
    trials = [(d, name) for d in dirs
              for name in sorted(os.listdir(os.path.join(d, 'skeletons')))
              if name.endswith('.json')]
    jobs = []
    for i, (data_dir, name) in enumerate(trials):
        joint_positions = spik.load_skeleton_json(
            os.path.join(data_dir, 'skeletons', name))
        with open(os.path.join(data_dir, 'palms', name)) as f:
            palms = json.load(f)
        hand = palms.get('offered_hand')
        rec = dict(kind='early', data_dir=data_dir, name=name, job=None)
        if hand in ('L', 'R') and palms.get(hand) is not None:
            np.random.seed(zlib.crc32(name.encode()))
            start = None
            if getattr(args, 'pre_move', False):
                dist = args.pre_move_max_distance
                if args.pre_move_max_time > 0:
                    dist = min(dist, rcpt.pre_move_distance_for_time(
                        args.pre_move_max_time))
                end = rcpt.pre_move_target(
                    palms[hand]['position'],
                    spik.human_standing_xy(joint_positions),
                    args.pre_move_standoff, max_distance=dist)
                if float(np.linalg.norm(end[:2])) >= \
                        rcpt.PRE_MOVE_MIN_DISTANCE:
                    start = end
            planned = planner.plan(joint_positions, palms, hand, i + 1,
                                   start_base_pose=start)
            early = planned.get('transition_early')
            if early is not None:
                rec['job'] = copy.deepcopy(early)
        jobs.append(rec)
        print('[collect {}/{}] {} {}: {}'.format(
            i + 1, len(trials), data_dir, name,
            'job' if rec['job'] else '-'), flush=True)
    with open(out, 'wb') as f:
        pickle.dump(jobs, f)
    print('[collect] {} 試行、job {} 個を {} に保存'.format(
        len(jobs), sum(1 for j in jobs if j['job']), out))


PRESS_TRIALS = ['20261118_181611/person_01', '20261118_181611/person_02',
                '20261118_183429/person_02', '20261118_183429/person_03',
                '20261118_202605/person_01', '20261118_203516/person_03',
                '20261119_144102/person_01', '20261119_144102/person_02']
PRESS_ROOT = '/tmp/run_camera_pipeline_test_logs'
_NECK = {'neck_y_joint', 'neck_p_joint', 'neck_r_joint'}
_LINE = re.compile(r'^(\d\d):(\d\d):(\d\d)\.(\d+) \[execute\] 手を下ろしながら')


def _translate_post(pp, off):
    pp = copy.deepcopy(pp)
    for key in ('target_position', 'hand_position', 'base_position'):
        if pp.get(key) is not None:
            pp[key][0] += off[0]
            pp[key][1] += off[1]
    for base in pp.get('press_in_base_path') or []:
        base[0] += off[0]
        base[1] += off[1]
    return pp


def collect_press(out):
    """resolve_timing.py (11/19) と同じ押し込み後の姿勢の job。"""
    import rosbag
    args, planner = make_planner()
    robot = planner.refine_robot
    bags = []
    for b in sorted(glob.glob(os.path.expanduser(
            '~/ros/hand/src/aero_demo/data/bags/'
            'camera_pipeline_2026-11-1[89]*.bag'))):
        with rosbag.Bag(b) as bag:
            bags.append((bag.get_start_time(), bag.get_end_time(), b))
    jobs = []
    for trial in PRESS_TRIALS:
        d, person = trial.split('/')
        day = datetime.datetime.strptime(d[:8], '%Y%m%d')
        t_line = None
        for line in open('{}/{}/{}.log'.format(PRESS_ROOT, d, person),
                         encoding='utf-8', errors='replace'):
            m = _LINE.match(line)
            if m:
                t_line = (day + datetime.timedelta(
                    hours=int(m.group(1)), minutes=int(m.group(2)),
                    seconds=int(m.group(3)) + float('0.' + m.group(4)))
                ).timestamp()
        name = None
        for f in sorted(glob.glob('{}/{}/motions/*.json'.format(PRESS_ROOT,
                                                                 d))):
            mt = datetime.datetime.strptime(os.path.basename(f)[:15],
                                            '%Y%m%d_%H%M%S').timestamp()
            if mt <= t_line:
                name = os.path.basename(f)
        bag = [b for b in bags if b[0] <= t_line <= b[1]][0][2]
        js = None
        with rosbag.Bag(bag) as bg:
            for _, msg, _ in bg.read_messages(
                    topics=['/joint_states'],
                    start_time=rcpt.rospy.Time.from_sec(t_line - 1.0),
                    end_time=rcpt.rospy.Time.from_sec(t_line)):
                if len(msg.name) > 10:
                    js = dict(zip(msg.name, msg.position))
        base_dir = '{}/{}'.format(PRESS_ROOT, d)
        joints = spik.load_skeleton_json(os.path.join(base_dir, 'skeletons',
                                                      name))
        hs = json.load(open(os.path.join(base_dir, 'handshakes', name)))
        mo = json.load(open(os.path.join(base_dir, 'motions', name)))
        hand, arm, turn = hs['offered_hand'], hs['robot_arm'], hs['turn_deg']
        off = spik.human_translation_offset(
            joints, front_distance=args.human_front_distance)
        tj = spik.translate_joint_positions(joints, off)
        cj = spik.shift_torso_joints_from_surface(tj, off,
                                                  args.torso_surface_offset)
        post = _translate_post(hs['post_process'], off)
        ex = (mo.get('transition_refined')
              if mo.get('transition_executed') == 'refined'
              else mo.get('transition'))
        names = post['joint_names']
        av = np.array([js.get(n, v) if n not in _NECK else v
                       for n, v in zip(names, post['joint_angle_vector'])])
        new = copy.deepcopy(post)
        new['joint_angle_vector'] = [float(v) for v in av]
        sbs._place(robot, av, np.array([post['base_position'][0],
                                        post['base_position'][1],
                                        post['base_yaw']]))
        c = getattr(robot, '{}arm_end_coords'.format(arm))
        new['target_position'] = [float(v) for v in c.worldpos()]
        new['target_rot'] = c.worldrot().tolist()
        new['hand_position'] = new['target_position']
        new['hand_rot'] = new['target_rot']
        jobs.append(dict(
            kind='resolve', data_dir=base_dir, name='{}_{}'.format(person,
                                                                   name),
            executed_facing=(ex or {}).get('palm_facing'),
            job=dict(robot_arm=arm, hand=hand, post=new, turn_deg=turn,
                     translated_joints=tj, collision_joints=cj, offset=off,
                     time_limit=args.transition_resolve_time_limit)))
        print('[collect-press] {} {}'.format(trial, name), flush=True)
    with open(out, 'wb') as f:
        pickle.dump(jobs, f)
    print('[collect-press] job {} 個を {} に保存'.format(len(jobs), out))


# --- run ---------------------------------------------------------------------

def _lateral_end(tr, joints, hand):
    ap = tr.get('approach') or {}
    if ap.get('n_waypoints'):
        return float(ap['lateral_after'])
    if tr.get('verified') and tr.get('waypoints'):
        return sbs.human_lateral(joints, tr['waypoints'][-1]['base_position'],
                                 hand)
    return None


def summarize(tr, job):
    """計画 ``tr`` の記録 (人の項・横の距離・時間)。"""
    joints, hand = job['translated_joints'], job['hand']
    ok = bool(tr.get('verified') and tr.get('fraction', 0.0) >= 1.0)
    out = dict(ok=ok, fraction=tr.get('fraction'),
               time=tr.get('compute_time'), reason=tr.get('reason'),
               palm_facing=tr.get('palm_facing'), arm_deg=tr.get('arm_deg'),
               abduction_deg=tr.get('abduction_deg'),
               timed_out=tr.get('timed_out'))
    if not ok:
        return out
    press_turn = tr.get('turn_before', job['turn_deg'])
    arm = job['robot_arm']
    wps = [wp for wp in tr['waypoints'] if 'human_arm' in wp]
    totals, terms = [], []
    for wp in wps:
        t, e = sbs.waypoint_comfort(joints, hand, wp, press_turn, arm)
        totals.append(t)
        terms.append(e)
    end = wps[-1]
    # 測り方を揃えた比較用: develop の求め直し・ctb の腕の両方で終点を測る。
    measures = {}
    for mode in ('develop', 'skeleton'):
        t, e = sbs.waypoint_comfort(joints, hand, end, press_turn, arm, mode)
        measures[mode] = dict(total=t, terms=dict(zip(
            ht.TERM_NAMES, [float(v) for v in e])))
    # 指先の向き: ロボット (手先 +Z) と人 (手の指先) の真下からの角度 [度]。
    ha = end['human_arm']
    robot_rot = sbs.spik.palm_target_rot(
        sbs.palm_from_frame(ha['palm_position'], np.asarray(ha['palm_rot'])),
        end['turn_deg'], arm)
    human_rot = sbs.comfort_hand_rot(ha['palm_position'],
                                     np.asarray(ha['palm_rot']),
                                     end['turn_deg'], press_turn, arm)

    def down_deg(v):
        return float(np.degrees(np.arccos(np.clip(-v[2], -1.0, 1.0))))
    fingers = dict(robot_down_deg=down_deg(sbs.finger_direction(robot_rot)),
                   human_down_deg=down_deg(human_rot[:, 0]),
                   palm_rot_human_down_deg=down_deg(
                       np.asarray(ha['palm_rot'])[:, 0]))
    # 同じ人の掌のまま握りの向きだけ変えたときの、ロボットの指先の一番下向き。
    palm_end = sbs.palm_from_frame(ha['palm_position'],
                                   np.asarray(ha['palm_rot']))
    fingers['robot_best_down_deg'] = min(
        down_deg(sbs.finger_direction(sbs.spik.palm_target_rot(
            palm_end, float(t), arm))) for t in range(-180, 180, 2))
    fingers['palm_normal_z'] = float(np.asarray(ha['palm_rot'])[2, 1])
    ap = tr.get('approach') or {}
    out.update(
        lateral=_lateral_end(tr, joints, hand),
        lateral_before_approach=ap.get('lateral_before'),
        approach_mode=ap.get('mode') if ap.get('n_waypoints') else None,
        approach_reason=ap.get('reason'),
        alt_arm_reason=(ap.get('alt_arm') or {}).get('reason'),
        comfort_end=totals[-1],
        comfort_terms_end=dict(zip(ht.TERM_NAMES,
                                   [float(v) for v in terms[-1]])),
        comfort_path_max=max(totals),
        comfort_measures_end=measures,
        fingers_end=fingers,
        turn_end=end.get('turn_deg'),
        palm_z_end=float(end['human_arm']['palm_position'][2]),
        wrist_end=end['human_arm'].get('robot_hand_wrist'),
        joints_end=end['human_arm'].get('joints'),
        human_max=tr.get('human_max'))
    return out


def run(jobs_path, out, only=None):
    args, planner = make_planner()
    with open(jobs_path, 'rb') as f:
        jobs = pickle.load(f)
    variant = {k: v for k, v in os.environ.items()
               if k.startswith('SBS_') or k.startswith('MFIX_')}
    records = []
    for i, rec in enumerate(jobs):
        job = rec['job']
        if job is None or (only and rec['name'] not in only):
            continue
        r = dict(data_dir=rec['data_dir'], name=rec['name'], kind=rec['kind'],
                 hand=job['hand'], robot_arm=job['robot_arm'])
        np.random.seed(zlib.crc32(rec['name'].encode()))
        planner._restore_refine_robot()
        robot = planner.refine_robot
        common = (robot, job['robot_arm'], job['hand'], job['post'],
                  job['turn_deg'], job['translated_joints'],
                  job['collision_joints'], planner.refine_verification_pairs,
                  planner.refine_collision_pairs, planner.base_limits)
        t0 = time.time()
        if rec['kind'] == 'early':
            secured = sbs.plan_transition(
                *common, approach=False,
                time_limit=args.early_approach_transition_time)
            r['secured'] = summarize(secured, job)
            r['secured']['wall'] = time.time() - t0
            executed = secured
            if secured['verified']:
                planner._restore_refine_robot()
                t1 = time.time()
                refined = sbs.refine_transition(*common,
                                                copy.deepcopy(secured))
                r['refined'] = summarize(refined, job)
                r['refined']['wall'] = time.time() - t1
                if (refined.get('verified')
                        and refined['fraction'] >= secured['fraction']):
                    executed = refined
                    r['executed_which'] = 'refined'
                else:
                    r['executed_which'] = 'secured'
            r['executed'] = summarize(executed, job)
        else:
            tr = sbs.plan_transition(*common, approach=True, alt_arms=False,
                                     time_limit=job['time_limit'])
            r['executed'] = summarize(tr, job)
            r['executed']['wall'] = time.time() - t0
            r['executed_facing_real'] = rec.get('executed_facing')
        records.append(r)
        ex = r['executed']
        print('[run {}/{}] {} ok={} lat={} comfort={} t={:.1f}'.format(
            i + 1, len(jobs), rec['name'], ex['ok'],
            '%.2f' % ex['lateral'] if ex.get('lateral') is not None else '-',
            '%.2f' % ex['comfort_end'] if ex.get('comfort_end') is not None
            else '-', time.time() - t0), flush=True)
    with open(out, 'w') as f:
        json.dump(dict(variant=variant, records=records), f, indent=1,
                  ensure_ascii=False, default=float)
    print('[run] {} に保存'.format(out))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('collect', 'collect-press', 'run'))
    p.add_argument('dirs', nargs='*')
    p.add_argument('--jobs', required=True)
    p.add_argument('--out')
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    if a.mode == 'collect':
        collect(a.dirs, a.jobs)
    elif a.mode == 'collect-press':
        collect_press(a.jobs)
    else:
        run(a.jobs, a.out, a.only)


if __name__ == '__main__':
    main()
