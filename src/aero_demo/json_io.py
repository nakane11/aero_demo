#!/usr/bin/env python3
"""JSON I/O helpers shared by the pose/palm/handshake pipeline scripts."""

import glob
import json
import os


def iter_json_files(directory, pattern='*.json'):
    """Sorted list of files matching ``pattern`` under ``directory``."""
    return sorted(glob.glob(os.path.join(directory, pattern)))


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_skeleton_json(path):
    """Load a skeleton JSON file and return its joint positions."""
    with open(path) as f:
        data = json.load(f)
    return data['skeleton']['joint_positions']
