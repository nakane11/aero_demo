#!/usr/bin/env python3
# -*- coding:utf-8 -*-

## CMakeLists.txt の catkin_python_setup() から呼ばれる。
## 直接 python setup.py ... を実行してはいけない。

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['aero_demo'],
    package_dir={'': 'src'},
)

setup(**d)
