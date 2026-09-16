#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""aero_demo のライブラリ (ROS ノードではないモジュール).

ここに置くモジュールは rospy を import せず、ノードとしても起動しない。
ROS ノードはパッケージ直下の scripts/ros/ に置き、このパッケージを import
して使う (rospy に依存しないパイプラインのテスト・ツール用スクリプトは
scripts/ 直下に置く)。

  people_pose_types          … 姿勢推定結果のデータ型
  people_pose_estimator      … MediaPipe による人物姿勢推定
  palm_plane                 … 手のランドマークから手のひら平面を最小二乗推定
  palm_plane_view            … 骨格を skrobot の viewer に描く部品
  skeleton_drawing           … 骨格の viser 表示・画像への重ね描き
  aero_urdf_setup            … 手あり Aero URDF (aero_with_feetech_hand.urdf)
                               を ROS を source していなくても読み込めるよう
                               準備する (load_aero)
"""
