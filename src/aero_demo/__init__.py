#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""aero_demo のライブラリ (ROS ノードではないモジュール).

ここに置くモジュールは rospy を import せず、ノードとしても起動しない。
ROS ノードはパッケージ直下の scripts/ に置き、このパッケージを import
して使う。

  people_pose_types          … 姿勢推定結果のデータ型 (EstimationResult 含む)
  people_pose_estimator      … MediaPipe による人物姿勢推定
  palm_plane                 … 手のランドマークから手のひら平面を最小二乗推定
                               (geometry_msgs/Point は使うが rospy は不要)
  palm_plane_view            … 手のひら平面・骨格・ロボットを skrobot の
                               viewer に描く部品 (scripts/ の可視化ノードと
                               接触動作ノードが共有する)
  aero_urdf_setup            … 手あり Aero URDF (aero_with_feetech_hand.urdf)
                               を ROS を source していなくても読み込めるよう
                               準備する (load_aero)

カメラ無しで偽の姿勢を生成する版は rospy を使うので
scripts/fake_people_pose_estimator_ros.py にある。
"""
