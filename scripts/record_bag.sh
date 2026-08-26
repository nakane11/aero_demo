#!/bin/bash

# 保存先のディレクトリを作成（存在しない場合）
mkdir -p ~/rosbags

# 現在の日時をファイル名に含める
FILENAME=~/rosbags/human_tracking_$(date +%Y%m%d_%H%M%S).bag

echo "Recording rosbag to $FILENAME"
echo "Press Ctrl+C to stop recording."

rosbag record -O $FILENAME \
  /openni_camera/rgb/image_rect_color \
  /people_pose_estimation_mediapipe/pose \
  /people_pose_estimation_mediapipe/marker \
  /people_pose_estimation_mediapipe/output \
  /head_pose_estimation/output/head_pose \
  /scan \
  /human_tracker/human_state \
  /human_tracker/marker \
  /cmd_vel \
  /tf \
  /tf_static
