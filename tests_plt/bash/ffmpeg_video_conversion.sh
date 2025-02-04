#!/bin/bash

# Set base directory
base_dir="/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p"

# Use find and execute ffmpeg directly on each file
find "$base_dir" -type f -name "*.h264" -exec sh -c '
  for video_path do
    dir_path=$(dirname "$video_path")
    video_filename=$(basename "$video_path" .h264)
    mp4_path="$dir_path/$video_filename.mp4"
    
    echo "Converting $video_path to $mp4_path..."
    ffmpeg -hwaccel videotoolbox -r 25 -i "$video_path" -c:v h264_videotoolbox -r 25 -b:v 1750k "$mp4_path"
  done
' sh {} +
