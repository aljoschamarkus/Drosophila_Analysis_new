#!/bin/bash

# Set base directory
base_dir="$1"

find "$base_dir" -type f -name "*.h264" -exec bash -c '
    video_path="$1"
    dir_path=$(dirname "$video_path")
    video_filename=$(basename "$video_path" .h264)
    mp4_path="$dir_path/$video_filename.mp4"

    echo "Converting $video_path to $mp4_path..."
    ffmpeg -hwaccel videotoolbox -r 25 -i "$video_path" -c:v h264_videotoolbox -r 25 -b:v 1750k "$mp4_path"
' _ {} \;
