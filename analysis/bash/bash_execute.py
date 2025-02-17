import subprocess

MAIN_DIR = '/Users/aljoscha/Downloads/Hallo1'
SETTINGS_FILE = '/analysis/bash/beta_group.settings'

# Pass the variable as an argument to the Bash script
subprocess.run(["bash", "ffmpeg_video_conversion.sh", MAIN_DIR])

subprocess.run(["bash", "TRex_tracking.sh", MAIN_DIR, SETTINGS_FILE])

