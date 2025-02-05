#!/bin/bash

CONDA="/Users/aljoscha/miniforge3"
TREX="$CONDA/envs/beta/bin/TRex.app/Contents/MacOS/TRex"
	
# Activate conda environment
source $CONDA/bin/activate beta

MAIN_DIR="$1"
SETTINGS="$2"

# Loop through all directories in the main directory
for dir in "$MAIN_DIR"/*/; do

if [[ "$dir" == *"group"* ]]; then
        	INDIVIDUALS=5
    	elif [[ "$dir" == *"single"* ]]; then
		INDIVIDUALS=1
    	else
        	echo "Skipping directory $dir as it doesn't match 'group' or 'single'."
        	continue
    	fi

    	# Loop through all mp4 files in the current directory
    	for VIDEO in "$dir"*.mp4; do

		echo "Running TRex for $VIDEO with settings $SETTINGS"
		$TREX -i $VIDEO -s $SETTINGS -track_max_individuals $INDIVIDUALS -auto_quit -nowindow
    	done
done

# Deactivate conda environment
conda deactivate

echo "Processing complete!"