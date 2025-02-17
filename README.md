# Drosophila Experimental Data Analysis

## Overview
> Project to analyse and visualise video data of Drosophila melanogaster behavioural experiments. Extracting Basic movement Parameters such as speed and midline offset, as well as Group features such as neighbour distances and encounter Analysis.


---

## Project structure

### package (directory)
collection of modules relevant for the project
- `config_parameters` contains relevant parameters: video metadata, Arena size, genotypes, plotting choices, etc.
- `util_data_processing` contains functions to process the raw data and extract the relevant features.
- `util_plotting` contains functions to plot the data.
### analysis (directory)
contains the main scripts to analyse the data
#### analysis_main (subdirectory)
reads data, calculation of parameters and plotting
- `MAIN_data_preparation.py` reads the raw data (csv), overlays and scales it and saves it into a multi indexed data frame
- `MAIN_plot.py` enables selection of options and plots the data
#### bash (subdirectory)
contains files for the tracking via `TRex` 2 beta software
- `bash_execute.py` automates the execution of `TRex` via bash scripts
- `beta_groups.settings` contains the settings for the Tracking
- `ffmpeg_video_conversion.sh` converts h264 videos to mp4
- `TRex_tracking.sh` is the bash script to track the videos

## Data Structure

### Required input data structure
The data should be stored in a directory structure as follows:\
- The main directory contains two subdirectories: `sub_dir_groups` and `sub_dir_singles`.
- Each of these subdirectories contains several sub-subdirectories with the data.
- The data is stored in a `data` subdirectory within each sub-subdirectory.


- The `sub_sub_dir_group_name1` are stored individually as a default by the `PiVR`
- The data directories and the png-files are stored this way by `TRex` as a default
```
|-main_dir
|   |-sub_dir_groups
|   |   |-sub_sub_dir_group_name1
|   |   |   |-data
|   |   |   |   |-file0.csv
|   |   |   |   |-[...]
|   |   |   |-sub_sub_dir_group_name1.png
|   |   |   |-[...]
|   |   |-[...]
|   |-sub_dir_singles
|   |   |-sub_sub_dir_single_name1
|   |   |   |-data
|   |   |   |   |-file0.csv
|   |   |   |   |-[...]
|   |   |   |-sub_sub_dir_single_name1.png
|   |   |   |-[...]
|   |   |-[...]
```
The csv-files contain the following (relevant) columns:
- `frame` the frame number
- `MIDLINE_OFFSET` the midline offset of the larva
- `SPEED#wcentroid (cm/s)` the speed of the larva which is not actually in cm/s but in pixels per frame
- `X#wcentroid (cm)` the x-coordinate of the larva
- `Y#wcentroid (cm)` the y-coordinate of the larva

  The multi indexed pandas data frames are stored in a csv file with the following structure:

- df_initial.pkl
    - Index names: ['sub_dir', 'condition', 'genotype', 'individual_id', 'frame']
    - Columns: ['midline_offset', 'speed', 'x', 'y', 'midline_offset_signless']

- df_groups.pkl
    - Index names: ['midline_offset', 'speed', 'x', 'y', 'midline_offset_signless']
    - Columns: ['midline_offset', 'speed', 'x', 'y', 'midline_offset_signless']

- df_group_parameters.pkl
    - Index names: ['sub_dir', 'condition', 'genotype', 'group_type', 'group_id', 'individual_id', 'frame']
    - Columns: ['midline_offset', 'speed', 'x', 'y', 'midline_offset_signless', 'pairwise_distance', 'nearest_neighbor_distance', 'encounter_count']

---

For further information feel free to ask me at `aljoscha.markus@uni-konstanz.de`

---

<!---
your comment goes here
and here
---
### **2. Data Processing** (`process_data.py`)
```python
def clean_data(df):
df.dropna(inplace=True)
return df
```
- Removes missing values from the dataset.

### **3. Visualization** (`plot_results.py`)
```python
import matplotlib.pyplot as plt
def plot_data(df):
    df.plot(x='Timestamp', y='Measurement 1')
    plt.show()
```
- Plots the first measurement over time.

## How to Use the Code
Step-by-step guide to running the code.

Example:
1. Place your input file (`experiment_data.csv`) in the project folder.
2. Run the main script:
   ```sh
python main.py --input experiment_data.csv
   ```
3. The output will include:
    - Processed data in `processed_data.csv`
          - A graph displayed in a new window.

## Notes
- If you encounter an error, check the input data format.
- The script requires Python 3.x and the Pandas/Matplotlib libraries.
- Contact [Your Name] for questions or improvements.
-->
