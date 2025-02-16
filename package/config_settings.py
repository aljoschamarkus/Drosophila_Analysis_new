# config_settings.py

############################################################################################################
# consistent between experiments
############################################################################################################

quality = [1296, 972] # png with quality[0] x quality[1] pixels
FPS = 30
group_size = 5
stimulation= [2, "µW/mm^2", 625,"nm"] # [percent max stimulation, max stimulation intensity, unit, wavelength, unit]
group_type = ["RGN", "AIB", "AGB"]
condition = ["group", "single"]

############################################################################################################
# variable betweeen experiments
############################################################################################################

# Experiments
dish_radius = 6.5 # cm
bootstrap_reps = 2
stimulation_intensity = 50 # percent max stimulation
main_dir = "/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101" # Main directory with ./group and ./single subdirectories
genotype = ["nompCxCrimson", "WTxCrimson", "nompCxWT"]

# semi necessary
circle_default = [7, 7, 6.5] # x midpoint, y midpoint, radius
data_len = 7191 # Number of frames

############################################################################################################
# analysis
############################################################################################################

speed_threshold = [0, 3] # cm/s
distance_threshold_encounter = 0.5 # cm
encounter_distance_threshold = 0.5 # cm
encounter_duration_threshold = [10, 1800] # frames

############################################################################################################
# plot
############################################################################################################

colors = [['red', 'blue', 'green'], ['salmon', 'cornflowerblue', 'mediumseagreen']]
line_styles = {condition[0]: "-", condition[1]: "--"}  # Line styles for conditions
markers = {condition[0]: "o", condition[1]: "s"}  # Markers for conditions
color_mapping = {
    (cond, geno): color
    for cond, row in zip(condition, colors)
    for geno, color in zip(genotype, row)
}

unit_xy = "cm"
unit_speed = "cm/s"
stimulation_used = f"{stimulation[2]}{stimulation[3]}, {stimulation_intensity * stimulation[0] * 0.01}{stimulation[1]}" # ...nm, ...µW/mm^2