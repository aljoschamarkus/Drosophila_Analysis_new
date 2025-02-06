# config_settings.py

# Fixed parameters
main_dir = "/Users/aljoscha/Downloads/2012_nompC_Crimson_WT_4min_50p_TRex_beta_2101" # Main directory with ./group and ./single subdirectories
condition = ["group", "single"]
genotype = ["nompCxCrimson", "WTxCrimson", "nompCxWT"]
quality = [1296, 972] # png with quality[0] x quality[1] pixels
dish_radius = 6.5
group_size = 5
data_len = 7191 # Number of frames
FPS = 30
stimulation= [50, 2, "µW/mm^2", 625,"nm"] # [percent max stimulation, max stimulation intensity, unit, wavelength, unit]
stimulation_used = f"{stimulation[3]}{stimulation[4]}, {stimulation[0] * stimulation[1] * 0.01}{stimulation[2]}"
# Chosen parameters
circle_default = [7, 7, 6.5] # x midpoint, y midpoint, radius
bootstrap_reps = 2
unit_xy = "cm"
unit_speed = "cm/s"
# Chosen plot parameters
colors = [['red', 'blue', 'green'], ['salmon', 'cornflowerblue', 'mediumseagreen']]
line_styles = {condition[0]: "-", condition[1]: "--"}  # Line styles for conditions
markers = {condition[0]: "o", condition[1]: "s"}  # Markers for conditions
color_mapping = {
    (cond, geno): color
    for cond, row in zip(condition, colors)
    for geno, color in zip(genotype, row)
}