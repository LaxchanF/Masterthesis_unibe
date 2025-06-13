from PIL import Image
from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import seaborn as sns
from matplotlib.patches import Patch


# Assign one color for each condition
def assign_color(angle):
    if angle < -30:
        return palette[5]  # e.g., dark
    elif -30 < angle < 30:
        return palette[4]  # medium
    elif angle > 30:
        return palette[5]  # light
    else:
        return 'k'  # fallback

# Generate a continuous cubehelix colormap
palette = sns.color_palette("hls", 8)

# Define the directory containing the CSV files
DIR_PATH = "C:/Masterthesis_unibe/my_work/dnns/Blender_Auswertung/"
file_pattern = os.path.join(DIR_PATH, "*_accuracies.csv")   

# Get a list of matching CSV files
csv_files = glob.glob(file_pattern)
filenames = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

# Names = {}
# for item in filenames:
#     Name, Trainingsmodus = str.split(item, sep='_')[:2]
#     Names[Name] = Trainingsmodus


# Print results
cnn_name = []
for name in filenames:
    cnn_name.append(name)
filesave_name=[]
# # Loop through each CSV file
for idx, file in enumerate(csv_files):  # Limit to first 5 files
    list_with_name_and_trainingmodus = str.split(file, sep='\\')[1].split(sep='.')[0].split(sep='_')[:2]
    list_with_name_and_trainingmodus.append(".png") # looks like alexnetdiverse.png
    filesave_name.append(''.join(list_with_name_and_trainingmodus))
    
print(filesave_name)

imgs = [Image.open(p) for p in filesave_name]


# --- Define grid ---
cols = 5
rows = 2

# --- Get max width and height of each image ---
img_widths = [img.width for img in imgs]
img_heights = [img.height for img in imgs]

max_width = max(img_widths)
max_height = max(img_heights)

# --- Create new blank image for the grid ---
grid_width = cols * max_width
grid_height = rows * max_height

combined = Image.new('RGB', (grid_width, grid_height), color='white')

# --- Paste images in grid ---
for index, img in enumerate(imgs):
    row = index // cols
    col = index % cols
    x_offset = col * max_width
    y_offset = row * max_height
    combined.paste(img, (x_offset, y_offset))

# --- Save final image ---
combined.save("combined_grid.png")
print("Grid image saved as 'combined_grid.png'")