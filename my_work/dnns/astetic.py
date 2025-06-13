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

# # Loop through each CSV file
for idx, file in enumerate(csv_files[1:9]):  # Limit to first 5 files
    list_with_name_and_trainingmodus = str.split(file, sep='\\')[1].split(sep='.')[0].split(sep='_')[:2]
    list_with_name_and_trainingmodus.append(".png") # looks like alexnetdiverse.png
    filesave_name = ''.join(list_with_name_and_trainingmodus)
    whitespace = 1  # space insert
    list_with_name_and_trainingmodus[whitespace:whitespace] = [' ']  # Insert "3" within "b"
    string_for_title= ''.join(list_with_name_and_trainingmodus[:3]) 


    data = pd.read_csv(file)
    acc_data = data['accuracy'].astype(float)
    angle_data = data['angle'].astype(int)
    epoch_data = data['epoch'].astype(int)
    x = epoch_data
    y = angle_data
    zs = acc_data


    # Combine into a DataFrame for easier manipulation
    df = pd.DataFrame({'epoch': x, 'angle': y, 'accuracy': zs})


    # Sort data
    df.sort_values(by=['epoch', 'angle'], inplace=True)

    # Set number of colors = number of unique epochs
    num_epochs = df['epoch'].nunique()

    # Plotting 
    plt.close('all')

    # Plot setup
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
        
            
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    for i, (epoch, group) in enumerate(df.groupby('epoch')):
        group = group.sort_values(by='angle').reset_index(drop=True).copy()

        # Color logic


        group['colour'] = group['angle'].apply(assign_color)

        # Create segments: [(x0, y0, z0), (x1, y1, z1)]
        segments = []
        colors = []
        for j in range(len(group) - 1):
            seg = [
                (group['epoch'][j], group['angle'][j], group['accuracy'][j]),
                (group['epoch'][j + 1], group['angle'][j + 1], group['accuracy'][j + 1])
            ]
            segments.append(seg)
            colors.append(group['colour'][j])  # color by the first point of the segment

        # Add line segments to the plot
        lc = Line3DCollection(segments, colors=colors, linewidths=2)
        ax.add_collection3d(lc)

        # Also plot scatter points
        ax.scatter(group['epoch'], group['angle'], group['accuracy'], color=group['colour'], s=20)

        
        # ax.plot([epoch]*len(group), group['angle'], group['accuracy'],
        #         group['colour'])

    # # --- Plot lines within each epoch ---
    # for i, (epoch, group) in enumerate(df.groupby('epoch')):
    #     ax.plot([epoch]*len(group), group['angle'], group['accuracy'],
    #             color=c_list(i))


    # --- Optional surface plane ---
    x_min, x_max = df['epoch'].min(), df['epoch'].max()
    xx = np.linspace(x_min, x_max, 200)
    yy = np.linspace(-180, 180, 200)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.full_like(XX, 1/3)

    ax.plot_surface(XX, YY, ZZ, color='salmon', alpha=0.1, edgecolor='none'), 

    # Set axis limits to match surface
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-180, 180)
    ax.set_zlim(0, 1)

    # --- Aesthetic Tweaks ---
    ax.set_xlabel('Epochs', labelpad=10, fontsize=12)
    ax.set_ylabel('Angle (°)', labelpad=10, fontsize=12)
    ax.set_zlabel('Accuracy', labelpad=10, fontsize=12)
    ax.view_init(elev=30, azim=135)

    # Customize ticks and grid for better readability
    ax.xaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.yaxis._axinfo['grid'].update(color='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.zaxis._axinfo['grid'].update(color='lightgray', linestyle='--', linewidth=0.5, alpha=0.3)

    # # Create custom legend handles
    # legend_elements = [
    #     Patch(facecolor='k', label='Trainingviews (-30° or 30°)'),
    #     Patch(facecolor=palette[4], label='Interpolation (-30° < x < 30°)'),
    #     Patch(facecolor=palette[5], label= 'Extrapolation (x < -30 or x > 30)')
    # ]

    # # Add the legend manually
    # ax.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=False)

    # Title
    plt.title(string_for_title, fontsize=10, fontweight='bold', y=1.05)


    plt.savefig(filesave_name, bbox_inches='tight')

    # plt.show()

