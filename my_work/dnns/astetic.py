from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

# Define the directory containing the CSV files
DIR_PATH = "C:/Masterthesis_unibe/my_work/dnns/Blender_Auswertung/"
file_pattern = os.path.join(DIR_PATH, "*_accuracies.csv")   

# Get a list of matching CSV files
csv_files = glob.glob(file_pattern)
filenames = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

# Print results
cnn_name = []
for name in filenames:
    cnn_name.append(name)

# # Loop through each CSV file
for idx, file in enumerate(csv_files[:1]):  # Limit to first 5 files
    data = pd.read_csv(file)
    acc_data = data['accuracy'].astype(float)
    angle_data = data['angle'].astype(int)
    epoch_data = data['epoch'].astype(int)
    x = acc_data
    y = angle_data
    zs = epoch_data


    # Combine into a DataFrame for easier manipulation
    df = pd.DataFrame({'accuracy': x, 'angle': y, 'epoch': zs})

    # Sort data
    df.sort_values(by=['epoch', 'angle'], inplace=True)

    # set colors for plots. epoch wise
    c_list = get_cmap('summer_r', df['epoch'].nunique())

    # Plotting 
    plt.close('all')

    # Plot setup
    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    
    # # Loop through colors for epochs creating points 
    # for i, (epoch, group) in enumerate(df.groupby('epoch')):
    #     ax.scatter(group['accuracy'], group['angle'], [epoch]*len(group),
    #             color=c_list(i), label=f'Epoch {epoch}', alpha=0.9)
        

    # --- Plot lines within each epoch ---
    for i, (epoch, group) in enumerate(df.groupby('epoch')):
        ax.plot(group['accuracy'], group['angle'], [epoch]*len(group),
                color=c_list(i))

    # # --- Plot lines across epochs for each angle ---
    # for angle, group in df.groupby('angle'):
    #     group_sorted = group.sort_values(by='epoch')
    #     ax.plot(group_sorted['accuracy'], [angle]*len(group_sorted), group_sorted['epoch'],
    #             color='black', linewidth=1.0, alpha=0.5)


    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')


    # --- Optional surface plane ---
    xx = np.full((100, 100), 1/3)
    yy = np.linspace(-180, 180, 100)
    zz = np.linspace(1, 6, 100)
    YY, ZZ = np.meshgrid(yy, zz)
    ax.plot_surface(xx, YY, ZZ, color='salmon', alpha=0.2, edgecolor='none')

    # --- Aesthetic Tweaks ---
    ax.set_xlabel('Accuracy', labelpad=10, fontsize=12)
    ax.set_ylabel('Angle (°)', labelpad=10, fontsize=12)
    ax.set_zlabel('Epoch', labelpad=10, fontsize=12)
    ax.view_init(elev=30, azim=135)

    #ax.grid(False)
    ax.xaxis.line.set_color((0., 0., 0., 0.3))
    ax.yaxis.line.set_color((0., 0., 0., 0.3))
    ax.zaxis.line.set_color((0., 0., 0., 0.3))

        # Customize ticks and grid for better readability
    ax.xaxis._axinfo['grid'].update(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.yaxis._axinfo['grid'].update(color='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.zaxis._axinfo['grid'].update(color='lightgray', linestyle=':', linewidth=0.5, alpha=0.3)

    # Optionally set specific ticks for better legibility
    ax.set_xticks(np.linspace(min(x), max(x), 5))  # Adjust as needed
    ax.set_yticks(np.linspace(-180, 180, 7))       # Example: every 60°
    ax.set_zticks(range(int(min(zs)), int(max(zs)) + 1))  # Epoch ticks

    # Legend
    ax.legend(loc='upper left', fontsize=9, frameon=False, )

    # Title
    plt.title(cnn_name[idx], fontsize=10, fontweight='bold', y=1.05)
    ax.view_init(elev=-25, azim=40, roll=-60)
    plt.show()

