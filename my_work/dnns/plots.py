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

# Dynamic color list based on the number of points
colors = ('r', 'b', 'g', 'c', 'm', 'y')

c_list= []
for c in colors:
    c_list.extend([c]* 73)

# for c in colors:
#     c_list.extend([c]* 73)
# # Loop through each CSV file
for idx, file in enumerate(csv_files[:5]):  # Limit to first 5 files

    print(file)
    data = pd.read_csv(file)

    acc_data = data['accuracy'].astype(float)
    angle_data = data['angle'].astype(int)
    epoch_data = data['epoch'].astype(int)

    

    x = acc_data
    y = angle_data
    zs = epoch_data

    # Ensure x, y, zs have the same length
    if len(x) != len(y) or len(x) != len(zs):
        print(f"Skipping {file}: inconsistent data lengths.")
        continue

    # Check for NaN or Inf values
    if x.isna().any() or y.isna().any() or zs.isna().any():
        print(f"Skipping {file}: NaN values found in the data.")
        continue

    # Plotting
    plt.close('all')
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(projection="3d")

    # Scatter plot with dynamic color list
    ax.scatter(x, y, zs, c = c_list)


    # Set dynamic axis limits based on the data range
    ax.set_xlim(0, max(x))  # X-axis range
    ax.set_ylim(min(y), max(y))  # Y-axis range
    ax.set_zlim(min(zs), max(zs))  # Z-axis range


    # Create a surface at x = 1/3
    xx = np.full((100, 100), 1/3)  # Constant x value
    yy = np.linspace(-180, 180, 100)  # Y-axis range
    zz = np.linspace(1, 6, 100)  # Z-axis range
    YY, ZZ = np.meshgrid(yy, zz)  # Create a meshgrid

    ax.plot_surface(xx, YY, ZZ, color='r', alpha=0.5)  # Semi-transparent red plane


    # Axis labels
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Angle')
    ax.set_zlabel('Epoch')
    ax.view_init(elev=-45, azim=-70, roll=50)
    plt.title(cnn_name[idx], y=1.05)




    # # Save the figure with the base filename
    # base_name = os.path.basename(file).split("_")[0]
    # plot_filename = os.path.join(DIR_PATH, f"{base_name}_3d_plot.png")
    # plt.savefig(plot_filename)
    #plt.tight_layout()
    plt.show()
