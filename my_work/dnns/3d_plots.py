import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams.update({
    'font.size': 12,           # Base font size
    'axes.titlesize': 14,      # Title font size
    'axes.labelsize': 12,      # X/Y label size
    'xtick.labelsize': 10,     # X tick label size
    'ytick.labelsize': 10,     # Y tick label size
    'legend.fontsize': 10,     # Legend font size
    'figure.titlesize': 16     # Figure title (if using suptitle)
})
# ------------------- Color Setup -------------------
colors_selected = [
    "#1b9e77",  # Teal Green - extrapolation line
    "#a6dba0",  # Light Teal - extrapolation scatter
    "#7570b3",  # Muted Purple - interpolation line
    "#dadaeb",  # Light Lavender - interpolation scatter
    "#e7298a"   # Deep Red - training angles scatter
]

def assign_color(angle):
    if angle < -30 or angle > 30:
        return colors_selected[0], colors_selected[1]
    elif -30 < angle < 30:
        return colors_selected[2], colors_selected[3]
    else:
        return colors_selected[0], colors_selected[1]
    




# ------------------- Data Setup -------------------
DIR_PATH = "C:/Masterthesis_unibe/my_work/dnns/Blender_Auswertung/"
file_pattern = os.path.join(DIR_PATH, "*_accuracies.csv")
csv_files = glob.glob(file_pattern)

# ------------------- Plot Loop -------------------
for i, file in enumerate(csv_files[:10]):  # Limit to first 10 files
    df = pd.read_csv(file).sort_values(by=['epoch', 'angle']).reset_index(drop=True)

    filename_parts = os.path.basename(file).replace('.csv', '').split('_')
    title_text = f"{filename_parts[0]}"
    save_name = f"{filename_parts[0]}_{filename_parts[1]}_3dplot.png"
    save_path = os.path.join(DIR_PATH, save_name)

    fig = plt.figure()  # A4 portrait
    ax = fig.add_subplot(projection='3d')

    for epoch, group in df.groupby('epoch'):
        group = group.sort_values(by='angle').reset_index(drop=True).copy()
        group[['line_color', 'scatter_color']] = group['angle'].apply(lambda angle: pd.Series(assign_color(angle)))

        segments = []
        line_colors = []

        for j in range(len(group) - 1):
            angle1 = group['angle'][j]
            angle2 = group['angle'][j + 1]
            p1 = (group['epoch'][j], angle1, group['accuracy'][j])
            p2 = (group['epoch'][j + 1], angle2, group['accuracy'][j + 1])
            segments.append([p1, p2])

            if (angle1 <= -30 and angle2 <= -30) or (angle1 >= 30 and angle2 >= 30):
                color = colors_selected[0]
            elif -30 < angle1 < 30 and -30 < angle2 < 30:
                color = colors_selected[2]
            else:
                color = colors_selected[2]
            line_colors.append(color)

        lc = Line3DCollection(segments, colors=line_colors, linewidths=2)
        ax.add_collection3d(lc)

        mask_training_angles = group['angle'].isin([-30, 30])
        ax.scatter(group['epoch'][mask_training_angles],
                   group['angle'][mask_training_angles],
                   group['accuracy'][mask_training_angles],
                   color=colors_selected[4],
                   s=20,
                   alpha=0.9)

    x_min, x_max = df['epoch'].min(), df['epoch'].max()
    xx = np.linspace(x_min, x_max, 200)
    yy = np.linspace(-180, 180, 200)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.full_like(XX, 1/3)
    ax.plot_surface(XX, YY, ZZ, color='salmon', alpha=0.1, edgecolor='none')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-180, 180)
    ax.set_zlim(0, 1)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Angle (°)')
    ax.set_zlabel('Accuracy')
    ax.set_title(title_text, fontsize=16, pad=5)
    ax.view_init(elev=30, azim=135)

    # Remove gridlines
    ax.grid(False)

    # Save each plot
    plt.tight_layout()
    # plt.show()
    # plt.savefig(save_path, dpi=300, bbox_inches = "tight")
    # plt.close()
    print(f"Saved: {save_path}")