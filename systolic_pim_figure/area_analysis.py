import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Path to your Times New Roman .ttf file
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()

# Example x-axis categories
x_labels = ["Vector\nUnit", "1x16\nSA", "2x16\nSA", "4x16\nSA", "8x16\nSA", "16x16\nSA"]
x = range(len(x_labels))
font=24
# Example data for the three curves (approximate)
eight_gb =      [1.25, 1.28, 1.48, 1.86, 2.48, 3.94]
sixteen_gb =    [2.05, 2.08, 2.28, 2.66, 3.28, 4.74]
twentyfour =    [2.85, 2.88, 3.08, 3.46, 4.08, 5.54]
thirtytwo_gb =  [3.65, 3.68, 3.88, 4.26, 4.88, 6.34]
# Create the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
marker_size=10
# Plot the three lines
ax.plot(x, eight_gb, marker='^', markersize=marker_size, color='blue', label='PIM Module with 8Gb Capcity')
ax.plot(x, sixteen_gb, marker='o', markersize=marker_size, color='red', label='PIM Module with 16Gb Capcity')
ax.plot(x, twentyfour, marker='>', markersize=marker_size, color='green', label='PIM Module with 24Gb Capcity')
# ax.plot(x, thirtytwo_gb, marker='s', markersize=marker_size, color='orange', label='PIM Module with 32Gb Capcity')

# Optionally place special star markers to highlight specific points
# (matching the style in your sample figure)
ax.plot(x[-3], sixteen_gb[-3],      marker='*', color='black',   markersize=28)
# ax.plot(x[-2], eight_gb[-2],      marker='*', color='black',   markersize=28)

# Add horizontal reference lines (approximate positions)
# Adjust these y-values and labels as needed
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5)
ax.axhline(y=1.8, color='black', linestyle='--', linewidth=1.5)
ax.axhline(y=2.6, color='black', linestyle='--', linewidth=1.5)
# ax.axhline(y=3.4, color='black', linestyle='--', linewidth=1.5)

# Customize x-axis
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=0)
for label in ax.get_xticklabels():
    label.set_fontproperties(font_prop)
for label in ax.get_yticklabels():
    label.set_fontproperties(font_prop)
ax.set_yticks(range(1, 8))
ax.tick_params(axis='both', which='major', labelsize=font)
ax.set_ylim(0, 7.6)
# Label axes and add a title
# ax.set_xlabel("Processing Unit Architecture", fontsize=font)
ax.set_ylabel("Normalized Area to\nGDDR6 8Gb DRAM Module", fontsize=font)
font_prop.set_size(font)
ax.yaxis.label.set_fontproperties(font_prop)  # Set ylabel font to font_prop
# Show legend
font_prop.set_size(font)
ax.legend(loc='upper left', prop=font_prop)

ax.grid(axis='y', linestyle='--', alpha=0.4)

# Make layout a bit nicer and display
plt.tight_layout()
plt.savefig('area_analysis.png')
