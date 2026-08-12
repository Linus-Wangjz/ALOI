import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# Path to your Times New Roman .ttf file
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
dram_types = ['DDR5 (x4)', 'DDR5 (x8)', 'DDR5 (x16)', 'LPDDR5', 'AiM\n(GDDR6)', 'FIMDRAM\n(HBM2)']
Device_capacity_means = [243, 122, 61, 48, 37, 68]  # GB
Device_capacity_errors = [159, 80, 40, 41, 19, 50]  # GB
# internal_bandwidth = [3.2, 6.4, 12.8, 12.8, 32, 25.6]  # GB/s per Bank
# internal_bandwidth = [1.6, 3.2, 6.4, 6.4, 32, 12.8]  # GB/s per Bank
internal_bandwidth = [3.2, 3.2, 1.6, 3.2, 16, 16]  # TB/s per Bank

# dram_types = ['DDR5\n(x4)', 'DDR5\n(x8 x16)', 'LPDDR5', 'GDDR6', 'HBM3']
# module_capacity_means = [30.4, 30.4, 11.9, 18.4, 13.6]  # Gb
# module_capacity_errors = [19.9, 19.9, 10.3, 9.6, 9.9]  # Gb
# # internal_bandwidth = [102.4, 204.8, 204.8, 512, 819.2]  # GB/s per Bank
# internal_bandwidth = [51.2, 102.4, 102.4, 512, 409.6]  # GB/s per Bank


# Plotting two separate charts side by side as per the provided style

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
font=20

# First plot: Internal Bandwidth
ax1.bar(dram_types, internal_bandwidth, color='grey', width=0.6)
# ax1.set_ylabel('PIM Bandwidth per Bank (GB/s)', fontsize=font)
# ax1.set_ylim(0.01, 34.99)
for label in ax1.get_xticklabels():
    label.set_fontproperties(font_prop)
    label.set_rotation(90) 
for label in ax1.get_yticklabels():
    label.set_fontproperties(font_prop)
ax1.tick_params(axis='x', labelsize=font)  # Increase tick label size
ax1.tick_params(axis='y', labelsize=font)  # Increase tick label size
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# Second plot: Bank Capacity with error bars
ax2.bar(dram_types, Device_capacity_means, yerr=Device_capacity_errors, color='grey',
# ax2.bar(dram_types, module_capacity_means, yerr=module_capacity_errors, color='grey',
        error_kw=dict(ecolor='black', elinewidth=2.5, capsize=10, capthick=2))
# ax2.set_ylabel('Bank Capacity (Gb)', fontsize=font)
# ax2.set_ylim(0.01, 1.99)
for label in ax2.get_xticklabels():
    label.set_fontproperties(font_prop)
    label.set_rotation(90) 
for label in ax2.get_yticklabels():
    label.set_fontproperties(font_prop)
ax2.tick_params(axis='x', labelsize=font)  # Increase tick label size
ax2.tick_params(axis='y', labelsize=font)  # Increase tick label size
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Adjust layout for better visualization
fig.tight_layout()
plt.savefig('bandwidth_capacity_analysis.png')
