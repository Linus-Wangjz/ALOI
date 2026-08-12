import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# # sudo apt update
# # sudo apt install ttf-mscorefonts-installer
# # sudo fc-cache -fv
# # Path to your Times New Roman .ttf file
# font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams["font.family"] = font_prop.get_name()

# --- 1) Your data: arithmetic intensity (FLOPs/Byte) vs. performance (flops) ---
# Attention kernel

seqlen=4096

llama2_7b_score_flops = 32 * 128 * seqlen * 2  # 32 heads, 128 dim, seqlen=4096
llama3_8b_score_flops = 8 * 4 * 128 * seqlen * 2 / 1e12  # 8 heads, 128 dim, GQA=4, seqlen=4096
llama3_70b_score_flops = 8 * 8 * 128 * seqlen * 2 / 1e12  # 8 heads, 128 dim, GQA=8, seqlen=4096

llama2_7b_score_byte = (32 * 128 + 32 * 128 * seqlen) * 2  # Byte
llama3_8b_score_byte = (32 * 128 + 8 * 128 * seqlen) * 2  # Byte
llama3_70b_score_byte = (64 * 128 + 8 * 128 * seqlen) * 2  # Byte

llama2_7b_output_flops = 32 * 128 * seqlen * 2  # 32 heads, 128 dim, seqlen=4096
llama3_8b_output_flops = 8 * 4 * 128 * seqlen * 2  # 8 heads, 128 dim, GQA=4, seqlen=4096
llama3_70b_output_flops = 8 * 8 * 128 * seqlen * 2  # 8 heads, 128 dim, GQA=8, seqlen=4096

llama2_7b_output_byte = (32 * seqlen + 32 * 128 * seqlen) * 2  # Byte
llama3_8b_output_byte = (32 * seqlen + 8 * 128 * seqlen) * 2  # Byte
llama3_70b_output_byte = (64 * seqlen + 8 * 128 * seqlen) * 2  # Byte

llama2_7b_att_flops = (llama2_7b_score_flops + llama2_7b_output_flops) * 32  # 32 decoders
llama3_8b_att_flops = (llama3_8b_score_flops + llama3_8b_output_flops) * 32  # 32 decoders
llama3_70b_att_flops = (llama3_70b_score_flops + llama3_70b_output_flops) * 80  # 80 decoders

llama2_7b_att_byte = (llama2_7b_score_byte + llama2_7b_output_byte) * 32  # 32 decoders
llama3_8b_att_byte = (llama3_8b_score_byte + llama3_8b_output_byte) * 32  # 32 decoders
llama3_70b_att_byte = (llama3_70b_score_byte + llama3_70b_output_byte) * 80  # 80 decoders

llama2_7b_att_oi = llama2_7b_att_flops / llama2_7b_att_byte
llama3_8b_att_oi = llama3_8b_att_flops / llama3_8b_att_byte
llama3_70b_att_oi = llama3_70b_att_flops / llama3_70b_att_byte

DGX_H100_BW = 3.35*8
CENT_BW = 16*128
att_oi = np.array([llama2_7b_att_oi, llama3_8b_att_oi, llama3_70b_att_oi])
att_perf = np.array([llama2_7b_att_oi * DGX_H100_BW, llama3_8b_att_oi * DGX_H100_BW, llama3_70b_att_oi * DGX_H100_BW])

# Fully‐connected (FC) kernel
dim = 8192
fc_oi_lst = []
fc_flops_lst = []
for batch in [4,8,16,32,64,128]:
    fc_flops = batch * dim * dim * 2  # FC: batch_size * dim * dim * 2 (FLOPs)
    fc_byte = (batch * dim + dim * dim) * 2  # FC: (batch_size * dim + dim * dim) * 2 (Byte)
    fc_oi = fc_flops / fc_byte  # Arithmetic intensity (FLOPs/Byte)
    fc_flops_lst.append(fc_oi * DGX_H100_BW)
    fc_oi_lst.append(fc_oi)
fc_oi    = np.array(fc_oi_lst)
fc_perf  = np.array(fc_flops_lst)
print(f"FC arithmetic intensity (FLOPs/Byte): {fc_oi}")
print(f"FC performance (flops): {fc_perf}")

# optional: vary alpha to mimic the fading markers
att_alpha = np.interp(att_oi, (att_oi.min(), att_oi.max()), (0.3, 1.0))
fc_alpha  = np.interp(fc_oi,  (fc_oi.min(),  fc_oi.max()),  (0.3, 1.0))

# --- 2) Roofline bounds ---
DGX_H100_peak_flops   = 989*8       # peak compute (flops)
DGX_H100_int_thres   = DGX_H100_peak_flops / DGX_H100_BW       # intersection (FLOPs/Byte)
DGX_H100_slope       = DGX_H100_peak_flops / DGX_H100_int_thres

DGX_H100_x_roof = np.logspace(0, 3, 200)
DGX_H100_y_roof = np.minimum(DGX_H100_slope * DGX_H100_x_roof, DGX_H100_peak_flops)

CENT_peak_flops   = 16*128       # peak compute (flops)
CENT_int_thres   = CENT_peak_flops / 16 / 12.8       # intersection (FLOPs/Byte)
CENT_slope       = CENT_peak_flops / CENT_int_thres

CENT_x_roof = np.logspace(0, 3, 200)
CENT_y_roof = np.minimum(CENT_slope * CENT_x_roof, CENT_peak_flops)

DREAM_peak_flops   = 16*4*80       # peak compute (flops)
DREAM_int_thres   = DREAM_peak_flops / CENT_slope       # intersection (FLOPs/Byte)
DREAM_slope       = CENT_slope

DREAM_x_roof = np.logspace(0, 3, 200)
DREAM_y_roof = np.minimum(DREAM_slope * DREAM_x_roof, DREAM_peak_flops)


# --- 3) Plotting ---
plt.figure(figsize=(6, 4))

# roofline
# plt.loglog(CENT_x_roof, CENT_y_roof, linestyle=':', color='blue', linewidth=2)
# plt.loglog(DGX_H100_x_roof, DGX_H100_y_roof, linestyle=':', color='gray', linewidth=2)
plt.plot(CENT_x_roof, CENT_y_roof, linestyle=':', color='blue', linewidth=2)
plt.plot(DREAM_x_roof, DREAM_y_roof, linestyle=':', color='red', linewidth=2)
plt.plot(DGX_H100_x_roof, DGX_H100_y_roof, linestyle=':', color='gray', linewidth=2)

# data points
for x, y, a in zip(att_oi, att_perf, att_alpha):
    plt.scatter(x, y, color='orange', s=100, alpha=a, edgecolors='white', label='_nolegend_')
for x, y, a in zip(fc_oi, fc_perf, fc_alpha):
    plt.scatter(x, y, color='blue',   s=100, alpha=a, edgecolors='white', label='_nolegend_')

# manually build legend
plt.scatter([], [], color='orange', s=100, edgecolors='white', label='Attention')
plt.scatter([], [], color='blue',   s=100, edgecolors='white', label='FC')
plt.legend(loc='upper left', frameon=True)

# vertical line dividing memory‐ vs compute‐bound
# plt.axvline(CENT_int_thres, linestyle='--', color='blue')
# plt.axvline(DGX_H100_int_thres, linestyle='--', color='gray')
plt.text(2,    10, 'memory bound',  fontsize=12)
plt.text(150,  10, 'compute bound', fontsize=12)

# axes labels and limits
plt.xlabel('Arithmetic intensity (FLOPs/Byte)', fontsize=12)
plt.ylabel('Performance (flops)',          fontsize=12)
plt.xlim(1, 400)
plt.ylim(1, 10000)
# plt.grid(which='both', linestyle='--', linewidth=0.5)

# plt.tight_layout()
plt.savefig('roofline_plot.png', dpi=300)
