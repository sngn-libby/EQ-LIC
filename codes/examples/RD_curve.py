from matplotlib import pyplot as plt
plt.plot([0.1438, 0.2187, 0.3134, 0.5089], [27.8, 29.86, 31.45, 32.93], marker='o', label='GMM (fp)')
plt.plot([0.1411, 0.2229, 0.3394, 0.5102], [27.77, 29.21, 30.5, 32.38], marker='o', label='GMM (LSQ 8-bit)')

plt.plot([0.147, 0.2347, 0.3655, 0.5435], [27.48, 29.04, 30.65, 32.34], marker='o', label='Joint (fp)')
plt.plot([0.145, 0.2323, 0.3643, 0.544], [27.51, 29.07, 30.66, 32.29], marker='o', label='Joint (LSQ 8-bit)')

plt.plot([0.1556, 0.2523, 0.3869, 0.5782], [27.2, 28.73, 30.25, 31.99], marker='o', label='MS (fp)')
plt.plot([0.1578, 0.2508, 0.3889, 0.5757], [27.22, 28.69, 30.14, 31.88], marker='o', label='MS (LSQ 8-bit)')


# plt.plot([0.16, 0.25, 0.39, 0.56], [27.20, 28.72, 30.04, 31.31], marker='o', label='LSQ+ 8-bit')
# plt.plot([0.16, 0.24, 0.39, 0.56, 0.79], [27.19, 28.83, 29.85, 31.48, 32.02], marker='o', label='int offset 8-bit')
# plt.plot([0.15, 0.24, 0.37, 0.54], [27.25, 28.61, 30.00, 31.58], marker='o', label='cubic STE 8-bit')
# plt.plot([0.16, 0.25, 0.39, 0.56, 0.79], [27.18, 28.66, 29.85, 31.47, 32.02], marker='o', label='fp & int offset 8-bit')
# plt.plot([0.17, 0.26, 0.40, 0.60, 0.91], [26.25, 27.53, 28.63, 29.72, 30.94], marker='o', label='LSQ 4-bit')
# plt.plot([0.17, 0.26, 0.39, 0.61, 0.90], [26.22, 27.51, 28.41, 29.58, 31.00], marker='o', label='int offset 4-bit')
# plt.plot([0.16, 0.25, 0.38, 0.57, 0.84], [26.93, 28.48, 29.57, 31.42, 32.62], marker='o', label='LSQ 6-bit')
# plt.plot([0.18, 0.28, 0.42, 0.65, 1.00], [24.16, 24.88, 25.37, 26.03, 26.96], marker='o', label='LSQ 2-bit')

# plt.plot([0.13, 0.28, 0.50], [27.76, 30.42, 32.37], marker='o', label='low bpp (32-bit)')
# plt.plot([0.26, 0.92], [27.53, 30.29], marker='o', label='high bpp (4-bit)')
# plt.hlines(27.53, 0.1, 0.3, color='gray', linestyle='--', linewidth=2)
# plt.hlines(30.29, 0.25, 1, color='gray', linestyle='--', linewidth=2)
# plt.title('')
plt.xlabel('bits per pixel (bpp)')
plt.ylabel('PSNR (dB)')
# plt.ylabel('MS-SSIM')
plt.legend()
plt.show()
