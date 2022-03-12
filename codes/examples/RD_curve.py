from matplotlib import pyplot as plt

plt.plot([0.13, 0.19, 0.28, 0.50, 0.73], [27.76, 29.08, 30.42, 32.37, 34.24], marker='o', label='32-bit (fp)')
plt.plot([0.16, 0.26, 0.38, 0.55, 0.90], [27.42, 28.65, 30.21, 31.53, 32.73], marker='o', label='8-bit')
# plt.plot([0.16, 0.25, 0.38, 0.57, 0.84], [26.93, 28.48, 29.57, 31.42, 32.62], marker='o', label='6-bit')
plt.plot([0.17, 0.26, 0.40, 0.60, 0.91], [26.25, 27.53, 28.63, 29.72, 30.94], marker='o', label='4-bit')
plt.xlabel('bits per pixel (bpp)')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.show()
