from sewar.full_ref import psnr, ssim, msssim, uqi, scc, sam, ergas, rase, vifp
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

gt = Image.open('C:/flicker_cherry_picked/2801_gt.png')
gt_gray = ImageOps.grayscale(gt)
gt = np.array(gt)
# out = Image.open('C:/flicker_cherry_picked/q3/2801.png')
# out = Image.open('C:/flicker_cherry_picked/q4/2801.png')
out = Image.open('C:/flicker_cherry_picked/LSQ_4bit_q5/2801.png')
out_gray = ImageOps.grayscale(out)
out = np.array(out)

se = np.abs(np.int32(gt) - np.int32(out)) * 5
print(gt.mean())
print(out.mean())
print(se.mean())
se = Image.fromarray(se.astype('uint8'))
se.show()

f_gt = np.fft.fft2(gt_gray)
f_gt = np.fft.fftshift(f_gt)
m_gt = 20 * np.log(np.abs(f_gt))
f_out = np.fft.fft2(out_gray)
f_out = np.fft.fftshift(f_out)
m_out = 20 * np.log(np.abs(f_out))
m_e = m_gt - m_out
plt.imshow(m_e, cmap='gray')
plt.show()

print(f'PSNR: {psnr(gt, out)}')
print(f'SSIM: {ssim(gt, out)}')
print(f'MS-SSIM: {msssim(gt, out)}')
print(f'UQI: {uqi(gt, out)}')
print(f'ERGAS: {ergas(gt, out)}')
print(f'SCC: {scc(gt, out)}')
print(f'SAM: {sam(gt, out)}')
print(f'RASE: {rase(gt, out)}')
print(f'VIF-P: {vifp(gt, out)}')

