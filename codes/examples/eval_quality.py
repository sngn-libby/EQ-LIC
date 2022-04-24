from sewar.full_ref import uqi, psnr, msssim, scc, sam
from PIL import Image
import numpy as np

gt = Image.open('C:/flicker_cherry_picked/801_gt.png')
gt = np.array(gt)
out = Image.open('C:/flicker_cherry_picked/LSQ 4-bit q5/801_rec.png')
out = Image.open('C:/flicker_cherry_picked/q4/02_36.08_0.170_0.985.png')
out = np.array(out)
print(f'PSNR: {psnr(gt, out)}')
print(f'MS-SSIM: {msssim(gt, out)}')
print(f'UQI: {uqi(gt, out)}')
print(f'SCC: {scc(gt, out)}')
print(f'SAM: {sam(gt, out)}')
