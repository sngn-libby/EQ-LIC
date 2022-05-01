from sewar.full_ref import psnr, ssim, msssim, uqi, scc, sam, ergas, rase, vifp
from PIL import Image
import numpy as np

gt = Image.open('C:/flicker_cherry_picked/801_gt.png')
gt = np.array(gt)
# out = Image.open('C:/flicker_cherry_picked/q3/801.png')
# out = Image.open('C:/flicker_cherry_picked/q4/801.png')
out = Image.open('C:/flicker_cherry_picked/LSQ 4-bit q5/801_rec.png')
out = np.arra
y(out)

se = (gt - out) ** 2
se = Image.fromarray(se)
se.show()

print(f'PSNR: {psnr(gt, out)}')
print(f'SSIM: {ssim(gt, out)}')
print(f'MS-SSIM: {msssim(gt, out)}')
print(f'UQI: {uqi(gt, out)}')
print(f'ERGAS: {ergas(gt, out)}')
print(f'SCC: {scc(gt, out)}')
print(f'SAM: {sam(gt, out)}')
print(f'RASE: {rase(gt, out)}')
print(f'VIF-P: {vifp(gt, out)}')

