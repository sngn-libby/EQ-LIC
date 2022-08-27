# MS
x = [0.13, 0.19, 0.28, 0.5, 0.73, 0.9144, 1.2342, 1.6191]
y = [27.76, 29.08, 30.42, 32.37, 34.24, 36.91, 38.84, 40.63]

# Joint
# x = [0.1105, 0.187, 0.2877, 0.4324, 0.6388, 0.8853, 1.2003, 1.5873]
# y = [28.09, 29.65, 31.36, 33.09, 35.09, 36.99, 38.93, 40.64]

# GMM
# x = [0.1196, 0.1838, 0.271, 0.4174, 0.5945, 0.8057]
# y = [28.58, 29.97, 31.34, 33.39, 35.12, 36.7]

bpp = 0.58
psnr = 32.47

if psnr < y[0]:
    m = (x[1] - x[0]) / (y[1] - y[0])
    fp_bpp = m * psnr - m * y[0] + x[0]
    print(f'bpp = {fp_bpp:.4f}')
    print(f'bpp loss = {bpp / fp_bpp * 100 - 100:.2f}%')

for i in range(len(y)):
    if y[i - 1] < psnr < y[i]:
        m = (x[i] - x[i - 1]) / (y[i] - y[i - 1])
        fp_bpp = m * psnr - m * y[i] + x[i]
        print(f'bpp = {fp_bpp:.4f}')
        print(f'bpp loss = {bpp/fp_bpp*100 - 100:.2f}%')

if bpp > x[-1]:
    m = (y[-1] - y[-2]) / (x[-1] - x[-2])
    fp_psnr = m * bpp - m * x[-1] + y[-1]
    print(f'psnr = {fp_psnr:.2f}')
    print(f'psnr loss = {fp_psnr - psnr:.2f}')

for i in range(len(x)):
    if x[i - 1] < bpp < x[i]:
        m = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        fp_psnr = m * bpp - m * x[i] + y[i]
        print(f'psnr = {fp_psnr:.2f}')
        print(f'psnr loss = {fp_psnr - psnr:.2f}')
