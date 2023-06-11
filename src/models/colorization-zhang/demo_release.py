
import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs, tens_l_orig_col, tens_l_rs_col) = preprocess_img(img, HW=(256,256))

#create random mask
# a = np.full(()), False)
# a[:1000] = True
# np.random.shuffle(a)
# print(a)


if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map

mask = np.zeros((256*256,))
mask[:int(0.01 * 256 * 256)] = 1
np.random.shuffle(mask)
mask = np.reshape(mask, (1,1,256,256))
print(mask)
mask = torch.from_numpy(mask).float()
# resize and concatenate to original L channel
print(mask.shape)
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
img_hints = postprocess_tens(tens_l_orig, torch.mul(tens_l_rs_col,mask))

out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

out_img_with_hints_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs,input_B = tens_l_rs_col, mask_B = mask).cpu())


plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)
plt.imsave('%s_siggraph17_hints_in.png'%opt.save_prefix, out_img_with_hints_siggraph17)
plt.imsave('%s_siggraph17_hints_out.png'%opt.save_prefix, out_img_with_hints_siggraph17)


plt.figure(figsize=(12,8))
plt.subplot(2,3,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(img_hints)
plt.title('Input with hints')
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(out_img_with_hints_siggraph17)
plt.title('Output with hints (SIGGRAPH 17)')
plt.axis('off')

plt.show()