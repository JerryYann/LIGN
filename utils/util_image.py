import os
import math
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import cv2 as cv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
'''
# --------------------------------------------
# img processing
# --------------------------------------------

'''

def data_normal(data):
    d_min = data.min()
    d_max = data.max()
    dst = d_max - d_min
    norm_data = torch.sub(data,d_min).true_divide(dst)
    norm_data = (norm_data-0.5).true_divide(0.5)
    return norm_data


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


# layer the input
def disassemble(image, channels_num, l):
    k = 256 // l
    h, w = image.shape[0], image.shape[1]
    img_layers = np.zeros((h, w, l*channels_num))
    for c in range(channels_num):
        count = 0
        for i in range(l):
            img_temp = image.copy()[:, :, c]
            if i == 0:
                index = (img_temp < i+k+count)
            elif i == k-1:
                index = (i+count <= img_temp)
            else:
                index = (i+count <= img_temp) * (img_temp < i+k+count)
            img_temp[~index] = 0
            img_layers[:, :, i+l*c] = img_temp
            count += k-1
    return img_layers



def noiseGauss(img, sigma, t=False):
    if t:
        np.random.seed(0)
    temp_img = np.float32(np.copy(img))
    h = temp_img.shape[0]
    w = temp_img.shape[1]
    noise = np.random.randn(h, w) * sigma
    noisy_img = np.zeros(temp_img.shape, np.float32)

    if img.ndim == 2:
        img = np.expand_dims(img,2)
    if img.shape[2] == 1:
        noise = np.expand_dims(noise, 2)
        noisy_img = temp_img + noise
    else:
        noisy_img = temp_img + np.random.normal(0, sigma, img.shape)

    return noisy_img


#
# scaling
#
def scaling(img):
    ch = img.shape[0]
    temp = torch.zeros(img.shape)
    for c in range(ch):
        img_c = img.clone()[c]
        img_min = torch.min(img_c)
        temp[c] = 255 * (img_c - img_min) / torch.max(img_c - img_min)
    return temp


def layer_con(x):
    h,w = x.shape[0],x.shape[1]
    temp = np.zeros((h, w, 1))
    for c in range(16):
        temp[:, :, 0] += x[:, :, c]

    return temp


'''
# --------------------------------------------
# get image paths
# --------------------------------------------
'''


def get_image_paths(dataroot):
    paths = None
    if dataroot is not None:
        paths = sorted(_get_paths_from_images(dataroot))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


IMG_FORMAT = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(suffix) for suffix in IMG_FORMAT)


'''
# --------------------------------------------
# read image from path
# opencv read BGR numpy image (utf8)
# --------------------------------------------
'''

def imread_uint(path, n_channels=3):
    if n_channels == 1:
        image = cv.imread(path, 0)
        image = np.expand_dims(image, axis=2)
    elif n_channels == 3:
        image = cv.imread(path, cv.IMREAD_UNCHANGED)
        if image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        else:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv.imwrite(img_path, img)


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def augment_img_tensor4(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])


def uint2single(image):

    return np.float32(image)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())


def uint2tensor3(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()


def single2tensor3(image):
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()


def tensor2uint(image):

    image = image.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    return np.uint8((image*255.0).round())


def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, border=0):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

