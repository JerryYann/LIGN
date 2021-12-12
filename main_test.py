import os.path
import logging

import numpy as np
from collections import OrderedDict
import torch

from utils import util_logger
from utils import util_image as util


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 50             # noise level for noisy image
    model_name = 'gray-50'

    if 'color' in model_name:
        n_channels = 3
    else:
        n_channels = 1

    model_pool = 'model_zoo'  # fixed
    testsets = r'E:\aYJR\datasets\BSD500\val'     # fixed
    results = 'results'       # fixed
    result_name = '_' + model_name     # fixed

    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets)               # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)


    logger_name = result_name
    util_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from lign import net


    model = net()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))

        img_H = util.imread_uint(img, n_channels=n_channels)
        H, W, _ = img_H.shape
        if H % 2 ** 4 != 0:
            H -= H % 2 ** 4
        if W % 2 ** 4 != 0:
            W -= W % 2 ** 4
        img_H = img_H[:H, :W, ]

        img_L = np.copy(img_H)

        # --------------------------------
        # add noise
        # --------------------------------
        img_L = util.noiseGauss(img_L, noise_level_img, t=True)
        img_LD = util.disassemble(img_L, n_channels, 8)
        img_LD1 = util.disassemble(img_L, n_channels, 4)

        img_L = img_L / 255.
        img_LD = img_LD / 255.
        img_LD1 = img_LD1 / 255.
        img_H = img_H / 255.


        img_L = util.single2tensor3(img_L)
        img_H = util.single2tensor3(img_H)
        img_LD = util.single2tensor3(img_LD)
        img_LD1 = util.single2tensor3(img_LD1)
        img_L_cat = torch.cat([img_L, img_LD, img_LD1], dim=0).unsqueeze(0)

        img_L_cat = img_L_cat.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        sp, res = model(img_L_cat)

        res = util.tensor2uint(res)
        img_H = util.tensor2uint(img_H)

        if need_H:

            psnr = util.calculate_psnr(res, img_H, border=0)
            ssim = util.calculate_ssim(res, img_H, border=0)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(res, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
