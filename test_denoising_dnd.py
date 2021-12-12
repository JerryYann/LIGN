import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as sio
from lign_real import net
from utils import util_image as util
from skimage import img_as_ubyte
from torch.utils.data import Dataset


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img


class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))

        self.noisy_filenames = [os.path.join(rgb_dir, 'input', x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.noisy_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        noisy = np.float32(load_img(self.noisy_filenames[tar_index]))
        img_LD = util.disassemble(noisy*255., 3, 8)
        img_LD1 = util.disassemble(noisy*255., 3, 4)
        img_LD = img_LD / 255.
        img_LD1 = img_LD1 / 255.

        img_L = util.single2tensor3(noisy)
        img_LD = util.single2tensor3(img_LD)
        img_LD1 = util.single2tensor3(img_LD1)

        noisy = torch.cat([img_L, img_LD, img_LD1], dim=0).unsqueeze(0)

        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        return noisy, noisy_filename


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)


def bundle_submissions_srgb_v1(submission_folder, session):
    out_folder = os.path.join(submission_folder, session)
    try:
        os.mkdir(out_folder)
    except:
        pass
    israw = False
    eval_version = "1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=np.object)
        for bb in range(20):
            filename = '%04d_%d.mat' % (i + 1, bb + 1)
            s = sio.loadmat(os.path.join(submission_folder, filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat' % (i + 1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )

parser = argparse.ArgumentParser(description='RGB denoising evaluation on DND dataset')
parser.add_argument('--input_dir', default=r'input_dir',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/dnd/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./model_zoo/dnd.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=16, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdir(args.result_dir+'matfile')
mkdir(args.result_dir+'png')

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

model_restoration = net()

model_restoration.load_state_dict(torch.load(args.weights), strict=True)

model_restoration.cuda()

model_restoration.eval()


with torch.no_grad():
    psnr_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_noisy = data_test[0].cuda().squeeze(1)
        filenames = data_test[1]
        _, rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.clamp(rgb_restored,0,1)
     
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()


        for batch in range(len(rgb_noisy)):
            denoised_img = img_as_ubyte(rgb_restored[batch])
            save_img(args.result_dir + 'png/'+ filenames[batch][:-4] + '.png', denoised_img)
            save_file = os.path.join(args.result_dir+ 'matfile/', filenames[batch][:-4] +'.mat')
            sio.savemat(save_file, {'Idenoised_crop': np.float32(rgb_restored[batch])})

  
bundle_submissions_srgb_v1(args.result_dir+'matfile/', 'srgb_results_for_server_submission/')
os.system("rm {}".format(args.result_dir+'matfile/*.mat'))
