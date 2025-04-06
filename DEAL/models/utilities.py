import os
import cv2
import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from math import exp

class pre_process(data.Dataset):
    def __init__(self, dataroot, imlist_pth, transform=None, resize_to=None):
        self.dataroot = dataroot
        self.transform = transform
        self.resize_to = resize_to
        self.imlist = self.flist_reader(imlist_pth)

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        im_name = self.imlist[index]
        im_input = self.sample_loader(im_name)

        if self.resize_to:
            im_input = cv2.resize(im_input, self.resize_to)

        if self.transform:
            im_input = self.transform(im_input)

        return im_input

    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

    def sample_loader(self, im_name):
        im_pth = os.path.join(self.dataroot, im_name)
        im_input = Image.open(im_pth).convert('RGB')
        return im_input

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)