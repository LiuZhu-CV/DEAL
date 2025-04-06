import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class NG(nn.Module):
    def __init__(self, ng_types: list) -> None:
        super().__init__()

        self.type_num = len(ng_types)
        self.feature_extractor = FeatureExtractor(self.type_num)
        self.ng_types = ng_types

    def forward(self, img_input):
        batch_size = len(img_input)
        weights_type = self.feature_extractor(img_input)
        weights_type = weights_type.view(batch_size, 5, self.type_num)

        img_batch = []
        for i, weight_type in enumerate(weights_type):
            single_img = img_input[i]
            weight_type = torch.softmax(weight_type, 1)
            seed = random.random()

            for w_t in weight_type:
                single_img = sum(w_t[ind] * _NAME_TO_NG_FUNC[ng_type](single_img, seed) for ng_type, ind in zip(self.ng_types, range(self.type_num)))
            img_batch.append(single_img.unsqueeze(0))

        img_out = torch.cat(img_batch, dim=0)
        return img_out

class FeatureExtractor(nn.Module):
    def __init__(self, type_num):
        super().__init__()

        self.encoder_type = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2), nn.InstanceNorm2d(16, affine=True),
            *conv_downsample(16, 32, normalization=True),
            *conv_downsample(32, 64, normalization=True)
        )

        self.decoder_type = nn.Sequential(
            *conv_downsample(64, 128, normalization=True),
            *conv_downsample(128, 128), nn.Dropout(p=0.5),
            nn.Conv2d(128, type_num * 5, 8, padding=0),
        )

    def forward(self, img_input):
        fea = self.encoder_type(img_input)
        weights_type = self.decoder_type(fea)
        return weights_type

def conv_downsample(in_filters, out_filters, normalization=False):
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))

    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers

def less_contrast1(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_contrast(video, 0.9)

def less_contrast2(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_contrast(video, 0.7)

def less_contrast3(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_contrast(video, 0.5)

def less_saturation1(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_saturation(video, 0.9)

def less_saturation2(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_saturation(video, 0.7)

def less_saturation3(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_saturation(video, 0.5)

def smoother1(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_sharpness(video, 0.9)

def smoother2(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_sharpness(video, 0.7)

def smoother3(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_sharpness(video, 0.5)

def smoother4(video: torch.Tensor, *args) -> torch.Tensor:
    return torchvision.transforms.functional.adjust_sharpness(video, 0.3)

def gaussian_blur(image_tensor, *args):
    blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(1.0, 1.5))
    blurred_tensor = blur(image_tensor)
    return blurred_tensor

def down_up_sample1(image_tensor, *args):
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    upsample = nn.Upsample(size=image_tensor.shape[-2:], mode='bilinear')
    x_downsampled = pool(image_tensor)
    x_upsampled = upsample(x_downsampled.unsqueeze(0))
    return x_upsampled.squeeze()

def down_up_sample2(image_tensor, *args):
    pool = nn.MaxPool2d(kernel_size=3, stride=3)
    upsample = nn.Upsample(size=image_tensor.shape[-2:], mode='bilinear')
    x_downsampled = pool(image_tensor)
    x_upsampled = upsample(x_downsampled.unsqueeze(0))
    return x_upsampled.squeeze()

def down_up_sample3(image_tensor, *args):
    pool = nn.MaxPool2d(kernel_size=4, stride=4)
    upsample = nn.Upsample(size=image_tensor.shape[-2:], mode='bilinear')
    x_downsampled = pool(image_tensor)
    x_upsampled = upsample(x_downsampled.unsqueeze(0))
    return x_upsampled.squeeze()

def posterize(video: torch.Tensor, *args):
    factor = 5
    if factor >= 8:
        return video
    if video.dtype != torch.uint8:
        video_type = video.dtype
        video = (video * 255).to(torch.uint8)
        return (torchvision.transforms.functional.posterize(video, factor) / 255).to(video_type)
    return torchvision.transforms.functional.posterize(video, factor)

def add_new_striped_noise(image, *args):
    intensity = 0.02
    nv = intensity * 25.5

    beta = nv * random.random()
    c, h, w = image.size(0), image.size(1), image.size(2)
    g_col = torch.normal(0, beta, (w,)).cuda()
    g_noise = g_col.repeat(1, h)
    noise_img = image + g_noise.view(1, h, w)

    return noise_img

_NAME_TO_NG_FUNC = \
    {
    'Smoother1': smoother1,
    'Smoother2': smoother2,
    'Smoother3': smoother3,
    'Smoother4': smoother4,

    'LessSaturation1': less_saturation1,
    'LessSaturation2': less_saturation2,
    'LessSaturation3': less_saturation3,

    'LessContrast1': less_contrast1,
    'LessContrast2': less_contrast2,
    'LessContrast3': less_contrast3,

    "Posterize": posterize,

    'GaussianBlur': gaussian_blur,

    'DownUpSample1': down_up_sample1,
    'DownUpSample2': down_up_sample2,
    'DownUpSample3': down_up_sample3,

    'add_new_striped_noise': add_new_striped_noise
    }