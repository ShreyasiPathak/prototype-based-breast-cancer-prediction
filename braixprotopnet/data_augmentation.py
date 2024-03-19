import torch
from torch import Tensor
from typing import Tuple, Dict
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class MyHorizontalFlip:
    """Flip horizontally."""

    def __init__(self):
        pass

    def __call__(self, x, breast_side):
        #if breast_side=='L':
        #    return TF.hflip(x) #L->R
        if breast_side=='R':
            return TF.hflip(x) #R->L (following GMIC)
        else:
            return x

# function copied from https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            #"ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 15.0, num_bins), True), 
            #"Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }

class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def data_transform_gmic(img_size):    
    mean_val = (0.485, 0.456, 0.406)
    std_dev = (0.229, 0.224, 0.225)

    transform_aug = transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=(15),translate=(0.1,0.1),scale=(0.8,1.6),shear=(25)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std=0.005),
        transforms.Normalize(mean=mean_val, std=std_dev)
    ])
    
    transform_no_aug = transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_val, std=std_dev)
    ])

    transform_no_aug_no_norm = transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ]) 
    
    return transform_aug, transform_no_aug, transform_no_aug_no_norm

def data_transform_pipnet(img_size: list): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    
    transform_aug = transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(img_size[0], img_size[1]), scale=(0.95, 1.)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ]) 

    transform_no_aug = transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        normalize
    ])

    transform_no_aug_no_norm = transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        transforms.Grayscale(3),
        transforms.ToTensor()
    ]) 

    return transform_aug, transform_no_aug, transform_no_aug_no_norm

def cbisddsm_transforms(img_size: list): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size[0], img_size[1])),
                            transforms.Grayscale(3),
                            transforms.ToTensor(),
                            normalize
                        ])
    
    transform1 = transforms.Compose([
                    transforms.Resize(size=(img_size[0], img_size[1])), 
                    TrivialAugmentWideNoColor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(size=(img_size[0], img_size[1]), scale=(0.95, 1.))
                ])

    transform2 = transforms.Compose([
                    TrivialAugmentWideNoColor(),
                    transforms.Grayscale(3),
                    transforms.ToTensor(),
                    normalize
                ])        

    return transform1, transform2, transform_no_augment