from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, RandomResizedCrop, ToTensor, Normalize, \
    TenCrop, FiveCrop, PILToTensor, Lambda
from torchvision.transforms.functional import equalize
import torch


def get_random_crops(img, crop_size, num_crops):
    crops = [transforms.RandomCrop(crop_size)(img) for _ in range(num_crops)]
    return crops


resize_to_64x64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

resize_to_128x128 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.GaussianBlur((3, 3)),
    # transforms.RandomHorizontalFlip(1),
    # transforms.Grayscale(3),
    # transforms.Normalize()
    # transforms.RandomInvert(),
    transforms.RandomAffine(0.4),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

resize_to_224x224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

trivial_augment = transforms.Compose([
    transforms.TrivialAugmentWide(num_magnitude_bins=42),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data_set_3 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_magnitude_bins=42),
    transforms.ToTensor()
])

data_set_4 = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.9, contrast=0.1),
])

nothing = transforms.Compose([
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

crop = transforms.Compose([transforms.RandomCrop((224, 224)),
                           ToTensor()])
crop_reducing = transforms.Compose([transforms.Lambda(lambda img: img.resize((img.size[0]//2, img.size[1]//2))),
                            transforms.CenterCrop((256, 256)),
                           ToTensor()])

test2 = transforms.Compose([transforms.RandomCrop((256, 256)),
                                            ToTensor()])

stack_crop = transforms.Compose([FiveCrop(224),  # this is a list of PIL Images
            Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops]))
                                 ])
