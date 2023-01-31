import os

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from src.cnn import *
from src.data_augmentations import *
from torch_lr_finder import LRFinder
# data


def lr_finder(data_dir, model, batch_size, data_augmentation, start_lr):
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentation)
    # train_data1 = ImageFolder(os.path.join(data_dir, 'train'), transform=data_set_2)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=data_augmentation)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=data_augmentation)
    train_loader = DataLoader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                      batch_size=batch_size,
                                      shuffle=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=False)
    # settings
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)

    # lr_ finder
    lr_finder = LRFinder(model, optimizer, criterion, device=torch.device("cpu"))
    lr_finder.range_test(train_loader, val_loader=val_loader, start_lr=start_lr, end_lr=0.04, num_iter=200,
                         step_mode="exp")
    lr_finder.plot(log_lr=False)
    lr_finder.reset()


if __name__=="__main__":
    data_dir = default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      '..', 'dataset')
    data_augmentation = resize_to_224x224
    batch_size = 32
    model = ModelZeroSix()
    lower_bound_lr = 10e-7
    lr_finder(data_dir, model, batch_size, data_augmentation, lower_bound_lr)

