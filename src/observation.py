import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from cnn import ModelZeroOne
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def load_data(path):
    data_dir = path
    train_data = ImageFolder(os.path.join(data_dir, 'train'))
    return train_data


if __name__ == '__main__':
    tb = SummaryWriter()
    model = ModelZeroOne()
    transforms.ToTensor()
    train_set = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    tb.add_image("images", grid)
    tb.add_graph(model, images)
    tb.close()

