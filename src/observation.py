import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from cnn import ModelZeroOne
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder
import data_augmentations
import matplotlib.pyplot as plt
import numpy as np


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def load_data(path, data_augmentations):
    """load data"""
    if data_augmentations is None:
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError
    data_dir = path
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    return train_data


def monitor_training(tb, train_loss, train_score, val_score, epoch):
    tb.add_scalar("Train Loss", train_loss, epoch)
    # tb.add_scalar("Correct", total_correct, epoch)
    tb.add_scalar("Train Accuracy", train_score, epoch)
    tb.add_scalar("Val Accuracy", val_score, epoch)
    # tb.add_histogram("conv1.bias", model.conv1.bias, epoch)
    # tb.add_histogram("conv1.weight", model.conv1.weight, epoch)
    # tb.add_histogram("conv2.bias", model.conv2.bias, epoch)
    # tb.add_histogram("conv2.weight", model.conv2.weight, epoch)


def matplotlib_imshow(img, one_channel=False):
    """plot images"""
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# ToDo: Try to get projector for visualizing embeddings

"""
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def projector(images, labels, tb):
    # select random images and their target indices
    # images, labels = select_n_random(trainset.data, trainset.targets)

    # get the class labels for each image
    # class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 64 * 64)
    tb.add_embedding(features,
                        metadata=labels,
                        label_img=images.unsqueeze(1))

"""
if __name__ == '__main__':
    tb = SummaryWriter('runs/try^')
    model = ModelZeroOne()
    transforms.ToTensor()
    train_set = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
                          [data_augmentations.resize_and_colour_jitter])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    images, labels = next(iter(train_loader))
    # visualize embeddings
    # projector(images, labels, tb)
    # create grid of images
    grid = torchvision.utils.make_grid(images)
    # show images
    # matplotlib_imshow(grid, one_channel=True)
    tb.add_image("images", grid)
    # inspect graph
    tb.add_graph(model, images)
    tb.close()

