import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from src.cnn import ModelZeroOne
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder
import src.data_augmentations as data_augmentations
import matplotlib.pyplot as plt
import numpy as np
from src.cnn import *
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, rotate, gaussian_blur
from torchvision.transforms.functional import equalize


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


def monitor_training(tb, train_loss, train_score,epoch, model, val_loss=None, val_score=None):
    tb.add_scalar("Train Loss", train_loss, epoch)
    if val_loss is not None:
        tb.add_scalar("Validation_loss", val_loss, epoch)
        tb.add_scalars("Train_Validation_Loss", {"train loss": train_loss,
                                                "val_loss": val_loss}, epoch)
        tb.add_scalar("Train Accuracy", train_score, epoch)
        tb.add_scalar("Val Accuracy", val_score, epoch)
        tb.add_scalars("Train_Val_accuracy", {"train_acc": train_score, "val_acc": val_score}, epoch)
        counter = 1
        # monitor weights and biases and gradients of them
        for layer in model.model:
            if hasattr(layer, 'weight'):
                tb.add_histogram("layer_weight_histo" + str(counter), layer.weight, epoch)
                tb.add_histogram("layer_weight_grad__histo" + str(counter), layer.weight.grad, epoch)
            if hasattr(layer, 'bias'):
                tb.add_histogram("layer_bias_histo" + str(counter) + "bias", layer.bias, epoch)
                tb.add_histogram("layer_bias_grad_histo" + str(counter) + "bias", layer.bias.grad, epoch)
            counter += 1


def matplotlib_imshow(img, label, counter, prediction=None, one_channel=False, dir_name="predictions"):
    """plot images"""
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if prediction is None:
        file_name = "img" + str(counter) + "_true" + str(int(label)) + ".png"
        file_path = os.path.join(dir_name, file_name)
    else:
        file_name = "img" + str(counter) + "_true" + str(int(label)) + "_pred" + str(int(prediction)) + ".png"
        file_path = os.path.join(dir_name, file_name)
    if one_channel:
        plt.imsave(file_path, npimg, cmap="Greys")
    else:
        plt.imsave(file_path, np.transpose(npimg, (1, 2, 0)))


def visualize_classification(model, model_path, data_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'dataset'), num_predictions=200, folder_name="predictions"):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=data_augmentations.crop)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=num_predictions, shuffle=False)
    images, labels = next(iter(val_loader))
    counter = 0
    for image, label in zip(images, labels):
        # add batchsize dimension
        batch_image = image.unsqueeze(0)
        print(batch_image)
        with torch.no_grad():
            predicted_class = model.predict(batch_image)
        if int(label.item()) != int(predicted_class.item()):
            matplotlib_imshow(image, label, counter, predicted_class)
        counter += 1


def tb_images():
    tb = SummaryWriter('runs/imagesss')
    model = ModelZeroSeven()
    # train_set = load_data(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'),
    #                      [data_augmentations.data_set_4])
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'dataset')
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations.crop)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    images, labels = next(iter(train_loader))
    # create grid of images
    grid = torchvision.utils.make_grid(images)
    # show images
    # matplotlib_imshow(grid, one_channel=True)
    tb.add_image(str(labels), grid)
    # inspect graph
    tb.add_graph(model, images)


if __name__ == '__main__':
    # tb_images()
    model = ModelZeroSeven()
    visualize_classification(model, "../models/fast_model")

