""" File with CNN models. Add your custom CNN model here. """
import numpy
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import List


class SampleModel(nn.Module):
    """
    A sample PyTorch CNN model
    """
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        # The input features for the linear layer depends on the size of the input to the convolutional layer
        # So if you resize your image in data augmentations, you'll have to tweak this too.
        self.fc1 = nn.Linear(in_features=4500, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class ModelZeroOne(nn.Module):
    """first model"""

    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(ModelZeroOne, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3)),
            nn.LeakyReLU(0.02),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3)),
            nn.LeakyReLU(0.025),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x


class ModelZeroTwo(nn.Module):
    """first model"""

    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(ModelZeroTwo, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(50176, 10),
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x


class ModelZeroThree(nn.Module):
    """first model"""
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(ModelZeroThree, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=42, kernel_size=(3, 3)),
            nn.BatchNorm2d(42),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=42, out_channels=54, kernel_size=(3, 3)),
            nn.BatchNorm2d(54),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=54, out_channels=54, kernel_size=(3, 3)),
            nn.BatchNorm2d(54),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
            nn.Linear(2646, 10)
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def predict(self, x):
        pred = self.forward(x)
        _, prediction = torch.max(pred.data, 1)
        return prediction


class ModelZeroFour(nn.Module):
    """first model"""
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(ModelZeroFour, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.AvgPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(2304, 10)
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def predict(self, x):
        pred = self.forward(x)
        _, prediction = torch.max(pred.data, 1)
        return prediction


class ModelZeroFive(nn.Module):
    """first model"""
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(ModelZeroFive, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=42, kernel_size=(3, 3)),
            nn.BatchNorm2d(42),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=42, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2304, 10)
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def predict(self, x):
        pred = self.forward(x)
        _, prediction = torch.max(pred.data, 1)
        return prediction


class ModelZeroSix(nn.Module):
    """first model"""
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(ModelZeroSix, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=42, kernel_size=(3, 3)),
            nn.BatchNorm2d(42),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(3, 3)),
            nn.BatchNorm2d(42),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=62, kernel_size=(3, 3)),
            nn.BatchNorm2d(62),
            nn.ReLU(),
            nn.Conv2d(in_channels=62, out_channels=62, kernel_size=(3, 3)),
            nn.BatchNorm2d(62),
            nn.AdaptiveMaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(248, 10),
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def predict(self, x):
        pred = self.forward(x)
        _, prediction = torch.argmax(pred.data, -1)
        print(prediction)
        return prediction


class ModelZeroSeven(nn.Module):
    """first model"""
    def __init__(self, input_shape=(3, 64, 64), num_classes=10):
        super(ModelZeroSeven, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=42, kernel_size=(3, 3)),
            nn.BatchNorm2d(42),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=42, kernel_size=(3, 3)),
            nn.BatchNorm2d(42),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=42, out_channels=60, kernel_size=(3, 3)),
            nn.BatchNorm2d(60),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 3)),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.42),
            nn.Linear(540, 10),
        )

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def predict(self, x):
        pred = self.forward(x)
        prediction = torch.argmax(pred.data, 1)
        return prediction


def get_conv_model(num_filters_per_layer: List[int], conv_kernel_size) -> nn.Module:
    """
    Builds a deep convolutional model with varying number of convolutional
    layers (and # filters per layer) for MNIST input using pytorch.

    Args:
        num_filters_per_layer (list): List specifying the number of filters for each convolutional layer

    Returns:
        convolutional model with desired architecture

    Note:
        for each element in num_filters_per_layer:
            convolution (conv_kernel_size, num_filters, stride=1, padding=0) (use nn.Conv2d(..))
            relu (use nn.ReLU())
            max_pool(pool_kernel_size) (use nn.MaxPool2d(..))

        flatten layer (already given below)
        linear layer
        log softmax as final activation
    """
    assert len(num_filters_per_layer) > 0, "len(num_filters_per_layer) should be greater than 0"
    pool_kernel_size = 2

    # OrderedDict is used to keep track of the order of the layers
    layers = OrderedDict()

    in_channels = 3
    counter = 1
    input_W = 64
    input_H = 64
    for filters in num_filters_per_layer:
        layers["conv"+str(counter)] = nn.Conv2d(in_channels=in_channels, out_channels=filters, stride=1, padding=0,
                                                kernel_size=conv_kernel_size)
        # layers["dropout"+str(counter)] = nn.Dropout()
        layers["relu"+str(counter)] = nn.ReLU()
        layers["max_pool"+str(counter)] = nn.MaxPool2d(kernel_size=pool_kernel_size)
        in_channels = filters
        # output size conv
        output_W = (input_W - conv_kernel_size) + 1
        output_H = (input_H - conv_kernel_size) + 1
        output_K = filters
        # output size max pool
        output_W = (output_W - pool_kernel_size) // pool_kernel_size + 1
        output_H = (output_H - pool_kernel_size) // pool_kernel_size + 1
        input_W = output_W
        input_H = output_H
        counter += 1
    # output flatten
    conv_output_size = output_H * output_W * output_K
    layers['flatten'] = nn.Flatten()
    layers['linear'] = nn.Linear(conv_output_size, 10)
    # Log Softmax if LLLoss is used
    # layers['log_softmax'] = nn.LogSoftmax(dim=1)

    return nn.Sequential(layers)


