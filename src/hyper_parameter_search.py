import neps
import time
import os
import torch
from cnn import get_conv_model
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import logging
import shutil
from torch.utils.data import DataLoader
import torch.nn as nn
from data_augmentations import resize_to_64x64


def get_pipeline_space() -> dict:
    """ Define a hyperparameter search-space.

        hyperparameters:
          num_filters_1   from    4 to   32 (int, log)
          num_filters_2   from    4 to   32 (int, log)
          num_filters_3   from    4 to   32 (int, log)
          lr              from 1e-6 to 1e-1 (float, log)
          optimizer            Adam or  SGD (categorical, order is important for tests)
          epochs          from 1 to 9 (fidelity parameter)

        Returns:
            Pipeline space dictionary

        Note:
            Please name the hyperparameters and order them as given above (needed for testing)
        """

    pipeline_space = dict(
        num_filters_1=neps.IntegerParameter(lower=3, upper=32, log=True),
        num_filters_2=neps.IntegerParameter(lower=3, upper=32, log=True),
        num_filters_3=neps.IntegerParameter(lower=3, upper=32, log=True),
        lr=neps.FloatParameter(lower=1e-6, upper=1e-1, log=True),
        optimizer=neps.CategoricalParameter(choices=["Adam", "SGD"]),
        epochs=neps.IntegerParameter(lower=5, upper=20, is_fidelity=True)
    )
    return pipeline_space


def training(model, optimizer, criterion, train_loader, validation_loader) -> float:
    """
    Function that trains the model for one epoch and evaluates the model on the validation set. Used by the searcher.

    Args:
        model (nn.Module): Model to be trained.
        optimizer (torch.nn.optim): Optimizer used to train the weights (depends on the pipeline space).
        criterion (nn.modules.loss) : Loss function to use.
        train_loader (torch.utils.Dataloader): Data loader containing the training data.
        validation_loader (torch.utils.Dataloader): Data loader containing the validation data.

    Returns:
        (float) validation error for the epoch.
    """
    for i, data in enumerate(train_loader):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    correct = 0
    images_counter = 0
    for data in validation_loader:
        image, label = data
        output = model(image)
        output = output.argmax(axis=1)
        correct += (output == label).sum()
        images_counter += label.shape[0]
    validation_accuracy = int(correct) / images_counter
    return 1 - validation_accuracy


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    """Evaluate the performance of a given model.

      Args:
          model: PyTorch model
          data_loader : Validation data structe

      Returns:
          accuracy

      Note:

    """
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    accuracy = correct / len(data_loader.sampler)
    return accuracy



