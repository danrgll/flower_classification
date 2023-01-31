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
from data_augmentations import resize_to_64x64, crop
from hyper_parameter_search import training, evaluate_accuracy, get_conv_model
from cnn import *


def get_pipeline_space() -> dict:
    """ Define a hyperparameter search-space.

        hyperparameters:
          num_filters_1   from    4 to   32 (int, log)
          num_filters_2   from    4 to   32 (int, log)
          num_filters_3   from    4 to   32 (int, log)
          lr              from 1e-6 to 1e-1 (float, log)
          optimizer            Adam or  SGD (categorical)
          epochs          from 1 to 9 (fidelity parameter)

        Returns:
            Pipeline space dictionary

        Note:
            Please name the hyperparameters and order them as given above (needed for testing)
        """
    pipeline_space = dict(
        # num_filters_1=neps.IntegerParameter(lower=32, upper=32, log=True),
        # num_filters_2=neps.IntegerParameter(lower=64, upper=64, log=True),
        # num_filters_3=neps.IntegerParameter(lower=64, upper=64, log=True),
        kernel_size=neps.CategoricalParameter(choices=[3]),
        lr=neps.FloatParameter(lower=1e-7, upper=1e-1, log=True),
        optimizer=neps.CategoricalParameter(choices=["Adam"]),
    )
    return pipeline_space


def run_pipeline(lr, optimizer, kernel_size, num_filters_1=32, num_filters_2=64, num_filters_3=64, batch_size=32,
                 epochs=30):
    """ Train and evaluate a model given some configuration

    Args:
        lr: Learning rate passed to the optimizer to train the neural network
        num_filters_1: Number of filters for the first conv layer
        num_filters_2: Number of filters for the second conv layer

    Returns:
        Dictionary of loss and info_dict which contains additional information about the runs.

    Note:
        Keep in mind that we want to minimize the error (not loss function of
        the training procedure), for that we use (1-val_accuracy) as our val_error.

    """
    start = time.time()

    # load data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'dataset')
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=crop)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=crop)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=False)
    # define loss
    criterion = torch.nn.CrossEntropyLoss()

    # create model
    num_filters_per_layer = [num_filters_1, num_filters_2, num_filters_3]

    # model = get_conv_model(num_filters_per_layer, kernel_size)
    model = ModelZeroThree()
    # choose optimizer
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    val_errors = []
    i = 0
    while i < epochs:
        val_errors.append(training(model, optimizer, criterion, train_loader, val_loader))
        i += 1
    train_accuracy = evaluate_accuracy(model, train_loader)
    val_accuracy = evaluate_accuracy(model, val_loader)
    end = time.time()
    print("  Epoch {} / {} Val Error: {}".format(epochs, epochs, val_errors[-1]).ljust(2))

    return {
        "loss": val_errors[-1],
        "info_dict": {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_time": end - start,
            "val_errors": val_errors,
            "cost": epochs
        }
    }


def bayesian():
    logging.basicConfig(level=logging.INFO)

    pipeline_space = get_pipeline_space()
    if os.path.exists("results/bayesian_optimization"):
        shutil.rmtree("results/bayesian_optimization")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/bayesian_optimization",
        max_evaluations_total=20,
        searcher="bayesian_optimization"
    )
    previous_results, pending_configs = neps.status(
        "results/bayesian_optimization"
    )
    # neps.plot("results/bayesian_optimization")


if __name__ == "__main__":
    bayesian()
