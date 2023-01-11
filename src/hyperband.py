import neps
import time
import os
import torch
from cnn import get_conv_model
from torchvision.datasets import ImageFolder
import logging
import shutil
from torch.utils.data import DataLoader
from data_augmentations import resize_to_64x64
from hyper_parameter_search import evaluate_accuracy, training


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


def run_pipeline(pipeline_directory, previous_pipeline_directory, num_filters_1, num_filters_2, num_filters_3,
                 lr, optimizer, batch_size=32, epochs=9) -> dict:
    """ Evaluate a function with the given parameters and return a loss.
        NePS tries to minimize the returned loss. In our case the function is
        the training and validation of a model, the budget is the number of
        epochs and the val_error(1-validation_accuracy) which we use as our loss.

    Args:
        num_filters_1: Number of filters for the first conv layer
        num_filters_2: Number of filters for the second conv layer
        num_filters_3: Number of filters for the third conv layer
        lr: Learning rate passed to the optimizer to train the neural network
        optimizer: Optimizer used to train the neural network ("Adam" or "SGD")
        epochs: Number of epochs to train the model(if not set by NePS it is by default 9)
        batch_size: number of images in one batch
        pipeline_directory: Directory where the trained model will be saved
        previous_pipeline_directory: Directory containing stored model of previous HyperBand iteration

    Returns:
        Dictionary of loss and info_dict which contains additional information about the runs.

    Note:
        Please notice that the optimizer is determined by the pipeline space.
    """
    start = time.time()

    # retrieve the number of filters and create the model
    model = get_conv_model([num_filters_1, num_filters_2, num_filters_3])
    checkpoint_name = "checkpoint.pth"
    start_epoch = 0

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'dataset')

    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=resize_to_64x64)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=resize_to_64x64)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_data,
                            batch_size=batch_size,
                            shuffle=False)

    # define loss
    criterion = torch.nn.NLLLoss()
    # criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # We make use of checkpointing to resume training models on higher fidelities
    if previous_pipeline_directory is not None:
        # Read in state of the model after the previous fidelity rung
        checkpoint = torch.load(previous_pipeline_directory / checkpoint_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epochs_previously_spent = checkpoint["epoch"]
    else:
        epochs_previously_spent = 0

    start_epoch += epochs_previously_spent

    val_errors = list()

    for epoch in range(start_epoch, epochs):
        print("  Epoch {} / {} ...".format(epoch + 1, epochs).ljust(2))
        # Call the training function, get the validation errors and append them to val errors
        val_errors.append(training(model, optimizer, criterion, train_loader, val_loader))
    train_accuracy = evaluate_accuracy(model, train_loader)
    val_accuracy = evaluate_accuracy(model, val_loader)

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        pipeline_directory / checkpoint_name,
    )
    end = time.time()
    print("  Epoch {} / {} Val Error: {}".format(epochs, epochs,
                                                 val_errors[-1]).ljust(2))
    return {
        "loss": val_errors[-1],
        "info_dict": {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "val_errors": val_errors,
            "train_time": end - start,
            "cost": epochs - epochs_previously_spent
        },
        "cost": epochs - epochs_previously_spent
    }

def hyperband():
    pipeline_space = get_pipeline_space()

    logging.basicConfig(level=logging.INFO)
    if os.path.exists("results/hyperband"):
        shutil.rmtree("results/hyperband")
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="results/hyperband",
        max_cost_total=10*50,
        searcher="hyperband"

    )
    previous_results, pending_configs = neps.status(
        "results/hyperband"
    )
    neps.plot("results/hyperband")


if __name__ == "__main__":
    hyperband()

