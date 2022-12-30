import time
import neps
import torch

from lib.conv_model import get_conv_model
from lib.utilities import set_seed, evaluate_accuracy
from lib.train_epoch import training


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
        num_filters_1=neps.IntegerParameter(lower=4, upper=32, log=True),
        num_filters_2=neps.IntegerParameter(lower=4, upper=32, log=True),
        num_filters_3=neps.IntegerParameter(lower=4, upper=32, log=True),
        lr=neps.FloatParameter(lower=1e-6, upper=1e-1, log=True),
        optimizer=neps.CategoricalParameter(choices=["Adam", "SGD"]),
        epochs=neps.IntegerParameter(lower=1, upper=9, is_fidelity=True)
    )
    return pipeline_space


def run_pipeline(pipeline_directory, previous_pipeline_directory, num_filters_1, num_filters_2, num_filters_3,
                 lr, optimizer, epochs=9) -> dict:
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

    train_loader, validation_loader, test_loader = load_mnist_minibatched(batch_size=32, n_train=4096, n_valid=512)
    # define loss
    criterion = torch.nn.NLLLoss()

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
        val_errors.append(training(model, optimizer, criterion, train_loader, validation_loader))
    train_accuracy = evaluate_accuracy(model, train_loader)
    test_accuracy = evaluate_accuracy(model, test_loader)

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
            "test_accuracy": test_accuracy,
            "val_errors": val_errors,
            "train_time": end - start,
            "cost": epochs - epochs_previously_spent
        },
        "cost": epochs - epochs_previously_spent
    }
