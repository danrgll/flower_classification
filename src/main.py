import argparse
import logging
import os
import time

import numpy as np
import torch.optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision.datasets import ImageFolder

from monitor import monitor_training
from src.cnn import *
from src.data_augmentations import *
from src.eval.evaluate import eval_fn
from src.training import train_fn
from torch_lr_finder import LRFinder


def main(data_dir,
         torch_model,
         num_epochs=10,
         batch_size=50,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer=torch.optim.Adam,
         data_augmentations=None,
         save_model_str=None,
         load_model_str=None,
         use_all_data_to_train=False,
         exp_name=''):
    """
    Training loop for configurableNet.
    :param torch_model: model that we are training
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during training (torch.optim.Optimizer)
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :param save_model_str: path of saved models (str)
    :param use_all_data_to_train: indicator whether we use all the data for training (bool)
    :param exp_name: experiment name (str)
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if data_augmentations is None:
        data_augmentations = transforms.ToTensor()
    elif isinstance(data_augmentations, list):
        data_augmentations = transforms.Compose(data_augmentations)
    elif not isinstance(data_augmentations, transforms.Compose):
        raise NotImplementedError

    # Load the dataset
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    train_data1 = ImageFolder(os.path.join(data_dir, 'train'), transform=data_set_2)
    # train_data2 = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    # not recommended to use data augmenation here..?
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=data_augmentations)
    test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=data_augmentations)

    channels, img_height, img_width = train_data[0][0].shape

    # image size
    input_shape = (channels, img_height, img_width)

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    score = []

    if use_all_data_to_train:
        train_loader = DataLoader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                  batch_size=batch_size,
                                  shuffle=True)
        logging.warning('Training with all the data (train, val and test).')
    else:
        train_loader = DataLoader(dataset=ConcatDataset([train_data, train_data1]),
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False)

    model = torch_model(input_shape=input_shape,
                        num_classes=len(train_data.classes)).to(device)
    if load_model_str is not None:
        model.load_state_dict(torch.load(load_model_str))
    # instantiate optimizer
    # possible to add weight_decay=1e-5
    optimizer = model_optimizer(model.parameters(), lr=learning_rate)
    # optimizer = model_optimizer(model.parameters(), lr=learning_rate)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=10e-7)


    # Info about the model being trained
    # You can find the number of learnable parameters in the model here
    logging.info('Model being trained:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    # start tensorboard
    tb = SummaryWriter("runs/modelzerosix")



    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = train_fn(model, optimizer, train_criterion, train_loader, device)

        logging.info('Train accuracy: %f', train_score)

        if not use_all_data_to_train:
            test_score, test_loss = eval_fn(model, val_loader, device, train_criterion)
            logging.info('Validation accuracy: %f', test_score)
            score.append(test_score)
        # monitoring training
        monitor_training(tb, train_loss, train_score, test_loss, test_score, epoch, model)
        # scheduler step
        print(scheduler.get_last_lr())
        scheduler.step()

    tb.close()
    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + str(int(time.time())))
        torch.save(model.state_dict(), save_model_str)

    if not use_all_data_to_train:
        logging.info('Accuracy at each epoch: ' + str(score))
        logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(score)) + '%')
        logging.info('Accuracy of model at final epoch: ' + str(100 * score[-1]) + '%')


if __name__ == '__main__':
    """
    This is just an example of a training pipeline.

    Feel free to add or remove more arguments, change default values or hardcode parameters to use.
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss, 'NLLL': torch.nn.NLLLoss}  #
    # Feel
    # free to add
    # more
    opti_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}  # Feel free to add more

    cmdline_parser = argparse.ArgumentParser('DL WS20/21 Competition')

    cmdline_parser.add_argument('-m', '--model',
                                default='ModelZeroSix',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=100,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=32,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'dataset'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                # 0.0005445882245291535
                                default=10e-3,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adam',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-p', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-P', '--model_load_path',
                                default=None,
                                help='Path to load model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='default',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-d', '--data-augmentation',
                                default='resize_to_224x224',
                                help='Data augmentation to apply to data before passing to the model.'
                                     + 'Must be available in data_augmentations.py')
    cmdline_parser.add_argument('-a', '--use-all-data-to-train',
                                action='store_true',
                                help='Uses the train, validation, and test data to train the model if enabled.')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(
        data_dir=args.data_dir,
        torch_model=eval(args.model),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=opti_dict[args.optimizer],
        data_augmentations=eval(args.data_augmentation),  # Check data_augmentations.py for sample augmentations
        save_model_str=args.model_path,
        load_model_str=args.model_load_path,
        exp_name=args.exp_name,
        use_all_data_to_train=args.use_all_data_to_train
    )
