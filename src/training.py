from tqdm import tqdm
import time
import torchvision
import random
from torchvision.transforms.functional import adjust_contrast, adjust_brightness, gaussian_blur

from src.eval.evaluate import AverageMeter, accuracy


def train_fn(model, optimizer, criterion, loader, device):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0

    t = tqdm(loader)
    for images, labels in t:
        images = images.to(device)
        print("ACHTTTTUUNGGG")
        print(images.shape)
        print(labels)
        print(labels.dtype)
        labels = labels.to(device)
        rand = random.random()
        if rand < 0.05:
            images = gaussian_blur(images, random.choice([(3, 3), (5, 5)]))
        if 0.05 > rand > 0.1:
            images = torchvision.transforms.functional.adjust_contrast(images, random.choice([0.2, 0.25, 0.3,
                                                                                              0.35, 0.4, 0.45]))
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        acc = accuracy(logits, labels)
        n = images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)
        t.set_description('(=> Training) Loss: {:.4f}'.format(losses.avg))

    time_train += time.time() - time_begin
    print('training time: ' + str(time_train))
    return score.avg, losses.avg
