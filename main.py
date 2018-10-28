import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from collections import OrderedDict
import transforms as ext_transforms
from models.enet import ENet
from train import Train
from test import Test
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils
from PIL import Image
import time
import numpy as np

# Get the arguments
args = get_arguments()
use_cuda = args.cuda and torch.cuda.is_available()

def load_dataset(dataset):
    print("\nLoading dataset...\n")
    print("Selected dataset:", args.dataset)
    print("Dataset directory:", args.dataset_dir)
    print("Save directory:", args.save_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
        transforms.Resize((args.height, args.width)),
        ext_transforms.PILToLongTensor()
    ])

    # Get selected dataset
    # Load the training set as tensors
    train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the validation set as tensors
    val_set = dataset(
        args.dataset_dir,
        mode='val',
        transform=image_transform,
        label_transform=label_transform)
    val_loader = data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Load the test set as tensors
    test_set = dataset(
        args.dataset_dir,
        mode='test',
        transform=image_transform,
        label_transform=label_transform)
    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = train_set.color_encoding
    # Get number of classes to predict
    num_classes = len(class_encoding)
    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Validation dataset size:", len(val_set))
    # Get a batch of samples to display
    if args.mode.lower() == 'test':
        images, labels = iter(test_loader).next()
    else:
        images, labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())

    # Show a batch of samples and labels
    if args.imshow_batch and False:
        print("Close the figure window to continue...")
        label_to_rgb = transforms.Compose([
            ext_transforms.LongTensorToRGBPIL(class_encoding),
            transforms.ToTensor()
        ])
        color_labels = utils.batch_transform(labels, label_to_rgb)
        utils.imshow_batch(images, color_labels)

    # Get class weights from the selected weighing technique
    print("Weighing technique:", args.weighing)
    print("Computing class weights...")
    class_weights = 0
    # class_weights = np.array([3.5071, 1.8638])
    if args.weighing.lower() == 'enet':
        class_weights = enet_weighing(train_loader, num_classes)
    elif args.weighing.lower() == 'mfb':
        class_weights = median_freq_balancing(train_loader, num_classes)
    else:
        class_weights = None

    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float()
        # Set the weight of the unlabeled class to 0
        if args.ignore_unlabeled:
            ignore_index = list(class_encoding).index('unlabeled')
            class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, val_loader,
            test_loader), class_weights, class_encoding


def train(train_loader, val_loader, class_weights, class_encoding):
    print("\nTraining...\n")

    num_classes = len(class_encoding)

    # Intialize ENet
    model = ENet(num_classes)
    # Check if the network architecture is correct
    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ENet authors used Adam as the optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # Learning rate decay scheduler
    lr_updater = lr_scheduler.StepLR(optimizer, args.lr_decay_epochs,
                                     args.lr_decay)

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        model, optimizer, start_epoch, best_miou = utils.load_checkpoint(
            model, optimizer, args.save_dir, args.name)
        print("Resuming from model: Start epoch = {0} "
              "| Best mean IoU = {1:.4f}".format(start_epoch, best_miou))
    else:
        start_epoch = 0
        best_miou = 0
    # Start Training
    train = Train(model, train_loader, optimizer, criterion, metric, use_cuda)
    val = Test(model, val_loader, criterion, metric, use_cuda)
    for epoch in range(start_epoch, args.epochs):
        print(">> [Epoch: {0:d}] Training".format(epoch))
        lr_updater.step()
        epoch_loss, (iou, miou) = train.run_epoch(args.print_step)
        print(">> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))
        #preform a validation test
        if (epoch + 1) % 4 == 0 or epoch + 1 == args.epochs:
            print(">>>> [Epoch: {0:d}] Validation".format(epoch))
            loss, (iou, miou) = val.run_epoch(args.print_step)
            print(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, loss, miou))
            # Print per class IoU on last epoch or if best iou
            if epoch + 1 == args.epochs or miou > best_miou:
                for key, class_iou in zip(class_encoding.keys(), iou):
                    print("{0}: {1:.4f}".format(key, class_iou))
            # Save the model if it's the best thus far
            if miou > best_miou:
                print("Best model thus far. Saving...")
                best_miou = miou
                utils.save_checkpoint(model, optimizer, epoch + 1, best_miou,
                                      args)

    return model


def test(model, test_loader, class_weights, class_encoding):
    print("\nTesting...\n")

    num_classes = len(class_encoding)

    # We are going to use the CrossEntropyLoss loss function as it's most
    # frequentely used in classification problems with multiple classes which
    # fits the problem. This criterion  combines LogSoftMax and NLLLoss.
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if use_cuda:
        criterion = criterion.cuda()

    # Evaluation metric
    if args.ignore_unlabeled:
        ignore_index = list(class_encoding).index('unlabeled')
    else:
        ignore_index = None
    metric = IoU(num_classes, ignore_index=ignore_index)

    # Test the trained model on the test set
    test = Test(model, test_loader, criterion, metric, use_cuda)

    print(">>>> Running test dataset")

    loss, (iou, miou) = test.run_epoch(args.print_step)
    class_iou = dict(zip(class_encoding.keys(), iou))

    print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

    # Print per class IoU
    for key, class_iou in zip(class_encoding.keys(), iou):
        print("{0}: {1:.4f}".format(key, class_iou))

    # Show a batch of samples and labels
    if args.imshow_batch:
        print("A batch of predictions from the test set...")
        images, _ = iter(test_loader).next()
        predict(model, images, class_encoding)


def predict(model, images, class_encoding):
    images = Variable(images)
    if use_cuda:
        images = images.cuda()
    # Make predictions!
    predictions = model(images)
    # Predictions is one-hot encoded with "num_classes" channels.
    # Convert it to a single int using the indices where the maximum (1) occurs
    _, predictions = torch.max(predictions.data, 1)
    label_to_rgb = transforms.Compose([
        ext_transforms.LongTensorToRGBPIL(class_encoding),
        transforms.ToTensor()
    ])
    color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
    utils.imshow_batch(images.data.cpu(), color_predictions)

def single():
    # img = Image.open("berlin_000000_000019_leftImg8bit.png").convert('RGB')
    img = Image.open("winter_sentinel_sunny_1229.jpg").convert('RGB')
    # img = Image.open("spring_sentinel_cloudy_0.jpg").convert('RGB')
    # img = Image.open("winter_sentinel_sunny_632.png").convert('RGB')
    class_encoding= color_encoding = OrderedDict([
            ('unlabeled', (0, 0, 0)),
            ('road', (128, 64, 128)),
            ('sidewalk', (244, 35, 232))
    ])
    # Intialize a new ENet model
    num_classes = len(class_encoding)
    model = ENet(num_classes)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    # Load the previoulsy saved model state to the ENet model
    model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]
    img = img.resize((600, 360), Image.ANTIALIAS)
    start = time.time()
    images = transforms.ToTensor()(img)
    torch.reshape(images, (1, 3, 360, 600))
    images= images.unsqueeze(0)
    with torch.no_grad():
        images = Variable(images)
        images = images.cuda()
        predictions = model(images) 
        end = time.time()
        print(int(1/(end - start)),"FPS")
        _, predictions = torch.max(predictions.data, 1)
        label_to_rgb = transforms.Compose([ext_transforms.LongTensorToRGBPIL(class_encoding),transforms.ToTensor()])
        color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
        end = time.time()
        print(int(1/(end - start)),"FPS")
        utils.imshow_batch(images.data.cpu(), color_predictions)

if __name__ == '__main__':
    if args.mode.lower() == 'single':
        single()
    else:
        # Fail fast if the saving directory doesn't exist
        assert os.path.isdir(
            args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(args.dataset_dir)
        assert os.path.isdir(
            args.save_dir), "The directory \"{0}\" doesn't exist.".format(args.save_dir)
        # Import the requested dataset
        if args.dataset.lower() == 'camvid':
            from data import CamVid as dataset
        elif args.dataset.lower() == 'cityscapes':
            from data import Cityscapes as dataset
        elif args.dataset.lower() == 'ritscapes':
            from data import Ritscapes as dataset
        else:
            raise RuntimeError("\"{0}\" is not a supported dataset.".format(
                args.dataset))
        
        loaders, w_class, class_encoding = load_dataset(dataset)
        train_loader, val_loader, test_loader = loaders

        if args.mode.lower() in {'train', 'full'}:
            model = train(train_loader, val_loader, w_class, class_encoding)
            if args.mode.lower() == 'full':
                test(model, test_loader, w_class, class_encoding)
        elif args.mode.lower() == 'test':
            num_classes = len(class_encoding)
            model = ENet(num_classes)
            if use_cuda:
                model = model.cuda()
            optimizer = optim.Adam(model.parameters())
            model = utils.load_checkpoint(model, optimizer, args.save_dir, args.name)[0]
            test(model, test_loader, w_class, class_encoding)
        else:
            raise RuntimeError(
                "\"{0}\" is not a valid choice for execution mode.".format(
                    args.mode))
