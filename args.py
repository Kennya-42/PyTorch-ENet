from argparse import ArgumentParser

def get_arguments():
    """Defines command-line arguments, and parses them."""
    parser = ArgumentParser()
    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full','single'],
        default='train',
        help=("train: performs training and validation; test: tests the model "
              "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        action='store_true',
        help=("The model found in \"--checkpoint_dir/--name/\" and filename "
              "\"--name.h5\" is loaded."))

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=10,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr-decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.5")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=100,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 100")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")
    parser.add_argument(
        "--train_encoder",
        dest='train_encoder',
        action='store_true',
        help="Train the encoder alone.")
    parser.add_argument(
        "--train_decoder",
        dest='train_decoder',
        action='store_true',
        help="Train the decoder alone.")
    parser.add_argument(
        "--reset-optimizer",
        dest='reset-optimizer',
        action='store_true',
        help="Reset optimizer to train encoder and decoder.")
    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes','ritscapes'],
        default='cityscapes',
        help="Dataset to use. Default: cityscapes")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/ken/Documents/Dataset/",
        help="Path to the root directory of the selected dataset. "
        "Default: /home/ken/Documents/Dataset/")
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=600,
        help="The image height. Default: 600")
    parser.add_argument(
        "--weighing",
        choices=['enet', 'mfb', 'none'],
        default='enet',
        help="The class weighing technique to apply to the dataset. "
        "Default: enet")
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="The unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print-step",
        action='store_true',
        help="Print loss every step")
    parser.add_argument(
        "--imshow-batch",
        action='store_true',
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    parser.add_argument(
        "--no-cuda",
        dest='cuda',
        default='store_false',
        help="CPU only.")

    # Storage settings
    parser.add_argument(
        "--name",
        type=str,
        default='ENet',
        help="Name given to the model when saving. Default: ENet")
    parser.add_argument(
        "--save-dir",
        type=str,
        default='save',
        help="The directory where models are saved. Default: save")

    return parser.parse_args()
