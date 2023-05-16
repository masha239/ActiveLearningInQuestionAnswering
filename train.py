from datasets import load_from_disk

from src.model import ActiveQA
from src.utils import create_config
import argparse
import json
import torch


def get_args():
    parser = argparse.ArgumentParser(description='Train QA model')
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='config path (.json); if not provided, default config will be created'
    )
    parser.add_argument('--train', type=str, help='train dataset folder')
    parser.add_argument('--test', type=str, help='test dataset folder')
    parser.add_argument('--val', type=str, default=None, help='Val dataset folder')
    parser.add_argument('--val-answers', type=str, default=None, help='Val dataset folder')

    args = parser.parse_args()
    return args


def train(args):
    config_path = args.config
    if config_path is not None:
        with open(args.config) as f:
            config = json.load(f)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = create_config(device)

    train_dataset = load_from_disk(args.train)
    test_dataset = load_from_disk(args.test)

    qa = ActiveQA(config)
    qa.train(train_dataset, test_dataset)

    if args.val is not None and args.val_answers is not None:
        val_dataset = load_from_disk(args.val)
        val_answers_dataset = load_from_disk(args.val_answers)
        qa.evaluate(val_dataset, val_answers_dataset)


if __name__ == '__main__':
    args = get_args()
    train(args)
