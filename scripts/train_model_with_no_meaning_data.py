import argparse
import sys

import torch

sys.path.append("./src")
from plain_dataset import PlainDataset
from train_model import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and Evaluate a model either QuanvNN or ClassicalCNN with a no meaning dataset."
    )
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str)
    args = parser.parse_args()

    # Create a no meaning dataset.
    print("=== Create an easy dataset ===")
    dataset_size = 10
    channels = 1
    data_size = 16
    num_classes = 10
    assert (
        data_size > 15
    ), "data_size must be more than 15 as ClassicalCNN's kernel and pooling sizes are fixed as 5x5."
    print(
        f"(dataset size, channels, data_size, data_size = {(dataset_size, channels, data_size, data_size)}"
    )
    data = torch.rand(size=(dataset_size, channels, data_size, data_size))
    print(f"data.shape = {data.shape}")
    labels = torch.randint(low=0, high=num_classes, size=(dataset_size,))
    print(f"labels.shape = {labels.shape}")
    dataset = PlainDataset(data, labels)
    print()

    train(
        config_yaml_path=args.config_yaml_path,
        train_dataset=dataset,
        test_dataset=dataset,
        num_classes=num_classes,
    )
