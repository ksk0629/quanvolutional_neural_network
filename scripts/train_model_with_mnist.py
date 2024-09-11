import argparse
import sys

from torchvision import datasets
import torchvision.transforms as transforms

sys.path.append("./src")
from train_model import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and Evaluate a model either QuanvNN or ClassicalCNN with the MNIST data."
    )
    parser.add_argument("-c", "--config_yaml_path", required=False, type=str)
    args = parser.parse_args()

    # Get the MNIST datasets.
    train_dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    num_classes = 10
    print(f"The number of train dataset: {len(train_dataset)}")
    print(f"The number of test dataset: {len(test_dataset)}")

    train(
        config_yaml_path=args.config_yaml_path,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
    )
