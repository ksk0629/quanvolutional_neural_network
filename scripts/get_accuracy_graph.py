# #########################################
# This script is written for MNIST at the moment, but you can change for any dataset.
# The variabels are marked as CHANGE as a comment if they must be changed according to the dataset.
# #########################################
import os
import sys
import yaml

sys.path.append("./src")

import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms as transforms

from classical_cnn import ClassicalCNN
from quanv_nn import QuanvNN


if __name__ == "__main__":
    # Get data.
    dataset = datasets.MNIST(  # CHANGE
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Use the lot of data.
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset))
    data, labels = next(iter(data_loader))
    in_dim = data[0].shape
    num_classes = len(torch.unique(labels))
    total = len(data)

    # Load configs.
    config_dir_path = "./configs"  # CHANGE
    prefix = "mnist"  # CHANGE
    quanv_config_name = "quanv_nn.yaml"
    quanv_config_path = os.path.join(config_dir_path, f"{prefix}_{quanv_config_name}")
    with open(quanv_config_path, "r") as quanv_config_yaml:
        config = yaml.safe_load(quanv_config_yaml)
    config_model = config["model"]
    quanv_kernel_size = config_model["quanv_kernel_size"]
    quanv_num_filters = config_model["quanv_num_filters"]
    quanv_padding_mode = config_model["quanv_padding_mode"]

    # Load the QuanvNN.
    model_dir = "./models/mnist_qnn"  # CHANGE
    print(f"Start to load QuanvNN from {model_dir}.")
    quanv_nn = QuanvNN(
        in_dim,
        num_classes,
        quanv_kernel_size,
        quanv_num_filters,
        quanv_padding_mode,
    )
    filename_prefix = "model"  # CHANGE
    quanv_nn.load(model_dir, filename_prefix)

    # Load the ClassicalCNN.
    model_path = "./models/mnist_cnn/model_final_20.pth"  # CHANGE
    print(f"Start to load ClassicalCNN from {model_path}.")
    classical_cnn = ClassicalCNN(in_dim, num_classes)
    classical_cnn.load_state_dict(torch.load(model_path, weights_only=True))

    # Get the accuracy of the QNN.
    quanv_predictions = quanv_nn.classify(data, config["train"]["shots"])
    quanv_total_correct = (quanv_predictions == labels).sum().item()
    quanv_accuracy = quanv_total_correct / total
    # Get the accuracy of the CNN.
    classical_cnn_predictions = classical_cnn.classify(data)
    classical_cnn_total_correct = (classical_cnn_predictions == labels).sum().item()
    classical_cnn_accuracy = classical_cnn_total_correct / total

    # Make the plot.
    plot_data = {
        "QuanvNN": quanv_accuracy * 100,
        "ClassicalCNN": classical_cnn_accuracy * 100,
    }
    colour = ("green", "orange")
    fig, ax = plt.subplots(figsize=(8, 8))
    data = plot_data.values()
    labels = plot_data.keys()
    width = 0.4
    rect = ax.bar(labels, data, width)

    # Draw accuracy value on the bars.
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    ax.bar(labels, data, color=colour)

    ax.set_xlabel("Type of NN")
    ax.set_xlabel("Accuracy")
    ax.set_title("QuanvNN vs. ClassicalCNN over MNIST")  # CHANGE
    autolabel(rect)

    plt.show()

    fig.savefig("mnist_accuracy_graph.png", bbox_inches="tight")  # CHANGE
