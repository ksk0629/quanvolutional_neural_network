import sys

sys.path.append("./src")

import torch

from plain_dataset import PlainDataset
from quanv_nn import QuanvNN
from quanv_nn_trainer import QuanvNNTrainer
import utils


if __name__ == "__main__":
    # Fix the random seeds.
    seed = 42
    utils.fix_seed(seed)

    # Make an easy dataset.
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

    # Create an instance of QuanvNN.
    print("=== Create an instance of QuanvNN ===")
    in_dim = data[0].shape
    quanv_kernel_size = (3, 3)
    quanv_num_filters = 2
    qnn = QuanvNN(
        in_dim=in_dim,
        num_classes=num_classes,
        quanv_kernel_size=quanv_kernel_size,
        quanv_num_filters=quanv_num_filters,
    )
    print()

    # Train QNN.
    print("=== Train QNN ===")
    epochs = 100
    batch_size = 2
    save_steps = 25
    shots = 20480 * 2
    model_output_dir = "./models"
    model_name = "proto_train_qnn_model.pth"
    processed_data_filename = "proto_train_qnn_data.pt"
    processed_data_output_dir = "./data"
    print(f"epochs: {epochs}")
    print(f"batch_size: {batch_size}")
    print(f"save_steps: {save_steps}")
    print(f"shots: {shots}")
    print(f"model_output_dir: {model_output_dir}")
    print(f"processed_data_filename: {processed_data_filename}")
    print(f"processed_data_output_dir: {processed_data_output_dir}")
    qnn_trainer = QuanvNNTrainer(
        qnn=qnn,
        train_dataset=dataset,
        test_dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        save_steps=save_steps,
        shots=shots,
        model_output_dir=model_output_dir,
        model_name=model_name,
        processed_data_filename=processed_data_filename,
        processed_data_output_dir=processed_data_output_dir,
    )
    print("=== Train QNN ===")

    qnn_trainer.train_and_test()
