from numpy.typing import ArrayLike
import torch.nn as nn
import torch.nn.functional as F


class ClassicalConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, input_dimension: tuple[int, int, int], num_classes: int):
        super().__init__()
        self.kernel_size = 5
        self.num_classes = num_classes
        self.relu = F.relu()
        self.softmax = F.softmax()
        
        self.num_filter1 = 50
        self.conv1 = nn.Conv2d(in_channels=input_dimension[0], out_channels=self.num_filter1, kernel_size=self.kernel_size)
        
        self.num_filter2 = 64
        self.conv2 = nn.Conv2d(in_channels=self.num_filter1, out_channels=self.num_filter2, kernel_size=self.kernel_size)
        
        self.pooling_size = 2
        self.pool = nn.MaxPool2d(kernel_size=self.pooling_size, stride=self.pooling_size)
        
        self.fc1_input_size = (input_dimension[1] - self.num_filter2 + 1) // self.pooling_size
        self.fc1_output_size = 1024
        self.fc1 = nn.Linear(in_features=self.fc1_input_size, out_features=self.fc1_output_size)
        
        self.dropout = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(in_features=self.fc1_output_size, out_features=self.num_classes)
    
    def forward(self, x: ArrayLike):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.fc1(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return self.softmax(x)
