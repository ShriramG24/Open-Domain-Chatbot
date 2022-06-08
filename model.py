import torch
import torch.nn as nn

class FeedForwardModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardModel, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layer_1(x)
        output = self.relu(output)
        output = self.layer_2(output)
        output = self.relu(output)
        output = self.layer_3(output)
        return output
        