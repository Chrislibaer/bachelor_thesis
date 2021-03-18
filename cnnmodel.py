import torch
import torch.nn.functional as F
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, input_dims=10, output_dims=1):
        super(CNNModel, self).__init__()
        self.conv1d1 = nn.Conv1d(1, 16, kernel_size=1)
        self.conv1d2 = nn.Conv1d(16, 32, kernel_size=3)
        self.conv1d3 = nn.Conv1d(32, 64, kernel_size=5)
        self.conv1d4 = nn.Conv1d(64, 64, kernel_size=7)
        self.conv1d5 = nn.Conv1d(64, 64, kernel_size=9)
        self.fc1 = nn.Linear(64*80, 50)
        self.fc2 = nn.Linear(50, output_dims)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1d1(x))
        x = F.relu(self.conv1d2(x))
        x = F.relu(self.conv1d3(x))
        x = F.relu(self.conv1d4(x))
        x = F.relu(self.conv1d5(x))
        x = x.view(-1, 64*80)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x