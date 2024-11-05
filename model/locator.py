import torch
import torch.nn as nn

def RMSE_loss(y_pred, y_true):
    squared_errors = torch.sum((y_true-y_pred) ** 2, dim=1)
    squared_error = torch.sqrt(squared_errors)
    rmse_loss = torch.mean(squared_error)
    return rmse_loss

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(32 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):

        x = self.conv1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)

        x = self.conv4(x)
        x = nn.ReLU()(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = nn.ReLU()(x)

        x = self.fc2(x)
        x = nn.ReLU()(x)

        x = self.sigmoid(self.fc3(x))

        return x - 0.5


