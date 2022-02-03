import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
#                   VISDA
# ------------------------------------------------------------------------------


class DomainClassifierVisda(nn.Module):
    def __init__(self, input_dim=100):
        super(DomainClassifierVisda, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Linear(input_dim, 1, bias=True)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x


class DomainClassifierDANNVisda(nn.Module):
    def __init__(self, input_dim=100):
        super(DomainClassifierDANNVisda, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=True)
        self.fc2 = nn.Linear(input_dim, 2, bias=True)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x


class FeatureExtractorVisda(nn.Module):
    def __init__(self, input_dim=100):
        super(FeatureExtractorVisda, self).__init__()
        self.fc1 = nn.Linear(2048, input_dim, bias=True)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        return x


class DataClassifierVisda(nn.Module):
    def __init__(self, input_dim=100):
        super(DataClassifierVisda, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, 12)

    def forward(self, input):
        x = F.relu(self.fc1(input.view(input.size(0), -1)))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
