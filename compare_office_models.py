import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
#                   Office
# ------------------------------------------------------------------------------


class DomainClassifierOffice(nn.Module):
    def __init__(self, input_size=500):
        super(DomainClassifierOffice, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, 1, bias=True)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x


class DomainClassifierDANNOffice(nn.Module):
    def __init__(self, input_size=500):
        super(DomainClassifierDANNOffice, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, 2, bias=True)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x


class FeatureExtractorOffice(nn.Module):
    def __init__(self, n_hidden=256):
        super(FeatureExtractorOffice, self).__init__()
        self.fc1 = nn.Linear(2048, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class DataClassifierOffice(nn.Module):
    def __init__(self, n_hidden=256, class_num=1000):
        super(DataClassifierOffice, self).__init__()
        self.fc = nn.Linear(n_hidden, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x
