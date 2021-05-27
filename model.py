import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of children per tree levels for 2048 output points
tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]

class TopNet(nn.Module):
    def __init__(self, feature_dim):
        super(TopNet, self).__init__()
        self.pcn_encoder = PCNencoder(feature_dim)
        self.decoder = nn.Sequential(

        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits



class PCNencoder(nn.Module):
    def __init__(self, feature_dim):
        super(PCNencoder, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128 + 3, 512, 1)
        self.conv4 = nn.Conv1d(512, self.feature_dim, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(feature_dim)

    def forward(self, input):
        # first stacked pointnet layer
        x = F.relu(self.bn1(self.conv1(input)))
        x = F.relu(self.bn2(self.conv2(x)))
        # concat hidden feature vector, input
        x, _ = torch.max(x, 1)
        x = x.unsqueeze(1)
        x = x.repeat((1, input.size(1)))
        x = torch.cat([x, input], dim=2)
        # second stacked pointnet layer
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x, _ = torch.max(x, 1)

        return x
