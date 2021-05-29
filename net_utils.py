import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# encoder
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
        x, _ = torch.max(x, 2)
        # concat hidden feature vector, input
        x = x.unsqueeze(2)
        x = x.repeat((1, 1, input.size(2)))
        x = torch.cat([x, input], dim=1)
        # second stacked pointnet layer
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x, _ = torch.max(x, 2)

        return x

# mlp for root node of decoder
class decoder_mlp(nn.Module):
    def __init__(self, encoder_feature_dim, decoder_feature_dim, layer_dim, num_child_node):
        super(decoder_mlp, self).__init__()
        layer_dim = [encoder_feature_dim] + layer_dim
        self.decoder_feature_dim = decoder_feature_dim
        self.num_child_node = num_child_node
        self.mlp_layers = []

        for i in range(len(layer_dim) - 1):
            if i != len(layer_dim) - 2:
                self.mlp_layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
                self.mlp_layers.append(nn.BatchNorm1d(layer_dim[i + 1]))
                self.mlp_layers.append(nn.ReLU())
            else:
                self.mlp_layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1] * num_child_node))

        self.mlp_layers = nn.Sequential(*self.mlp_layers)

    def forward(self, feature):
        x = self.mlp_layers(feature)
        x = F.tanh(x)
        x = x.view(-1, self.decoder_feature_dim, self.num_child_node)
        feature_expand = feature.unsqueeze(2).repeat(1, 1, x.size(2))
        x = torch.cat([x, feature_expand], dim=1)

        return x

# mlp for leaf node of decoder
class decoder_mlp_conv(nn.Module):
    def __init__(self,encoder_feature_dim, decoder_feature_dim, tree_arch):
        super(decoder_mlp_conv, self).__init__()
        self.decoder_feature_dim = decoder_feature_dim
        self.tree_arch = tree_arch

        self.hier_mlpconv_layers = []
        for i, num_node in enumerate(self.tree_arch[1:]):
            input_channel = encoder_feature_dim + decoder_feature_dim
            output_channel = decoder_feature_dim
            if i == len(self.tree_arch[1:]) - 1:
                output_channel = 3
            layer_dim = [input_channel, int(input_channel / 2), int(input_channel / 2), int(input_channel / 4),
                         int(input_channel / 8), output_channel * num_node]

            self.mlpconv_layers = []
            for i in range(len(layer_dim) - 1):
                if i != len(layer_dim) - 2:
                    self.mlpconv_layers.append(nn.Conv1d(layer_dim[i], layer_dim[i + 1], 1))
                    self.mlpconv_layers.append(nn.BatchNorm1d(layer_dim[i + 1]))
                    self.mlpconv_layers.append(nn.ReLU())
                else:
                    self.mlpconv_layers.append(nn.Conv1d(layer_dim[i], layer_dim[i + 1], 1))
            self.hier_mlpconv_layers.append(nn.Sequential(*self.mlpconv_layers))

    def forward(self, x, feature):
        for i, num_node in enumerate(self.tree_arch[1:]):
            x = self.hier_mlpconv_layers[i].cuda()(x)
            x = F.tanh(x)
            total_node = np.prod([int(k) for k in self.tree_arch[:i+2]])
            if i != len(self.tree_arch[1:])-1:
                x = x.view(-1, self.decoder_feature_dim, total_node)
                feature_expand = feature.unsqueeze(2).repeat(1, 1, total_node)
                x = torch.cat([x, feature_expand], dim=1)
            else:
                x = x.view(-1, 3, 2048)

        return x
