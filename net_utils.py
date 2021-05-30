import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
# encoder
class PCNencoder(nn.Module):
    def __init__(self, feature_dim):
        super(PCNencoder, self).__init__()
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(512, 512, 1)
        self.conv4 = nn.Conv1d(512, self.feature_dim, 1)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        num_points = x.size(2)

        # first shared mlp
        x = F.relu(self.bn1(self.conv1(x)))
        feature = self.bn2(self.conv2(x))

        # point-wise maxpool
        g = torch.max(feature, 2, keepdim=True)[0]

        # expand and concat
        g = g.repeat((1, 1, num_points))
        x = torch.cat([g, feature], dim=1)

        # second shared mlp
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))

        # point-wise maxpool
        x = torch.max(x, 2)[0]

        return x

# mlp for root node of decoder
class decoder_mlp(nn.Module):
    def __init__(self, encoder_feature_dim, decoder_feature_dim, num_child_node):
        super(decoder_mlp, self).__init__()
        self.decoder_feature_dim = decoder_feature_dim
        self.num_child_node = num_child_node

        self.linear1 = nn.Linear(encoder_feature_dim, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 8 * num_child_node)
        self.bn1_m = nn.BatchNorm1d(256)
        self.bn2_m = nn.BatchNorm1d(64)


    def forward(self, feature):

        x = F.relu(self.bn1_m(self.linear1(feature)))
        x = F.relu(self.bn2_m(self.linear2(x)))
        x = self.linear3(x)
        x = F.tanh(x)

        # concat featur vector + x
        x = x.view(-1, self.decoder_feature_dim, self.num_child_node)
        feature_expand = feature.unsqueeze(2).repeat(1, 1, x.size(2))
        x = torch.cat([x, feature_expand], dim=1)

        return x

# mlp for leaf node of decoder
class decoder_mlp_conv(nn.Module):
    def __init__(self,encoder_feature_dim, decoder_feature_dim, tree_arch, npts):
        super(decoder_mlp_conv, self).__init__()
        self.decoder_feature_dim = decoder_feature_dim
        self.tree_arch = tree_arch
        self.npts = npts

        input_channel = encoder_feature_dim + decoder_feature_dim
        output_channel = decoder_feature_dim

        self.conv11_d = nn.Conv1d(input_channel, int(input_channel / 2), 1)
        self.conv12_d = nn.Conv1d(int(input_channel / 2), int(input_channel / 4), 1)
        self.conv13_d = nn.Conv1d(int(input_channel / 4), int(input_channel / 8), 1)
        self.conv14_d = nn.Conv1d(int(input_channel / 8), output_channel * tree_arch[1], 1)
        self.bn11_d = nn.BatchNorm1d(int(input_channel / 2))
        self.bn12_d = nn.BatchNorm1d(int(input_channel / 4))
        self.bn13_d = nn.BatchNorm1d(int(input_channel / 8))

        self.conv21_d = nn.Conv1d(input_channel, int(input_channel / 2), 1)
        self.conv22_d = nn.Conv1d(int(input_channel / 2), int(input_channel / 4), 1)
        self.conv23_d = nn.Conv1d(int(input_channel / 4), int(input_channel / 8), 1)
        self.conv24_d = nn.Conv1d(int(input_channel / 8), output_channel * tree_arch[2], 1)
        self.bn21_d = nn.BatchNorm1d(int(input_channel / 2))
        self.bn22_d = nn.BatchNorm1d(int(input_channel / 4))
        self.bn23_d = nn.BatchNorm1d(int(input_channel / 8))

        self.conv31_d = nn.Conv1d(input_channel, int(input_channel / 2), 1)
        self.conv32_d = nn.Conv1d(int(input_channel / 2), int(input_channel / 4), 1)
        self.conv33_d = nn.Conv1d(int(input_channel / 4), int(input_channel / 8), 1)
        self.conv34_d = nn.Conv1d(int(input_channel / 8), output_channel * tree_arch[3], 1)
        self.bn31_d = nn.BatchNorm1d(int(input_channel / 2))
        self.bn32_d = nn.BatchNorm1d(int(input_channel / 4))
        self.bn33_d = nn.BatchNorm1d(int(input_channel / 8))

        self.conv41_d = nn.Conv1d(input_channel, int(input_channel / 2), 1)
        self.conv42_d = nn.Conv1d(int(input_channel / 2), int(input_channel / 4), 1)
        self.conv43_d = nn.Conv1d(int(input_channel / 4), int(input_channel / 8), 1)
        self.conv44_d = nn.Conv1d(int(input_channel / 8), output_channel * tree_arch[4], 1)
        self.bn41_d = nn.BatchNorm1d(int(input_channel / 2))
        self.bn42_d = nn.BatchNorm1d(int(input_channel / 4))
        self.bn43_d = nn.BatchNorm1d(int(input_channel / 8))

        self.conv51_d = nn.Conv1d(input_channel, int(input_channel / 2), 1)
        self.conv52_d = nn.Conv1d(int(input_channel / 2), int(input_channel / 4), 1)
        self.conv53_d = nn.Conv1d(int(input_channel / 4), int(input_channel / 8), 1)
        self.conv54_d = nn.Conv1d(int(input_channel / 8), 3 * tree_arch[5], 1)
        self.bn51_d = nn.BatchNorm1d(int(input_channel / 2))
        self.bn52_d = nn.BatchNorm1d(int(input_channel / 4))
        self.bn53_d = nn.BatchNorm1d(int(input_channel / 8))


    def forward(self, x, feature):
            total_node = {}
            for i, node in enumerate(self.tree_arch):
                total_node[i] = np.prod([int(k) for k in self.tree_arch[:i+2]])

            x = F.relu(self.bn11_d(self.conv11_d(x)))
            x = F.relu(self.bn12_d(self.conv12_d(x)))
            x = F.relu(self.bn13_d(self.conv13_d(x)))
            x = self.conv14_d(x)
            x = x.view(-1, self.decoder_feature_dim, total_node[0])
            x = F.tanh(x)
            feature_expand = feature.unsqueeze(2).repeat(1, 1, total_node[0])
            x = torch.cat([x, feature_expand], dim=1)

            x = F.relu(self.bn21_d(self.conv21_d(x)))
            x = F.relu(self.bn22_d(self.conv22_d(x)))
            x = F.relu(self.bn23_d(self.conv23_d(x)))
            x = self.conv24_d(x)
            x = x.view(-1, self.decoder_feature_dim, total_node[1])
            x = F.tanh(x)
            feature_expand = feature.unsqueeze(2).repeat(1, 1, total_node[1])
            x = torch.cat([x, feature_expand], dim=1)

            x = F.relu(self.bn31_d(self.conv31_d(x)))
            x = F.relu(self.bn32_d(self.conv32_d(x)))
            x = F.relu(self.bn33_d(self.conv33_d(x)))
            x = self.conv34_d(x)
            x = x.view(-1, self.decoder_feature_dim, total_node[2])
            x = F.tanh(x)
            feature_expand = feature.unsqueeze(2).repeat(1, 1, total_node[2])
            x = torch.cat([x, feature_expand], dim=1)

            x = F.relu(self.bn41_d(self.conv41_d(x)))
            x = F.relu(self.bn42_d(self.conv42_d(x)))
            x = F.relu(self.bn43_d(self.conv43_d(x)))
            x = self.conv44_d(x)
            x = x.view(-1, self.decoder_feature_dim, total_node[3])
            x = F.tanh(x)
            feature_expand = feature.unsqueeze(2).repeat(1, 1, total_node[3])
            x = torch.cat([x, feature_expand], dim=1)

            x = F.relu(self.bn51_d(self.conv51_d(x)))
            x = F.relu(self.bn52_d(self.conv52_d(x)))
            x = F.relu(self.bn53_d(self.conv53_d(x)))
            x = self.conv54_d(x)
            x = x.view(-1, 3, self.npts)
            x = F.tanh(x)

            return x

def get_arch(tree_arch, npts):
    logmult = int(math.log2(npts/2048))
    assert 2048*(2**(logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch
    while logmult > 0:
        last_min_pos = np.where(arch==np.min(arch))[0][-1]
        arch[last_min_pos]*=2
        logmult -= 1
    return arch
