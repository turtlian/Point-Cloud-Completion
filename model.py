import torch
import torch.nn as nn
from net_utils import PCNencoder, decoder_mlp, decoder_mlp_conv
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

tree_arch = [2, 4, 4, 4, 4, 4]
mlp_dim = [256, 64, 8]

class TopNet(nn.Module):
    def __init__(self, encoder_feature_dim, device):
        super(TopNet, self).__init__()
        self.decoder_feature_dim = mlp_dim[-1]

        # encoder
        self.pcn_encoder= PCNencoder(encoder_feature_dim)
        # decoder: root node
        self.mlp = decoder_mlp(encoder_feature_dim, self.decoder_feature_dim, mlp_dim, tree_arch[0])
        # decoder: leaf node
        self.mlpconv_layers = decoder_mlp_conv(encoder_feature_dim, self.decoder_feature_dim, tree_arch, device)

    def forward(self, x):
        feature = self.pcn_encoder(x)
        x = self.mlp(feature)
        x = self.mlpconv_layers(x, feature)

        return x


if __name__ == '__main__':
    model = TopNet(1024).cuda()
    summary(model, (3,2048))
