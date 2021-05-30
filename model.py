import torch.nn as nn
from net_utils import PCNencoder, PCNdecoder, decoder_mlp, decoder_mlp_conv, get_arch
from torchsummary import summary

tree_arch = [2, 2, 2, 2, 2, 4, 4, 4]

class TopNet(nn.Module):
    def __init__(self, encoder_feature_dim, decoder_feature_dim, npts):
        super(TopNet, self).__init__()
        self.decoder_feature_dim = decoder_feature_dim
        self.tree_arch = get_arch(tree_arch, npts)

        # encoder
        self.pcn_encoder= PCNencoder(encoder_feature_dim)
        # decoder: root node
        self.mlp = decoder_mlp(encoder_feature_dim, self.decoder_feature_dim, self.tree_arch[0])
        # decoder: leaf node
        self.mlpconv_layers = decoder_mlp_conv(encoder_feature_dim, self.decoder_feature_dim,
                                               self.tree_arch, npts)

    def forward(self, x):
        feature = self.pcn_encoder(x)
        x = self.mlp(feature)
        x = self.mlpconv_layers(x, feature)

        return x


class PCN(nn.Module):
    def __init__(self, encoder_feature_dim, num_coarse, num_dense): # num_dense = npts
        super(PCN, self).__init__()

        self.encoder = PCNencoder(encoder_feature_dim)
        self.decoder = PCNdecoder(num_coarse, num_dense)

    def forward(self, x):
        v = self.encoder(x)
        y_coarse, y_detail = self.decoder(v)
        return v, y_coarse, y_detail


if __name__ == '__main__':
    # model = TopNet(1024, 8, 16384).cuda()
    model = PCN(1024, 1024, 16384).cuda()
    summary(model, (3,2048))
