import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, img_channels, features=64):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.features = features

        self.activations = nn.ModuleDict([
            ["tanh", nn.Tanh()],
            ["lrelu", nn.LeakyReLU(0.2)],
            ["relu", nn.ReLU()]
        ])

        self.dropout = nn.ModuleDict([
            ["True", nn.Dropout(0.5)],
            ["False", nn.Identity()]
        ])

        # Encoder
        self.e1 = self.conv_layer(img_channels, features, "lrelu", "True", "False", padding_mode="reflect")
        self.e2 = self.conv_layer(features, features*2, "lrelu", "True", "False", padding_mode="reflect")
        self.e3 = self.conv_layer(features*2, features*4, "lrelu", "True", "False", padding_mode="reflect")
        self.e4 = self.conv_layer(features*4, features*8, "lrelu", "True", "False", padding_mode="reflect")
        self.e5 = self.conv_layer(features*8, features*8, "lrelu", "True", "False", padding_mode="reflect")
        self.e6 = self.conv_layer(features*8, features*8, "lrelu", "True", "False", padding_mode="reflect")
        self.e7 = self.conv_layer(features*8, features*8, "lrelu", "True", "False", padding_mode="reflect")
        self.e8 = self.conv_layer(features*8, features*8, "relu", "False", "False")

        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=features*8, num_heads=8)

        # Decoder
        self.d1 = self.transpose_conv_layer(features*8, features*8, "relu", "True", "True")
        self.d2 = self.transpose_conv_layer(features*16, features*8, "relu", "True", "True")
        self.d3 = self.transpose_conv_layer(features*16, features*8, "relu", "True", "True")
        self.d4 = self.transpose_conv_layer(features*16, features*8, "relu", "True", "False")
        self.d5 = self.transpose_conv_layer(features*16, features*4, "relu", "True", "False")
        self.d6 = self.transpose_conv_layer(features*8, features*2, "relu", "True", "False")
        self.d7 = self.transpose_conv_layer(features*4, features, "relu", "True", "False")
        self.d8 = self.transpose_conv_layer(features*2, img_channels, "tanh", "False", "False")

    def conv_layer(self, in_channels, out_channels, activation, batch_norm, dropout, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, **kwargs),
            nn.BatchNorm2d(out_channels) if batch_norm == "True" else nn.Identity(),
            self.activations[activation],
            self.dropout[dropout]
        )

    def transpose_conv_layer(self, in_channels, out_channels, activation, batch_norm, dropout):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels) if batch_norm == "True" else nn.Identity(),
            self.activations[activation],
            self.dropout[dropout]
        )

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)    
        e2 = self.e2(e1)   
        e3 = self.e3(e2)   
        e4 = self.e4(e3)   
        e5 = self.e5(e4)   
        e6 = self.e6(e5)   
        e7 = self.e7(e6)   
        e8 = self.e7(e7)   

        # MHA
        b, c, h, w = e8.shape
        e8_flat = e8.view(b, c, h*w).permute(2, 0, 1)
        attn_output, _ = self.mha(e8_flat, e8_flat, e8_flat)
        attn_output = attn_output.permute(1, 2, 0).view(b, c, h, w)

        # Decoder
        d1 = self.d1(attn_output)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))
        
        return d8
    
'''gen = Generator(img_channels=3)
x = torch.randn(1, 3, 512, 512)
print(gen(x).shape)'''
