import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, img_channels, features):
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

        #Encoder
        self.e1 = self.conv_layer(in_channels=self.img_channels, out_channels=self.features,
                                  activation="lrelu", batch_norm="True", dropout="False", padding_mode="reflect")
        
        self.e2 = self.conv_layer(in_channels=self.features, out_channels=self.features * 2,
                                  activation="lrelu", batch_norm="True", dropout="False", padding_mode="reflect")
        
        self.e3 = self.conv_layer(in_channels=self.features * 2, out_channels=self.features * 4,
                                  activation="lrelu", batch_norm="True", dropout="False", padding_mode="reflect")
        
        self.e4 = self.conv_layer(in_channels=self.features * 4, out_channels=self.features * 8,
                                  activation="lrelu", batch_norm="True", dropout="False", padding_mode="reflect")
        
        self.e5 = self.conv_layer(in_channels=self.features * 8, out_channels=self.features * 8,
                                  activation="lrelu", batch_norm="True", dropout="False", padding_mode="reflect")
        
        self.e6 = self.conv_layer(in_channels=self.features * 8, out_channels=self.features * 8,
                                  activation="lrelu", batch_norm="True", dropout="False", padding_mode="reflect")
        
        self.e7 = self.conv_layer(in_channels=self.features * 8, out_channels=self.features * 8,
                                  activation="lrelu", batch_norm="True", dropout="False", padding_mode="reflect")
        
        self.e8 = self.conv_layer(in_channels=self.features * 8, out_channels=self.features * 8,
                                  activation="relu", batch_norm="False", dropout="False")
        

        #Decoder
        self.d1 = self.transpose_conv_layer(in_channels=self.features * 8, out_channels=self.features * 8,
                                            activation="relu", batch_norm="True", dropout="True")
        
        self.d2 = self.transpose_conv_layer(in_channels=self.features * 8 * 2, out_channels=self.features * 8,
                                            activation="relu", batch_norm="True", dropout="True")
        
        self.d3 = self.transpose_conv_layer(in_channels=self.features * 8 * 2, out_channels=self.features * 8,
                                            activation="relu", batch_norm="True", dropout="True")
        
        self.d4 = self.transpose_conv_layer(in_channels=self.features * 8 * 2, out_channels=self.features * 8,
                                            activation="relu", batch_norm="True", dropout="False")
        
        self.d5 = self.transpose_conv_layer(in_channels=self.features * 8 * 2, out_channels=self.features * 4,
                                            activation="relu", batch_norm="True", dropout="False")
        
        self.d6 = self.transpose_conv_layer(in_channels=self.features * 4 * 2, out_channels=self.features * 2,
                                            activation="relu", batch_norm="True", dropout="False")
        
        self.d7 = self.transpose_conv_layer(in_channels=self.features * 2 * 2, out_channels=self.features,
                                            activation="relu", batch_norm="True", dropout="False")
        
        self.d8 = self.transpose_conv_layer(in_channels=self.features * 2, out_channels=self.img_channels,
                                            activation="tanh", batch_norm="False", dropout="False")

    
    def conv_layer(self, in_channels, out_channels, activation, batch_norm, dropout, **kwargs):
        batch_norm_dict = nn.ModuleDict([
            ["True", nn.BatchNorm2d(out_channels)],
            ["False", nn.Identity()]
        ])

        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, padding=1, stride=2, **kwargs),
            batch_norm_dict[batch_norm],
            self.activations[activation],
            self.dropout[dropout]
        )
    
        return layer
    
    def transpose_conv_layer(self, in_channels, out_channels, activation, batch_norm, dropout, **kwargs):
        batch_norm_dict = nn.ModuleDict([
            ["True", nn.BatchNorm2d(out_channels)],
            ["False", nn.Identity()]
        ])

        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, **kwargs),
            batch_norm_dict[batch_norm],
            self.activations[activation],
            self.dropout[dropout]
        )

        return layer
    
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], 1))
        d3 = self.d3(torch.cat([d2, e6], 1))
        d4 = self.d4(torch.cat([d3, e5], 1))
        d5 = self.d5(torch.cat([d4, e4], 1))
        d6 = self.d6(torch.cat([d5, e3], 1))
        d7 = self.d7(torch.cat([d6, e2], 1))
        d8 = self.d8(torch.cat([d7, e1], 1))

        return d8