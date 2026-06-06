import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, cfg, in_shape, fc_hidden_dim, latent_dim, wm):
        super().__init__()

        # the encoder consists of a sequence of MaxPool and Conv2d layers
        # the architecture is constructed in _make_layers

        # the watermark will be attached to input as an extra channel
        # so we add 1 to the input channel
        self.features, num_pool, channels = self._make_layers(cfg, in_shape[0]+1)

        # each MaxPool layer will shrink the size by 2 for each dimension
        shrink = 2**num_pool
        self.features_shape = (channels, in_shape[1] // shrink, in_shape[2] // shrink)

        fc_in_dim = channels * self.features_shape[1] * self.features_shape[2]

        # use linear layers to transform features to latent variables.
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_in_dim, fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_dim, latent_dim),
        )

        # linearly transform the watermark so that its size is the same as input
        self.wm_linear = nn.Linear(wm.shape[0], in_shape[1] * in_shape[2])
        self.wm = torch.from_numpy(wm).float().unsqueeze(0)

    def _make_layers(self, cfg, channels):
        layers = []
        num_pool = 0
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                num_pool += 1
            else:
                layers += [
                    nn.Conv2d(channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.LeakyReLU(),
                ]
                channels = x
        return nn.Sequential(*layers), num_pool, channels

    def forward(self, x):
        # move the watermark to the same device as x
        if self.wm.get_device() != x.get_device():
            self.wm = self.wm.to(x.get_device())

        # attach the watermark to the input as an extra channel
        # this step is crucial.
        # otherwise, the training would be extremely difficult as the encoder does not have information about wm.
        wm_attach = self.wm_linear(self.wm)
        wm_attach = wm_attach.repeat(x.shape[0], 1)
        wm_attach = wm_attach.reshape([x.shape[0], 1, x.shape[2], x.shape[3]])

        x = torch.concatenate([x, wm_attach], 1)

        features = self.features(x)

        out = features.view(features.size(0), -1)
        out = self.fc_layers(out)

        return out


class Decoder(nn.Module):
    def __init__(self, cfg, feature_shape, fc_hidden_dim, latent_dim):
        super().__init__()

        # decoder and encoder are symmetrical
        # encoder uses MaxPool layers to downsample features.
        # decoder uses ConvTransposed2d layers to upsample features.
        # decoder aims to transform latent variables from latent space to the image space.
        # the final results are considered as watermarks and will be added to clean images.

        self.feature_shape = feature_shape
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, fc_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_hidden_dim, int(np.prod(feature_shape))),
        )

        self.features = self._make_layers(cfg, feature_shape[0])

    def _make_layers(self, cfg, in_channels):
        layers = []
        for idx, x in enumerate(cfg):
            if x == "M":
                layers += [torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)]
            else:
                if idx != len(cfg) - 1:
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        nn.BatchNorm2d(x),
                        nn.LeakyReLU(inplace=True),
                    ]
                else:
                    # do not add batch norm and relu because we do not want to constrain output range too much
                    layers += [
                        nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    ]
                in_channels = x

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), *self.feature_shape)
        out = self.features(x)

        return out


class AutoEncoder(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, in_shape, fc_hidden_dim, latent_dim, wm):
        super().__init__()

        self.encoder = Encoder(encoder_cfg, in_shape, fc_hidden_dim, latent_dim, wm)
        self.decoder = Decoder(decoder_cfg, self.encoder.features_shape, fc_hidden_dim, latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output

