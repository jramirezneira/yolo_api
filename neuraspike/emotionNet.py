# import the necessary packages
import torch.nn as nn
import torch.nn.functional as F


class EmotionNet(nn.Module):
    network_config = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M']

    def __init__(self, num_of_channels, num_of_classes):
        super(EmotionNet, self).__init__()
        self.features = self._make_layers(num_of_channels, self.network_config)
        self.classifier = nn.Sequential(
            nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_of_classes)
        )
        # self.classifier = nn.Sequential(nn.Linear(6 * 6 * 128, 64),
        #                                 nn.ELU(True),
        #                                 nn.Dropout(p=0.5),
        #                                 nn.Linear(64, num_of_classes))

    # instructs pytorch how to execute the defined layers in the network
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=True)
        out = self.classifier(out)
        return out

    # generate the convolutional layers within the network
    def _make_layers(self, in_channels, cfg):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # layers += [nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                        #    nn.BatchNorm2d(x),
                           nn.GroupNorm(8, x),
                        #    nn.ELU(inplace=True)]
                            nn.LeakyReLU(negative_slope=0.01, inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)