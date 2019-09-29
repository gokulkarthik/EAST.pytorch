from config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

config = {k:v for k,v in vars(Config).items() if not k.startswith("__")}

class EAST(nn.Module):


    def __init__(self, geometry="QUAD", label_method="single"):
        super(EAST, self).__init__()

        self.geometry = geometry
        self.label_method = label_method
        self.representation = geometry + "_" + label_method

        ## Feature Extraction Essentials
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # 1/32


        ## Feature Merging Essentials
        layer1 = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                               nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        layer2 = nn.Sequential(nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                               nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        layer3 = nn.Sequential(nn.Conv2d(256, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                               nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        layer4 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.feature_convs = nn.ModuleList([layer1, layer2, layer3, layer4])

        self.unpool = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        ## Output Layer Essentials
        self.out_score = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        if self.representation == "QUAD_single":
            self.out_geo = nn.Sequential(nn.Conv2d(32, 8, 1), nn.Sigmoid())
        elif self.representation == "QUAD_multiple":
            self.out_geo = nn.Sequential(nn.Conv2d(32, 8, 1))
        elif self.representation == "RBOX_single":
            self.out_geo = nn.Sequential(nn.Conv2d(32, 4, 1), nn.Sigmoid())
            self.out_angle = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

        self._init_weights()

        vgg16 = torchvision.models.vgg16(pretrained=True)

        self.copy_params_from_vgg16(vgg16)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]

        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        pool2 = h

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        pool5 = h


        f = [pool5, pool4, pool3, pool2]
        g = [None, None, None, None]
        h = [None, None, None, None]

        for i in range(4):

            if i == 0:
                h[i] = f[i]
            else:
                concat = torch.cat([g[i - 1], f[i]], dim=1)
                h[i] = self.feature_convs[i - 1](concat)

            if i <= 2:
                g[i] = self.unpool(h[i])
            else:
                g[i] = self.feature_convs[i](h[i])


        score_map = self.out_score(g[3])
        geo_map = self.out_geo(g[3])
        if self.representation == "QUAD_single":
            geometry_map = geo_map * 512
        elif self.representation == "QUAD_multiple":
            geometry_map = geo_map
        elif self.representation == "RBOX_single":
            angle_map = self.out_angle(g[3])
            angle_map = (angle_map - 0.5) * math.pi / 2
            geometry_map = torch.cat((geo_map, angle_map), dim=1)

        #print("pool1", pool5.size())
        #print("h1", h[0].size())
        #print("g1", g[0].size())

        return score_map, geometry_map

# test code
"""
east = EAST(geometry=config["geometry"])
x = torch.randint(low=0, high=255, size=(1, 3, 512, 512))
score_map, geometry_map = east.forward(x)
print(score_map.size(), geometry_map.size())
print(score_map, geometry_map)
"""