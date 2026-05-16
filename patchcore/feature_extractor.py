import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3

    def forward(self, x):

        x = self.layer1(x)

        feat2 = self.layer2(x)
        feat3 = self.layer3(feat2)

        return feat2, feat3
    
def extract_patches(feature_map):
    B, C, H, W = feature_map.shape
    patches = feature_map.permute(0, 2, 3, 1)
    patches = patches.reshape(-1, C)
    return patches