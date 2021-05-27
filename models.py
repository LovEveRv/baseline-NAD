import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_pretrained_vit


class ResNet(nn.Module):
    """
    Model used to pretrain.
    Based on torchvision.models.densenet161
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.resnet = models.resnet152(pretrained=False)
        self.resnet.fc = nn.Sequential()  # actually, this acts like the fc layer is removed.
        self.fc = nn.Linear(2048, num_classes, bias=True)

    def forward(self, x):
        embed = self.resnet(x)
        output = self.fc(embed)
        return output, (embed, )


class VGG(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.net = models.vgg16(pretrained=True)
        # actually, this acts like the fc layer is removed.
        self.net.classifier = nn.Sequential()
        self.fc = nn.Linear(512 * 7 * 7, num_classes, bias=True)

    def forward(self, x):
        embed = self.net(x)
        output = self.fc(embed)
        return output, (embed, )


class VGG_bn(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.net = models.vgg16_bn(pretrained=True)
        # actually, this acts like the fc layer is removed.
        self.net.classifier = nn.Sequential()
        self.fc = nn.Linear(512 * 7 * 7, num_classes, bias=True)

    def forward(self, x):
        embed = self.net(x)
        output = self.fc(embed)
        return output, (embed, )


class ViT(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        self.net = pytorch_pretrained_vit.ViT(
            'B_16_imagenet1k', pretrained=pretrained, image_size=64)
        # actually, this acts like the fc layer is removed.
        self.net.fc = nn.Sequential()
        self.fc = nn.Linear(768, num_classes, bias=True)

    def forward(self, x):
        embed = self.net(x)
        output = self.fc(embed)
        return output, (embed, )
