import torchvision
from torch import nn
from efficientnet_pytorch import EfficientNet


def get_model(num_classes: int):
    return efficientnet_legacy(num_classes)


def efficientnet(num_classes: int):
    model = torchvision.models.efficientnet_b0(weights=None, num_classes=num_classes)
    return model


def efficientnet_legacy(num_classes: int):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, num_classes)
    return model


def densenet(num_classes: int):
    model = torchvision.models.densenet121(weights=None, num_classes=num_classes)
    return model
