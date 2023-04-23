import torchvision
import os
from torch import nn
from efficientnet_pytorch import EfficientNet
from dotenv import load_dotenv

load_dotenv()


def get_model(num_classes: int):
    chosen_model = os.getenv('MODEL')
    if chosen_model == 'densenet':
        return densenet(num_classes=num_classes)
    elif chosen_model == 'efficentnet':
        return efficientnet(num_classes=num_classes)
    elif chosen_model == 'efficientnet_legacy':
        return efficientnet_legacy(num_classes=num_classes)
    else:
        return efficientnet(num_classes=num_classes)


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
