import torchvision
from efficientnet_pytorch import EfficientNet


def get_model(num_classes: int):
    return efficientnet(num_classes)


def efficientnet(num_classes: int):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    return model


def densenet(num_classes: int):
    model = torchvision.models.densenet121(weights=None, num_classes=num_classes)
    return model
