from efficientnet_pytorch import EfficientNet


def get_model(num_classes: int):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    return model
