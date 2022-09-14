import torch
from torch import nn
from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights

class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(1280, 6) 

    def forward(self, x):
        x = self.resnet_layer(x)
        x = torch.flatten(x, 1)
        x = self.Linear_layer(x)
        return x