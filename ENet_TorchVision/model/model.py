import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet50, efficientnet_b7


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ResNet50Mask(BaseModel):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50()
        self.resnet.fc = nn.Linear(2048, 18)

    def forward(self, x):
        x = self.resnet(x)
        return x


class EfficientNet_B7_Mask(BaseModel):
    def __init__(self):
        super().__init__()
        self.enet = efficientnet_b7()
        self.enet.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2560, 18)
        )
    
    def forward(self, x):
        x = self.enet(x)
        return x