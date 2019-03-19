import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init

vgg16 = models.vgg16(pretrained=True)

def conv_backend(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2).double()

class CSRNet(nn.Module):
    
    def __init__(self):
        super(CSRNet, self).__init__()
        features = list(vgg16.features)[:23]
        self.features = nn.ModuleList(features).eval()
        self.conv1 = conv_backend(512, 512)
        init.normal_(self.conv1.weight,std=0.01)
        self.conv2 = conv_backend(512, 512)
        init.normal_(self.conv2.weight,std=0.01)
        self.conv3 = conv_backend(512, 512)
        init.normal_(self.conv3.weight,std=0.01)
        self.conv4 = conv_backend(512, 256)
        init.normal_(self.conv4.weight,std=0.01)
        self.conv5 = conv_backend(256, 128)
        init.normal_(self.conv5.weight,std=0.01)
        self.conv6 = conv_backend(128, 64)
        init.normal_(self.conv6.weight,std=0.01)
        self.convfinal = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, dilation=1).double()
        init.normal_(self.convfinal.weight,std=0.01)
        
    def forward(self,x):
        for model in self.features:
            model=model.double()
            x = model(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.convfinal(x))
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        return x

def csrnet0():
    model = CSRNet()
    return model

def csrnet(model_name, pretrained=False):
    return{
        'csrnet': csrnet0(),
}[model_name]

