from torch import nn
import torch.nn.utils.weight_norm as weightNorm

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Classifier(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, input_data):
        logits = self.fc(input_data)
        return  logits

class Classifier_covnextT(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classifier_covnextT, self).__init__()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_data):
        logits = self.fc(input_data)
        return  logits

class Classifier_SwinT(nn.Module):
    def __init__(self, num_classes=1000):
        super(Classifier_SwinT, self).__init__()
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, input_data):
        logits = self.fc(input_data)
        return  logits