import torch
from torchvision import models

mnas05= models.mnasnet0_5(pretrained=False)
mnas075= models.mnasnet0_75(pretrained=False)
regnet_y_400mf = models.regnet_y_400mf(pretrained=False)
efficientnet=models.efficientnet_b0(pretrained=False)
densenet121 = models.densenet121(pretrained=False)
resnet18 = models.resnet18(pretrained=False)
lenet = models.googlenet(pretrained=False)

model=lenet
model.aux2.fc2 = torch.nn.Linear(1024, 10, bias=False)
model.aux1.fc2 = torch.nn.Linear(1024, 10, bias=False)
model.fc = torch.nn.Linear(10, 10, bias=False)

# Stampa la struttura del modello
for name, param in model.named_parameters():
    print(name, param.shape)
