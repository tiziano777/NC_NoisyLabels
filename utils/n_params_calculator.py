import torchvision.models as models

def model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

resnet18 = models.resnet18(pretrained=False)
densenet121 = models.densenet121(pretrained=False)
regnet_y_400mf = models.regnet_y_400mf(pretrained=False)
lenet = models.googlenet(pretrained=False)
mnas= models.mnasnet0_5(pretrained=False)
squeezenet= models.squeezenet1_0(pretrained=False)
mnas075= models.mnasnet0_75(pretrained=False)
efficientnet=models.efficientnet_b0(pretrained=False)

mnas05= models.mnasnet0_5(pretrained=False)             # 2M
mnas075= models.mnasnet0_75(pretrained=False)           # 3.2M
regnet_y_400mf = models.regnet_y_400mf(pretrained=False)# 4.3M
efficientnet=models.efficientnet_b0(pretrained=False)   # 5.2M
densenet121 = models.densenet121(pretrained=False)      # 8M
resnet18 = models.resnet18(pretrained=False)            # 11.3M
lenet = models.googlenet(pretrained=False)              # 13M

