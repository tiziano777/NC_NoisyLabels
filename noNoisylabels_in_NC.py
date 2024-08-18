import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Hooks import FeaturesHook

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg as scilin

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')

folder_path= './NC_experiments'

def grayscale_to_rgb(image):
    return image.convert("RGB")

# grey scale transformations: tensorizzazione e normalizzazione
grey_transform = transforms.Compose([
    transforms.Lambda(lambda x: grayscale_to_rgb(x)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# color transformations:  tensorizzazione e normalizzazione
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""CIFAR-10"""

# Caricamento del dataset di addestramento
cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)


# Caricamento del dataset di test
cifar10_testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)


#DataLoaders
cifar10_trainloader = torch.utils.data.DataLoader(cifar10_trainset, batch_size=128,
                                          shuffle=True, num_workers=2, drop_last=True)


cifar10_testloader = torch.utils.data.DataLoader(cifar10_testset, batch_size=128, num_workers=2, drop_last=True)

cifar10_lenet = models.googlenet(pretrained=False)
cifar10_resnet18 = models.resnet18(pretrained=False)
cifar10_densenet121 = models.densenet121(pretrained=False)
cifar10_regnet = models.regnet_y_400mf(pretrained=False)
cifar10_efficientnet=models.efficientnet_b0(pretrained=False)
cifar10_mnas075= models.mnasnet0_75(pretrained=False)
cifar10_mnas05= models.mnasnet0_5(pretrained=False)


cifar10_classes = 10

cifar10_densenet121.classifier = nn.Linear(cifar10_densenet121.classifier.in_features, cifar10_classes, bias=False)
cifar10_resnet18.fc = nn.Linear(cifar10_resnet18.fc.in_features, cifar10_classes, bias=False)
cifar10_lenet.fc = nn.Linear(cifar10_lenet.fc.in_features, cifar10_classes, bias=False)
cifar10_regnet.fc = nn.Linear(cifar10_regnet.fc.in_features, cifar10_classes, bias=False)

cifar10_efficientnet.classifier[1]=nn.Linear(cifar10_efficientnet.classifier[1].in_features, cifar10_classes, bias=False)
cifar10_mnas05.classifier[1]=nn.Linear(cifar10_mnas05.classifier[1].in_features, cifar10_classes, bias=False)
cifar10_mnas075.classifier[1]=nn.Linear(cifar10_mnas075.classifier[1].in_features, cifar10_classes, bias=False)

densenet_feature_size = cifar10_densenet121.classifier.in_features
resnet_feature_size = cifar10_resnet18.fc.in_features
regnet_feature_size = cifar10_regnet.fc.in_features
efficientnet_feature_size= cifar10_efficientnet.classifier[1].in_features
mnas_feature_size=cifar10_mnas05.classifier[1].in_features
lenet_feature_size=cifar10_lenet.fc.in_features


#OTHER DATASET COMMENTED******************************************

"""FASHION-MNIST"""

# Caricamento del dataset di addestramento Fashion MNIST
fashion_mnist_trainset = datasets.FashionMNIST(root='./data', train=True,
                                               download=True, transform=grey_transform)

# Caricamento del dataset di test Fashion MNIST
fashion_mnist_testset = datasets.FashionMNIST(root='./data', train=False,
                                              download=True, transform=grey_transform)

fashion_mnist_images_per_class = 500
fashion_mnist_total_images_to_select = fashion_mnist_images_per_class * len(fashion_mnist_trainset.classes)
fashion_mnist_sampler = SubsetRandomSampler(torch.randperm(len(fashion_mnist_trainset))[:fashion_mnist_total_images_to_select])

# DataLoaders per Fashion MNIST
fashion_mnist_trainloader = DataLoader(fashion_mnist_trainset, batch_size=32,
                                       sampler=fashion_mnist_sampler, num_workers=2, drop_last=True)

fashion_mnist_indices = torch.randperm(len(fashion_mnist_testset))[:384]  # Seleziona casualmente 384 indici per evaluation
fashion_mnist_test_sampler = SubsetRandomSampler(fashion_mnist_indices)                                      
fashion_mnist_testloader = DataLoader(fashion_mnist_testset, batch_size=32,
                                      sampler=fashion_mnist_test_sampler, num_workers=2)

fashion_mnist_densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
fashion_mnist_resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
fashion_mnist_lenet = models.googlenet(pretrained=False)
fashion_mnist_regnet400 = models.regnet_y_400mf(pretrained=False)
fashion_mnist_mnas075= models.mnasnet0_75(pretrained=False)
fashion_mnist_mnas05= models.mnasnet0_5(pretrained=False)
fashion_mnist_efficientnet=models.efficientnet_b0(pretrained=False)

fashion_mnist_classes = 10

fashion_mnist_densenet121.classifier = nn.Linear(fashion_mnist_densenet121.classifier.in_features, fashion_mnist_classes, bias=False)
fashion_mnist_resnet18.fc = nn.Linear(fashion_mnist_resnet18.fc.in_features, fashion_mnist_classes, bias=False)
fashion_mnist_lenet.fc = nn.Linear(fashion_mnist_lenet.fc.in_features, fashion_mnist_classes, bias=False)
fashion_mnist_regnet400.fc = nn.Linear(fashion_mnist_regnet400.fc.in_features, fashion_mnist_classes, bias=False)
fashion_mnist_mnas05.classifier[1]=nn.Linear(fashion_mnist_mnas05.classifier[1].in_features, fashion_mnist_classes, bias=False)
fashion_mnist_mnas075.classifier[1]=nn.Linear(fashion_mnist_mnas075.classifier[1].in_features, fashion_mnist_classes, bias=False)
fashion_mnist_efficientnet.classifier[1]=nn.Linear(fashion_mnist_efficientnet.classifier[1].in_features, fashion_mnist_classes, bias=False)
"""SVHN"""

'''
# Caricamento del dataset di addestramento SVHN
svhn_trainset = datasets.SVHN(root='./data', split='train',
                              download=True, transform=transform)

# Caricamento del dataset di test SVHN
svhn_testset = datasets.SVHN(root='./data', split='test',
                             download=True, transform=transform)

svhn_classes = 10
svhn_images_per_class = 3000
svhn_total_images_to_select = svhn_images_per_class * svhn_classes
svhn_sampler = SubsetRandomSampler(torch.randperm(len(svhn_trainset))[:svhn_total_images_to_select])

# DataLoaders per SVHN
svhn_trainloader = DataLoader(svhn_trainset, batch_size=256,
                              sampler=svhn_sampler, num_workers=2, drop_last=True)
            
svhn_testloader = DataLoader(svhn_testset, batch_size=256, num_workers=2, drop_last=True)

svhn_densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
svhn_resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
svhn_regnet = models.regnet_y_800mf(pretrained=False)
svhn_efficientnet=models.efficientnet_b0(pretrained=False)
svhn_lenet = models.googlenet(pretrained=False)

svhn_densenet121.classifier = nn.Linear(svhn_densenet121.classifier.in_features, svhn_classes, bias=False)
svhn_resnet18.fc = nn.Linear(svhn_resnet18.fc.in_features, svhn_classes, bias=False)
svhn_regnet.fc = nn.Linear(svhn_regnet.fc.in_features, svhn_classes, bias=False)
svhn_efficientnet.classifier[1]=nn.Linear(svhn_efficientnet.classifier[1].in_features, svhn_classes, bias=False)
svhn_lenet.fc = nn.Linear(svhn_lenet.fc.in_features, svhn_classes, bias=False)
'''
"""STL-10"""

'''
# Caricamento del dataset di addestramento STL-10
stl10_trainset = datasets.STL10(root='./data', split='train',
                                download=True, transform=transform)

# Caricamento del dataset di test STL-10
stl10_testset = datasets.STL10(root='./data', split='test',
                               download=True, transform=transform)

# DataLoaders per STL-10
stl10_trainloader = DataLoader(stl10_trainset, batch_size=128,
                               shuffle=True, num_workers=2, drop_last=True)
             
                            
stl10_testloader = DataLoader(stl10_testset, batch_size=128,shuffle=True, num_workers=2, drop_last=True)

stl10_densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
stl10_densenet169 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet169', pretrained=False)
stl10_resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
stl10_lenet = models.googlenet(pretrained=False)
stl10_regnet = models.regnet_y_800mf(pretrained=False)

stl10_classes = 10

stl10_densenet121.classifier = nn.Linear(stl10_densenet121.classifier.in_features, stl10_classes, bias=False)
stl10_resnet18.fc = nn.Linear(stl10_resnet18.fc.in_features, stl10_classes, bias=False)
stl10_lenet.fc = nn.Linear(stl10_lenet.fc.in_features, stl10_classes, bias=False)
stl10_regnet.fc = nn.Linear(stl10_regnet.fc.in_features, stl10_classes, bias=False)
'''

#*****************************************************************
def save_feature_F(module, input, output):
    features.get_feature_F(module, input, output)

def save_feature_H(module, input, output):
    features.get_feature_H(module,input,output)


"""
    Calcola la top-k accuracy per un batch di output di un modello.

    Args:
    outputs (torch.Tensor): Tensor contenente le predizioni del modello di dimensione (batch_size, num_classes).
    labels (torch.Tensor): Tensor contenente le etichette di dimensione (batch_size).
    k (int): Numero di top predizioni da considerare per calcolare l'accuratezza.

    Returns:
    float: La top-k accuracy per il batch fornito.
"""
def top_k_accuracy(outputs, labels, k=3):

    # Otteniamo i primi k indici di predizioni più alte per ciascun campione
    top_k_preds = torch.topk(outputs, k, dim=1).indices

    # Verifichiamo se l'etichetta corretta è tra i top k predizioni
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))

    # Calcoliamo la top-k accuracy
    top_k_acc = correct.sum().item() / labels.size(0)

    return top_k_acc

# MEMORIZATION METRIC
def compute_memorization(model, dataloader, mu_c_dict_test):
    global features
    features = FeaturesHook()
    features.clear()
    memorization = 0
    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            inputs, targets = inputs.to(device) , targets.to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()

            # Reset delle liste di features per il prossimo batch
            features.clear()
            # Itera sul tensore delle feature per gestire singoli campioni e aggiornre la media delle classi
            for b, features_sample in enumerate(current_features_F):
              y = targets[b].item()
              distance = torch.dist(features_sample, mu_c_dict_test[y])
              memorization += distance.item()
            
    return memorization/len(dataloader)

"""NEURAL COLLAPSE METRICS"""

'''
collect top-k accuracies,
mean centroid of each class,(mu_Class)
and global mean centroid.
'''

def compute_epoch_info(model, dataloader,eval_loader, optimizer, criterion, num_classes, mode='resnet'):
    global features
    features = FeaturesHook()
    features.clear()

    # Aggiungi l'hook al penultimo layer del modello
    if mode == 'resnet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
        handle_H = model.fc.register_forward_hook(save_feature_H)
    elif mode == 'densenet':
        handle_F = model.features.norm5.register_forward_hook(save_feature_F)
        handle_H = model.classifier.register_forward_hook(save_feature_H)
    elif mode == 'lenet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
        handle_H = model.fc.register_forward_hook(save_feature_H)
    elif mode== 'regnet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
        handle_H = model.fc.register_forward_hook(save_feature_H)
    elif mode== 'efficientnet':
        handle_F = model.classifier[0].register_forward_hook(save_feature_F)
        handle_H = model.classifier[1].register_forward_hook(save_feature_H)
    elif mode == 'mnas':
        handle_F = model.classifier[0].register_forward_hook(save_feature_F)
        handle_H = model.classifier[1].register_forward_hook(save_feature_H)
    else:
        raise ValueError("Mode not supported")

    mu_G = 0
    mu_c_dict = dict()
    mu_c_dict_test= dict()
    counter = dict()
    test_counter= dict()
    top1 = []
    eval_top1 = []

    model.train()
    for idx, (inputs, targets) in enumerate(dataloader):

            inputs, targets = inputs.to(device) , targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            current_features_F = features.features_F[-1].squeeze()
            #current_features_H = features.features_H[-1]

            # Reset delle liste di features per il prossimo batch
            features.clear()

            # Update network
            one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()

            # Itera sul tensore delle feature per gestire singoli campioni e aggiornre la media delle classi
            for b, features_sample in enumerate(current_features_F):
              # mu_G update
              mu_G += features_sample
              #mu_c update
              y = targets[b].item()
              if y not in mu_c_dict:
                mu_c_dict[y] = features_sample
                counter[y] = 1
              else:
                mu_c_dict[y] += features_sample
                counter[y] += 1

            #Compute accuracy
            prec1 = top_k_accuracy(outputs,targets ,1)
            top1.append(prec1)

    # Normalize to obtain final class means and global mean
    mu_G /= dataloader.dataset.data.shape[0]
    for k in mu_c_dict.keys():
          mu_c_dict[k] /= counter[k]  

    # evaluate epoch
    nc4_count=0
    model.eval()     
    for idx, (eval_inputs, eval_targets) in enumerate(eval_loader):
        with torch.no_grad():
            eval_inputs, eval_targets = eval_inputs.to(device) , eval_targets.to(device)
            eval_outputs = model(eval_inputs)
            eval_prec1=top_k_accuracy(eval_outputs,eval_targets ,1)
            eval_top1.append(eval_prec1)

            #collect features an centroids
            eval_features_F = features.features_F[-1].squeeze() #matrice (batch_size,f)
            centroid_tensors = torch.stack([torch.tensor(v) for v in mu_c_dict.values()])  #matrix of c points (c,f)
            features.clear()

            # Calculate Euclidean distances between F and the centroids
            centroid_distances = torch.cdist(eval_features_F, centroid_tensors, p=2) #(batch_size,cdist) tensor of distances
            # Trova l'indice del centroide più vicino per ogni punto F
            closest_centroids_indices = centroid_distances.argmin(dim=1) #(batch size) tensor containing the index of the minimum distance centroid

            # Controlla se la classe reale del campione corrisponde alla classe del centroide più vicino
            for real_class, closest_centroid_index in zip(eval_targets.cpu().numpy(), closest_centroids_indices):
                real_class = int(real_class)  
                closest_centroid_class = list(mu_c_dict.keys())[closest_centroid_index.item()]
                if real_class != closest_centroid_class:
                    nc4_count += 1

            nc4_ratio = nc4_count / len(eval_loader.dataset)

            # perform test centroids for memorization metric
            '''
            for b, features_sample in enumerate(eval_features_F):
              #mu_c test
              y = eval_targets[b].item()
              if y not in mu_c_dict_test:
                mu_c_dict_test[y] = features_sample
                test_counter[y] = 1
              else:
                mu_c_dict_test[y] += features_sample
                test_counter[y] += 1
            '''
    # normalize test centroids            
    #for k in mu_c_dict_test.keys():
    #      mu_c_dict_test[k] /= test_counter[k]

    handle_F.remove()
    handle_H.remove()

    return mu_G, mu_c_dict, sum(top1)/len(top1), sum(eval_top1)/len(eval_top1), nc4_ratio, mu_c_dict_test

"""#### $\textbf{Neural Collapse Properties}$"""

# NC1 (sigmaW)
def compute_Sigma_W(model, mu_c_dict, dataloader, mode='resnet'):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    if mode == 'resnet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
        handle_H = model.fc.register_forward_hook(save_feature_H)
    elif mode == 'densenet':
        handle_F = model.features.norm5.register_forward_hook(save_feature_F)
        handle_H = model.classifier.register_forward_hook(save_feature_H)
    elif mode == 'lenet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
        handle_H = model.fc.register_forward_hook(save_feature_H)
    elif mode== 'regnet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
        handle_H = model.fc.register_forward_hook(save_feature_H)
    elif mode== 'efficientnet':
        handle_F = model.classifier[0].register_forward_hook(save_feature_F)
        handle_H = model.classifier[1].register_forward_hook(save_feature_H)
    elif mode == 'mnas':
        handle_F = model.classifier[0].register_forward_hook(save_feature_F)
        handle_H = model.classifier[1].register_forward_hook(save_feature_H)
    else:
        raise ValueError("Mode not supported")

    Sigma_W = 0
    num_classes = len(mu_c_dict)

    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
          inputs, targets = inputs.to(device) , targets.to(device)
          outputs = model(inputs)

          current_features_F = features.features_F[-1].squeeze()
          current_features_H = features.features_H[-1]

          # Reset delle liste di features per il prossimo batch
          features.features_F = []
          features.features_H = []

        # Itera sul tensore delle feature per gestire singoli campioni e aggiornare la media delle classi
          for b, features_sample in enumerate(current_features_F):
            y = targets[b].item()
            Sigma_W += (features_sample - mu_c_dict[y]).unsqueeze(1) @ (features_sample - mu_c_dict[y]).unsqueeze(0)

    Sigma_W /= dataloader.dataset.data.shape[0]*num_classes

    handle_F.remove()
    handle_H.remove()

    return Sigma_W

#NC1 (SigmaB)
def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    #for key in mu_c_dict.keys():
        #print(mu_c_dict[key].shape)
    for i in range(K):
        # Ensure mu_G is the same shape as mu_c_dict[i] before subtraction
        mu_G_reshaped = mu_G.reshape(mu_c_dict[i].shape)
        Sigma_B += (mu_c_dict[i] - mu_G_reshaped).unsqueeze(1) @ (mu_c_dict[i] - mu_G_reshaped).unsqueeze(0)

    Sigma_B /= K
    return Sigma_B

#NC2
def compute_ETF(mu_c, mu_G, feature_size):
    K = len(mu_c.keys())
    M = torch.empty((K, feature_size))  # Set second param with right size of penultimate feature layer of the model

    # Calcolo delle distanze relative tra centroide di classe e centroide globale
    for key, value in mu_c.items():
        #print(value.shape)
        #print(mu_G.shape)
        M[key] = (value - mu_G) / torch.norm(value - mu_G, dim=-1, keepdim=True)

    MMT= M @ M.t()
    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))) / pow(K - 1, 0.5)

    ETF_metric = torch.norm(MMT - sub, p='fro')

    return ETF_metric.detach().numpy().item()

#NC3
def self_dual_alignment(A, mu_c, mu_G,feature_size):
    K = len(mu_c.keys())
    M = torch.empty((K, feature_size))  # Set second param with right size of penultimate feature layer of the model
    # Calcolo delle distanze relative tra centroide di classe e centroide globale
    for key, value in mu_c.items():
        M[key] = (value - mu_G) / torch.norm(value - mu_G, dim=-1, keepdim=True)

    MT=M.t()
    A = A.to(device)
    MT = MT.to(device)
    AMT_norm= A @ MT / (torch.norm(A @ MT,p='fro') )
    sub = 1 / pow(K - 1, 0.5) * (torch.eye(K) - 1 / K * torch.ones((K, K)))

    nc3 = torch.norm(AMT_norm.to(device) - sub.to(device), p='fro')

    return nc3.cpu().detach().numpy().item()

def show_plot(info_dict, mode, version, dataset_name):
    # Impostare uno stile di base e una palette di colori
    sns.set_style("whitegrid")  # Stile con griglia bianca
    sns.set_palette("Set2")  # Palette di colori accattivante

    # Creare una figura e gli assi per il plot 2x2
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Top-left: Accuracy Top-1
    sns.lineplot(x=np.arange(1, len(info_dict['train_acc']) + 1),
                 y=info_dict['train_acc'],
                 ax=axs[0, 0],
                 marker="o",
                 linewidth=2,
                 color='blue',
                 label='Train Accuracy')  

    sns.lineplot(x=np.arange(1, len(info_dict['eval_acc']) + 1),
             y=info_dict['eval_acc'],
             ax=axs[0, 0],
             marker="o",
             linewidth=2,
             color='orange',
             label='Test Accuracy')
    axs[0, 0].set_title('Accuracy', fontsize=16, fontweight='bold')
    axs[0, 0].set_xlabel('Epochs', fontsize=14)
    axs[0, 0].set_ylabel('Accuracy', fontsize=14)

    

    # center-top: NC1
    sns.lineplot(x=np.arange(1, len(info_dict['NC1']) + 1),
                 y=info_dict['NC1'],
                 ax=axs[0, 1],
                 marker="s",
                 linewidth=2,
                 color='green')  # Cambia il colore e il marker
    axs[0, 1].set_title('NC1 Metric', fontsize=16, fontweight='bold')
    axs[0, 1].set_xlabel('Epochs', fontsize=14)
    axs[0, 1].set_ylabel('Metric Value', fontsize=14)

    # Bottom-right: NC3
    sns.lineplot(x=np.arange(1, len(info_dict['NC2']) + 1),
                 y=info_dict['NC2'],
                 ax=axs[1, 0],
                 marker="^",
                 linewidth=2,
                 color='red',
                 label='NC2')  # Cambia il colore e il marker
    
    sns.lineplot(x=np.arange(1, len(info_dict['NC3']) + 1),
                 y=info_dict['NC3'],
                 ax=axs[1, 0],
                 marker="d",
                 linewidth=2,
                 color='purple',
                 label='NC3')  # Cambia il colore e il marker
    axs[1, 0].set_title('NC2-3 Metric', fontsize=16, fontweight='bold')
    axs[1, 0].set_xlabel('Epochs', fontsize=14)
    axs[1, 0].set_ylabel('Metric Value', fontsize=14)


    # center-low: NC4
    sns.lineplot(x=np.arange(1, len(info_dict['NC4']) + 1),
                 y=info_dict['NC4'],
                 ax=axs[1, 1],
                 marker="d",
                 linewidth=2,
                 color='black')  # Cambia il colore e il marker
    axs[1, 1].set_title('NC4 Metric', fontsize=16, fontweight='bold')
    axs[1, 1].set_xlabel('Epochs', fontsize=14)
    axs[1, 1].set_ylabel('Metric Value', fontsize=14)
    '''
    # upper-right: memorization
    sns.lineplot(x=np.arange(1, len(info_dict['mem']) + 1),
                 y=info_dict['mem'],
                 ax=axs[1, 2],
                 marker="d",
                 linewidth=2,
                 color='yellow')  # Cambia il colore e il marker
    axs[1, 2].set_title('Memorization Metric', fontsize=16, fontweight='bold')
    axs[1, 2].set_xlabel('Epochs', fontsize=14)
    axs[1, 2].set_ylabel('Metric Value', fontsize=14)
    '''

    # Aggiustiamo lo spazio tra i plot per una migliore visualizzazione
    plt.tight_layout()
    plt.show()
    # Salvare il grafico in un file PNG
    plt.savefig(f'epoch_{len(info_dict["train_acc"])}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

"""Train model and Store Results"""

info_dict = {
        #'A': [],
        #'b': [],
        #'H': [],
        'NC1': [],
        'NC2': [],
        'NC3': [],
        'NC4': [],
        #'mem': [],
        #'mu_G_train': [],
        'train_acc': [],
        'eval_acc': [],
        #'mu_G_test': [],
        #'test_acc1': [],
        #'test_acc3': []
}

##########################################################################

#define dataset, model, its version
dataset_name='fashion_mnist'
mode='densenet'
version='121'

#define model and relative datset to train and collapse
model = fashion_mnist_densenet121
num_classes = fashion_mnist_classes
feature_size =  densenet_feature_size

#flags
resnet=False
densenet=True
lenet=False
regnet=False
efficientnet=False
mnas=False

#dataset
trainloader = fashion_mnist_trainloader
testloader = fashion_mnist_testloader

#model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model = model.to(device)
############################################################################
#define parameters
epochs=50

# MSE+WD and low LR 
criterion= nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.00001, weight_decay=5e-4, betas=(0.9,0.999))

'''
#Laod weights
state_dict = torch.load(folder_path +f'/epoch_N_{mode}{version}_{dataset_name}_weights.pth')
model.load_state_dict(state_dict)

# Load the dictionary from the pickle file
with open(folder_path + f'/{mode}{version}_{dataset_name}_results.pkl', 'rb') as pkl_file:
    info_dict = pickle.load(pkl_file)
'''

for i in range(epochs):
    #TRAIN MODEL at current epoch

    #Evaluate NC properties at current epoch
    mu_G_train, mu_c_dict_train, train_acc, eval_acc, nc4, mu_c_dict_test = compute_epoch_info( model, trainloader, testloader, optimizer, criterion, num_classes, mode=mode)
    #mu_G_test, mu_c_dict_test, test_acc1, test_acc3 = compute_epoch_info( model, testloader, isTrain=False, mode='ResNet')

    #NC1
    Sigma_W = compute_Sigma_W( model, mu_c_dict_train, trainloader, mode=mode)
    Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)
    collapse_metric = float(np.trace(Sigma_W.cpu() @ scilin.pinv(Sigma_B.cpu().numpy())) / len(mu_c_dict_train.keys()))

    #NC2
    if resnet:
      A = model.fc.weight
    elif densenet:
      A = model.classifier.weight
    elif lenet:
      A = model.fc.weight
    elif regnet:
      A = model.fc.weight
    elif efficientnet:
      A = model.classifier[1].weight
    elif mnas:
      A = model.classifier[1].weight
    else:
      break

    ETF_metric = compute_ETF(mu_c_dict_train, mu_G_train,feature_size)

    #NC3
    alignment= self_dual_alignment(A, mu_c_dict_train, mu_G_train, feature_size)

    # MEMORIZATION

    #memorization= compute_memorization(trainloader, mu_c_dict_test)

################################################################################
    #Store NC properties
    info_dict['NC1'].append(collapse_metric)
    info_dict['NC2'].append(ETF_metric)
    info_dict['NC3'].append(alignment)
    info_dict['NC4'].append(nc4)
    #info_dict['mem'].append(memorization)
    # Store penultimate features weights values
    #info_dict['A'].append((A.detach().numpy()))
    #info_dict['H'].append(H.detach().numpy())

    #info_dict['mu_G_train'].append(mu_G_train.detach().numpy())
    # info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())

    #Store accuracies
    info_dict['train_acc'].append(train_acc)
    info_dict['eval_acc'].append(eval_acc)
    #info_dict['test_acc1'].append(test_acc1)
    #info_dict['test_acc5'].append(test_acc3)

    print('[epoch:'+ str(i + 1) +' | train top1:' + str(train_acc) +' | eval acc:' + str(eval_acc) +' | NC1:' + str(collapse_metric)+' | NC2:'+ str(ETF_metric)+' | NC3:'+ str(alignment)+' | NC4:'+str(nc4))
    
    if (i+1) % epochs == 0:
        show_plot(info_dict, mode, version, dataset_name)
    
    torch.save(model.state_dict(), folder_path + f'/no_noise/{dataset_name}/{mode}{version}/epoch_{i+1}_{mode}{version}_{dataset_name}_weights.pth')

    with open(folder_path + f'/no_noise/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_results.pkl', 'wb') as f:
        pickle.dump(info_dict, f)

exit(0)
