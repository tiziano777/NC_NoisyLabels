import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Hooks import FeaturesHook
from sample_tracker import SampleTracker

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg as scilin

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_cuda_devices():
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices available: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
            print(f"  - Memory Cached: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
            print(f"  - Total Memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
            print()
    else:
        print("No CUDA devices available.")

print_cuda_devices()

import warnings
warnings.filterwarnings('ignore')

import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def save_feature_F(module, input, output):
    features.get_feature_F(module, input, output)

# Aggiungi l'hook al penultimo layer del modello
def select_model(mode):
    handle_F=None
    if mode == 'resnet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
    elif mode == 'densenet':
        handle_F = model.features.norm5.register_forward_hook(save_feature_F)
    elif mode == 'lenet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
    elif mode== 'regnet':
        handle_F = model.avgpool.register_forward_hook(save_feature_F)
    elif mode== 'efficientnet':
        handle_F = model.classifier[0].register_forward_hook(save_feature_F)
    elif mode == 'mnas':
        handle_F = model.classifier[0].register_forward_hook(save_feature_F)
    else:
        raise ValueError("Mode not supported")

    return handle_F

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


num_samples = len(cifar10_trainset)
indices = np.arange(num_samples)
np.random.shuffle(indices)
shuffled_train_dataset= Subset(cifar10_trainset,indices)

#DataLoaders
cifar10_trainloader = torch.utils.data.DataLoader(shuffled_train_dataset, batch_size=128,
                                          shuffle=False, num_workers=2, drop_last=True)


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
'''
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
'''
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

# computes class centroids
def compute_mu_c(features, labels, num_classes):
    # Inizializza i tensori di output
    mu_c = torch.zeros((num_classes, features.size(1)), dtype=features.dtype, device=device)
    counter = torch.zeros(num_classes, dtype=torch.int32, device=device)
    
    # Itera su ogni classe e somma le feature corrispondenti
    for c in range(num_classes):
        mask = (labels == c)
        mu_c[c] = features[mask].sum(dim=0)
        counter[c] = mask.sum()
    
    return mu_c, counter

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
@measure_time
def compute_epoch_info(model, dataloader,eval_loader, optimizer, criterion, num_classes, feature_size, mode='resnet'):
    global features
    features = FeaturesHook()
    features.clear()
    handle_F= select_model(mode)

    mu_G = 0

    mu_c = torch.zeros((num_classes, feature_size), device=device)
    mu_c_test = torch.zeros((num_classes, feature_size), device=device)
    
    counter = torch.zeros(num_classes, dtype=torch.int32, device=device)
    test_counter= torch.zeros(num_classes, dtype=torch.int32, device=device)

    top1 = []
    eval_top1 = []

    model.train()
    for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = labels.to(device) # take corrupted labels (training labels)
            optimizer.zero_grad()
            outputs = model(inputs).logits

            current_features_F = features.features_F[-1].squeeze()
            # Reset delle liste di features per il prossimo batch
            features.clear()

            # Update network (one hot function or label smoothing function)
            one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
            #one_hot_targets = label_smoothing(targets, num_classes=num_classes).float()
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()

            # Aggiorna mu_G
            mu_G += current_features_F.sum(dim=0)

            mu_c_batch, counter_batch= compute_mu_c(current_features_F, targets, num_classes)
            mu_c += mu_c_batch
            counter += counter_batch

            #Compute accuracy
            prec1 = top_k_accuracy(outputs,targets ,1)
            top1.append(prec1)

    # Normalize to obtain final class means and global mean
    mu_G /= len(dataloader.dataset)
    for k in range(len(mu_c)):
          mu_c[k] /= counter[k] 

    # evaluate epoch
    nc4_count=0
    model.eval()     
    for idx, (eval_inputs, eval_targets) in enumerate(eval_loader):
        with torch.no_grad():
            eval_inputs = eval_inputs.to(device)
            eval_targets = eval_targets.to(device)
            eval_outputs = model(eval_inputs)
            eval_prec1=top_k_accuracy(eval_outputs,eval_targets ,1)
            eval_top1.append(eval_prec1)

            #collect features an centroids
            eval_features_F = features.features_F[-1].squeeze() #matrice (batch_size,f)
            features.clear()

            # Calculate Euclidean distances between F and the centroids
            centroid_distances = torch.cdist(eval_features_F, mu_c, p=2) #(batch_size,cdist) tensor of distances
            # Trova l'indice del centroide più vicino per ogni punto F
            closest_centroids_indices = centroid_distances.argmin(dim=1) #(batch size) tensor containing the index of the minimum distance centroid

            # Controlla se la classe reale del campione corrisponde alla classe del centroide più vicino
            for real_class, closest_centroid_index in zip(eval_targets.cpu().numpy(), closest_centroids_indices):
                real_class = int(real_class)  
                closest_centroid_class = closest_centroid_index.item()  # L'indice rappresenta direttamente la classe del centroide
                if real_class != closest_centroid_class:
                    nc4_count += 1

            nc4_ratio = nc4_count / len(eval_loader.dataset)

            # perform test centroids for memorization metric
            mu_c_test_batch, test_counter_batch= compute_mu_c(eval_features_F, eval_targets, num_classes)

            mu_c_test += mu_c_test_batch
            test_counter += test_counter_batch
            
    # normalize test centroids            
    for k in range(len(mu_c_test)):
        mu_c_test[k] /= test_counter[k]

    handle_F.remove()
    
    return mu_G, mu_c, sum(top1)/len(top1), sum(eval_top1)/len(eval_top1), nc4_ratio, mu_c_test

"""Neural Collapse Properties and personalized metrics"""

def compute_delta_distance(mu_c_tensor,features_batch,targets,batch_size):
    distances = torch.cdist(features_batch, mu_c_tensor)  # calcola tensore delle distanze [B,classes]
    centroid_distance = distances[torch.arange(batch_size), targets]  # take distance from label class centoroid
    mask = torch.ones(distances.shape, dtype=bool)
    mask[torch.arange(batch_size), targets] = False  # create a mask for deleteing extracted label centroid distances
    d_remaining = distances[mask].view(batch_size, num_classes-1) # take the rest of distances
    d_min, _ = torch.min(d_remaining, dim=1)  # find the minimum distance centroid from the remaining ones
            
    distance_batch= d_min - centroid_distance

    return distance_batch

# NC1 (sigmaW)
@measure_time
def compute_Sigma_W_and_distance(model, mu_c_tensor, dataloader,delta_distance_tracker,distance_tracker, mode='resnet'):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)
    Sigma_W = 0
    num_classes = len(mu_c_tensor)

    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            current_features_F = features.features_F[-1].squeeze()
            batch_size=len(current_features_F)

            # Reset delle liste di features per il prossimo batch
            features.features_F = []
            features.clear()

            #Sigma_W
            target_indices = targets.long()
            mu_c_batch = mu_c_tensor[target_indices].to(device)
            diff = current_features_F - mu_c_batch
            Sigma_W += (diff.unsqueeze(2) @ diff.unsqueeze(1)).sum(dim=0)

            #Relative Distance
            delta_distance_tracker.add_epoch_results(idx,compute_delta_distance(mu_c_tensor,current_features_F,targets, batch_size)) 

            #distance from label class centroid
            distances = torch.cdist(current_features_F, mu_c_tensor)  # calcola tensore delle distanze [B,classes]
            centroid_distance = distances[torch.arange(batch_size), targets]  # take distance from label class centroid
            distance_tracker.add_epoch_results(idx,centroid_distance)

            del inputs, outputs, current_features_F

    Sigma_W /= (len(dataloader.dataset)*num_classes)
    handle_F.remove()

    return Sigma_W

#NC1 (SigmaB)
def compute_Sigma_B(mu_c, mu_G):
    Sigma_B = 0
    K = len(mu_c)
    #for key in mu_c_dict.keys():
        #print(mu_c_dict[key].shape)
    for i in range(K):
        # Ensure mu_G is the same shape as mu_c_dict[i] before subtraction
        mu_G_reshaped = mu_G.reshape(mu_c[i].shape)
        Sigma_B += (mu_c[i] - mu_G_reshaped).unsqueeze(1) @ (mu_c[i] - mu_G_reshaped).unsqueeze(0)

    Sigma_B /= K
    return Sigma_B

#NC2
def compute_ETF(mu_c, mu_G, feature_size):
    K = len(mu_c)
    M = torch.empty((K, feature_size))  # Set second param with right size of penultimate feature layer of the model

    # Calcolo delle distanze relative tra centroide di classe e centroide globale
    for i in range(len(mu_c)):
        #print(value.shape)
        #print(mu_G.shape)
        M[i] = (mu_c[i] - mu_G) / torch.norm(mu_c[i] - mu_G, dim=-1, keepdim=True)

    MMT= M @ M.t()
    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))) / pow(K - 1, 0.5)

    ETF_metric = torch.norm(MMT - sub, p='fro')

    return ETF_metric.detach().numpy().item()

#NC3
def self_dual_alignment(A, mu_c, mu_G,feature_size):
    K = len(mu_c)
    M = torch.empty((K, feature_size))  # Set second param with right size of penultimate feature layer of the model
    # Calcolo delle distanze relative tra centroide di classe e centroide globale
    for i in range(len(mu_c)):
        M[i] = (mu_c[i] - mu_G) / torch.norm(mu_c[i] - mu_G, dim=-1, keepdim=True)

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
    # Salvare il grafico in un file PNG
    plt.savefig(f'epoch_{len(info_dict["train_acc"])}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def show_mean_var_relevations(tensor, mode, version, dataset_name, dict_type='delta_distance'):
    # Calcola le medie e le varianze per ogni riga (N,)
    means = tensor.mean(dim=1).cpu().numpy()
    variances = tensor.var(dim=1).cpu().numpy()


    # Impostazioni di Seaborn per un grafico più accattivante
    sns.set_style("whitegrid")
    sns.set_palette('deep')

    # Crea la figura e gli assi (3x3 layout)
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    # Prima colonna: Tutti i campioni
    sns.histplot(means, bins=30, ax=axes[0], color='skyblue')
    axes[0].set_title('All Samples: Histogram of Means')
    axes[0].set_xlabel('Mean')
    axes[0].set_ylabel('Count')

    sns.histplot(variances, bins=30, ax=axes[1], color='lightcoral')
    axes[1].set_title('All Samples: Histogram of Variances')
    axes[1].set_xlabel('Variance')
    axes[1].set_ylabel('Count')

    # Scatter plot con separazione dei campioni rumorosi
    sns.scatterplot(x=means, y=variances, ax=axes[2], color='green', label='Clean Samples')
    sns.kdeplot(x=means, y=variances, ax=axes[2], cmap="Blues_r", fill=True, alpha=0.5)
    axes[2].set_title('All Samples: Scatter plot of Means vs Variances')
    axes[2].set_xlabel('Mean')
    axes[2].set_ylabel('Variance')
    axes[2].legend()

    plt.tight_layout()
    # Salvare il grafico in un file PNG
    plt.savefig(f'mean_variance_epoch_{len(tensor[0])}_{dict_type}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

"""Train model and Store Results"""

info_dict = {
        'NC1': [],
        'NC2': [],
        'NC3': [],
        'NC4': [],
        'train_acc': [],
        'eval_acc': [],
        'mu_c': [],
}

##########################################################################

#define dataset, model, its version
dataset_name='cifar10'
mode='lenet'
version=''

#define model and relative dataset to train and collapse
model = cifar10_lenet
num_classes = cifar10_classes
feature_size =  lenet_feature_size

#flags
resnet=False
densenet=False
lenet=True
regnet=False
efficientnet=False
mnas=False

#datasets
train_dataset = shuffled_train_dataset
test_dataset = cifar10_testset 
trainloader = cifar10_trainloader
testloader = cifar10_testloader

#model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.to(device)
############################################################################
#define parameters
epochs=40

# MSE+WD and low LR 
criterion= nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001, betas=(0.9,0.999)) #, weight_decay=5e-4

'''
#Laod weights
state_dict = torch.load(folder_path +f'/epoch_N_{mode}{version}_{dataset_name}_weights.pth')
model.load_state_dict(state_dict)

# Load the dictionary from the pickle file
with open(folder_path + f'/{mode}{version}_{dataset_name}_results.pkl', 'rb') as pkl_file:
    info_dict = pickle.load(pkl_file)
'''

delta_distance_tracker=SampleTracker()
distance_tracker=SampleTracker()

for i in range(epochs):
    #TRAIN MODEL at current epoch

    #Evaluate NC properties at current epoch
    mu_G_train, mu_c_train, train_acc, eval_acc, nc4, mu_c_test = compute_epoch_info( model, trainloader, testloader, optimizer, criterion, num_classes, feature_size, mode=mode)

    #NC1 and stability metrics( f,d and delta_d are vectors of dimension N to append into a dict that store it as epoch result )
    Sigma_W = compute_Sigma_W_and_distance( model, mu_c_train, trainloader, delta_distance_tracker, distance_tracker, mode=mode)
    Sigma_B = compute_Sigma_B(mu_c_train, mu_G_train)
    collapse_metric = float(np.trace(Sigma_W.cpu() @ scilin.pinv(Sigma_B.cpu().numpy())) / len(mu_c_train))

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

    ETF_metric = compute_ETF(mu_c_train, mu_G_train,feature_size)

    #NC3
    alignment= self_dual_alignment(A, mu_c_train, mu_G_train, feature_size)

    ################################################################################
    #Store NC properties
    info_dict['NC1'].append(collapse_metric)
    info_dict['NC2'].append(ETF_metric)
    info_dict['NC3'].append(alignment)
    info_dict['NC4'].append(nc4)
    info_dict['mu_c'].append(mu_c_train)

    #Store accuracies
    info_dict['train_acc'].append(train_acc)
    info_dict['eval_acc'].append(eval_acc)



    if (i+1) % 5 == 0:
        show_plot(info_dict, mode, version, dataset_name) 

        delta_distances= delta_distance_tracker.tensorize()
        #distances= distance_tracker.tensorize()

        show_mean_var_relevations(delta_distances, mode, version, dataset_name, dict_type='delta_distance')
        del delta_distances
        
        '''
        if (i+1) % epochs == 0:
            visualizer = EmbeddingVisualizer2D(model,feature_size, mode, version, dataset_name, (i+1), device, trainloader, cifar10_noisy_trainset.corrupted_indices, mu_c_train, mu_c_weighted, mu_c_clean)
            visualizer.run()
        '''

        torch.save(model.state_dict(), folder_path + f'/no_noise/{dataset_name}/{mode}{version}/epoch_{i+1}_{mode}{version}_{dataset_name}_weights.pth')

        with open(folder_path + f'/no_noise/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_results.pkl', 'wb') as f:
            pickle.dump(info_dict, f)


    print(f'[epoch:{i + 1} | train top1:{train_acc:.4f} | eval acc:{eval_acc:.4f} | NC1:{collapse_metric:.4f} | NC4:{nc4:.4f} ]')
