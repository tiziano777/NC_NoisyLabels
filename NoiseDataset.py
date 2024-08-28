import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Hooks import FeaturesHook
from PIL import Image

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


# Definizione del tasso di rumore
noise_rate = 0.10

# Class to store data with only fake labels
class NoiseDataset(Dataset):
    def __init__(self, data, fake_labels):
        self.data = data
        self.fake_labels = fake_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        fake_label = self.fake_labels[idx]
        return data, fake_label

#class to store data with fake/clean labels
class NoisyLabelsDataset(Dataset):
    def __init__(self, dataset, real_labels, fake_labels):
        self.dataset = dataset
        
        # Inizializziamo le etichette reali e fake con le etichette originali del dataset
        self.real_labels = real_labels
        self.fake_labels = fake_labels

    
    def __getitem__(self, index):
        x = self.dataset[index]  # Ignoriamo l'etichetta reale dal dataset originale
        real_y = self.real_labels[index]  # Etichetta reale
        fake_y = self.fake_labels[index]  # Etichetta corrotta (fake)
        
        # Concateniamo verticalmente l'etichetta reale e quella fake per formare un array di output
        y_output = np.vstack((real_y, fake_y))
        
        return x, y_output

    def get_fake_labels(self):
        return self.real_labels
    
    def get_clean_labels(self):
        return self.fake_labels
    
    def __len__(self):
        return len(self.dataset)
    
# Class to manage both clean/fake labels   
class NoisyLabelsDatasetManager(Dataset):
    def __init__(self, dataset, noise_rate, random_state=1):
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        # Inizializziamo le etichette reali e fake con le etichette originali del dataset
        self.real_labels = np.array([self.dataset[i][1] for i in range(len(dataset))])
        self.fake_labels = self.real_labels.copy()
        
        # Applichiamo il rumore alle etichette fake
        self.inject_noise()

    def inject_noise(self):
        num_samples = len(self.dataset)
        num_corrupted = int(self.noise_rate * num_samples)
        
        corrupted_indices = np.random.choice(num_samples, size=num_corrupted, replace=False)
        
        for idx in corrupted_indices:
            # Generiamo un'etichetta fake per 10 classi
            noisy_label = np.random.randint(0, 10)
            self.fake_labels[idx] = noisy_label

    def __getitem__(self, index):
        x, _ = self.dataset[index]  # Ignoriamo l'etichetta reale dal dataset originale
        real_y = self.real_labels[index]  # Etichetta reale
        fake_y = self.fake_labels[index]  # Etichetta corrotta (fake)
        
        # Concateniamo verticalmente l'etichetta reale e quella fake per formare un array di output
        y_output = np.vstack((real_y, fake_y))
        
        return x, y_output

    def get_fake_labels(self):
        return self.real_labels
    
    def get_clean_labels(self):
        return self.fake_labels
    
    def take_first_n_noise_samples(self, n):
        # Calcola gli indici dei campioni in cui l'etichetta reale non corrisponde all'etichetta falsa
        matching_indices = np.where(self.real_labels != self.fake_labels)[0][:n]
        
        # Prepara i dati e le etichette per i campioni selezionati
        selected_data = [self.dataset[i][0] for i in matching_indices]
        selected_real_labels = self.real_labels[matching_indices]
        selected_fake_labels = self.fake_labels[matching_indices]
        
        # Converte le etichette in tensori di PyTorch
        selected_real_labels_tensor = torch.tensor(selected_real_labels, dtype=torch.long)
        selected_fake_labels_tensor = torch.tensor(selected_fake_labels, dtype=torch.long)
        
        # Crea un nuovo dataset con i campioni selezionati e le etichette concatenate
        selected_dataset = NoisyLabelsDataset(dataset=selected_data, real_labels=selected_real_labels_tensor, fake_labels=selected_fake_labels_tensor)
        
        return selected_dataset
    
    def take_first_n_clean_samples(self, n):
        # Calcola gli indici dei campioni in cui l'etichetta reale non corrisponde all'etichetta falsa
        matching_indices = np.where(self.real_labels == self.fake_labels)[0][:n]
        
        # Prepara i dati e le etichette per i campioni selezionati
        selected_data = [self.dataset[i][0] for i in matching_indices]
        selected_real_labels = self.real_labels[matching_indices]
        selected_fake_labels = self.fake_labels[matching_indices]
        
        # Converte le etichette in tensori di PyTorch
        selected_real_labels_tensor = torch.tensor(selected_real_labels, dtype=torch.long)
        selected_fake_labels_tensor = torch.tensor(selected_fake_labels, dtype=torch.long)
        
        # Crea un nuovo dataset con i campioni selezionati e le etichette concatenate
        selected_dataset = NoisyLabelsDataset(dataset=selected_data, real_labels=selected_real_labels_tensor, fake_labels=selected_fake_labels_tensor)
        
        return selected_dataset

    def divide_samples(self):
        # Split the dataset into two: one where real_label == fake_label and another where they don't match
        matching_indices = np.where(self.real_labels == self.fake_labels)[0]
        non_matching_indices = np.where(self.real_labels != self.fake_labels)[0]
        
        # Extract data and fake labels for matching and non-matching cases
        matching_data = [self.dataset[i][0] for i in matching_indices]
        matching_fake_labels = self.fake_labels[matching_indices]
        
        non_matching_data = [self.dataset[i][0] for i in non_matching_indices]
        non_matching_fake_labels = self.fake_labels[non_matching_indices]
        
        # Create combined datasets including data and fake labels
        clean_samples = NoiseDataset(matching_data, matching_fake_labels)
        noisy_labels_samples = NoiseDataset(non_matching_data, non_matching_fake_labels)
        
        return clean_samples, noisy_labels_samples

    def __len__(self):
        return len(self.dataset)
    

"""CIFAR-10"""

# Caricamento clean datasets
cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



# Creazione degli oggetti NoisyLabelsDataset
cifar10_noisy_trainset = NoisyLabelsDatasetManager(cifar10_trainset, noise_rate=noise_rate)

# Configurazione del DataLoader
cifar10_trainloader = DataLoader(cifar10_noisy_trainset, batch_size=256, shuffle=True, num_workers=2, drop_last=True)
cifar10_testloader = DataLoader(cifar10_testset, batch_size=256, num_workers=2, drop_last=True)

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

##################################################################################################################################################
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

# MEMORIZATION METRICS
def compute_memorization(model, dataloader, mu_c_dict_test):
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

    memorization = 0
    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()

            # Reset delle liste di features per il prossimo batch
            features.clear()
            # Itera sul tensore delle feature per gestire singoli campioni e aggiornare la media delle classi
            for b, features_sample in enumerate(current_features_F):
              y = targets[b].item()
              distance = torch.dist(features_sample, mu_c_dict_test[y])
              memorization += distance.item()
            
    return memorization/len(dataloader)

def compute_total_memorization(model, dataloader, mu_c_dict_test):
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

    memorization = 0
    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets[:,1:,:].squeeze().to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()

            # Reset delle liste di features per il prossimo batch
            features.clear()
            # Itera sul tensore delle feature per gestire singoli campioni e aggiornare la media delle classi
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

            inputs = inputs.to(device)
            targets = targets[:,1:,:].squeeze().to(device) # take only corrupted labels
            optimizer.zero_grad()
            outputs = model(inputs)

            current_features_F = features.features_F[-1].squeeze()
            #current_features_H = features.features_H[-1]

            # Reset delle liste di features per il prossimo batch
            features.clear()

            # Update network
            one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
            loss = criterion(outputs.logits, one_hot_targets)
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
            prec1 = top_k_accuracy(outputs.logits,targets ,1)
            top1.append(prec1)

    # Normalize to obtain final class means and global mean
    mu_G /= len(dataloader.dataset)
    for k in mu_c_dict.keys():
          mu_c_dict[k] /= counter[k]  

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
            for b, features_sample in enumerate(eval_features_F):
              #mu_c test
              y = eval_targets[b].item()
              if y not in mu_c_dict_test:
                mu_c_dict_test[y] = features_sample
                test_counter[y] = 1
              else:
                mu_c_dict_test[y] += features_sample
                test_counter[y] += 1
            
    # normalize test centroids            
    for k in mu_c_dict_test.keys():
        mu_c_dict_test[k] /= test_counter[k]

    handle_F.remove()
    handle_H.remove()

    return mu_G, mu_c_dict, sum(top1)/len(top1), sum(eval_top1)/len(eval_top1), nc4_ratio, mu_c_dict_test

"""#### $\textbf{Neural Collapse Properties}$"""

# NC1 (sigmaW)
def compute_Sigma_W_and_force(model, mu_c_dict, dataloader, mode='resnet'):
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

    force=[]
    distance=[]
    delta_distance=[]

    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
          inputs = inputs.to(device)
          targets = targets[:,1:,:].squeeze().to(device)
          outputs = model(inputs)

          current_features_F = features.features_F[-1].squeeze()
          #current_features_H = features.features_H[-1]

          # Reset delle liste di features per il prossimo batch
          features.features_F = []
          features.features_H = []

        # Itera sul tensore delle feature per gestire singoli campioni e aggiornare la media delle classi e la forza
          for b, features_sample in enumerate(current_features_F):
            y = targets[b].item()
            Sigma_W += (features_sample - mu_c_dict[y]).unsqueeze(1) @ (features_sample - mu_c_dict[y]).unsqueeze(0)

            # compute distance and force metric to evaluate sample behavior in the feature space respect their label centorid.
            force.append( (len(dataloader)/len(mu_c_dict.keys()))  * (1/(torch.abs(features_sample - mu_c_dict[y]))**2) * ( (mu_c_dict[y] - features_sample ) / (torch.abs(mu_c_dict[y] - features_sample )) ) )
            distance.append((torch.abs(features_sample - mu_c_dict[y])))

            class_indices=list(range(10))
            del class_indices[y]
            dist=dict()
            for i in class_indices:
                dist[i]=torch.dist(features_sample,mu_c_dict[i]).item()
            near_centroid_dist=min(dist, key=lambda x: dist[x])
            delta_distance.append(distance[-1] - near_centroid_dist)

    Sigma_W /= (len(dataloader.dataset)*num_classes)

    handle_F.remove()
    handle_H.remove()

    return Sigma_W, force, distance, delta_distance

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

# Track n samples and show relative distance between its real centroid and its fake centroid
def get_distance_n_noisy_saples(n,model,train_dataset, mu_c_dict_train):
    '''
    if distance is positive, sample is more near to its real centroid than its fake centorid...
    else it is the contrary, and the sample became far away from its real classification centroid.
    '''
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
    
    # Take first n noisy samples from trainining dataset
    tracked_noise_samples= train_dataset.take_first_n_noise_samples(n)
    tracked_noise_samples_loader=DataLoader(tracked_noise_samples, batch_size=1, num_workers=2)

    # foreach sample take its features and compute relative distance ( real - fake )
    distance_dict=dict()
    model.eval()
    for idx, (inputs, targets) in enumerate(tracked_noise_samples_loader):
        with torch.no_grad():
            y_real = targets[:,:1,:].squeeze().to(device)
            y_fake = targets[:,1:,:].squeeze().to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()
            distance_dict[str(idx)]= torch.dist(current_features_F,mu_c_dict_train[y_real.item()]).item() - torch.dist(current_features_F,mu_c_dict_train[y_fake.item()]).item()
            features.clear()

    return distance_dict

# Track n samples and show relative distance between its real centroid and the second nearest or avg other cetroid
def get_distance_n_saples(n,model,train_dataset, mu_c_dict_train):
    '''
    if distance is positive, sample is more near to its real centroid than other centorid...
    else it is the contrary, and the sample became far away from its real classification centroid.
    '''
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
    
    # Take first n noisy samples from trainining dataset
    tracked_clean_samples= train_dataset.take_first_n_clean_samples(n)
    tracked_clean_samples_loader=DataLoader(tracked_clean_samples, batch_size=1, num_workers=2)

    # foreach sample take its features and compute relative distance ( real - fake )
    distance_dict=dict()
    model.eval()
    for idx, (inputs, targets) in enumerate(tracked_clean_samples_loader):
        with torch.no_grad():
            y_real = targets[:,:1,:].squeeze().to(device)
            y_fake = targets[:,1:,:].squeeze().to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()
            class_indices=list(range(10))
            del class_indices[y_real.item()]
            dist=dict()
            for i in class_indices:
                dist[i]=torch.dist(current_features_F,mu_c_dict_train[i]).item()
            near_centroid_dist=min(dist, key=lambda x: dist[x])

            distance_dict[str(idx)]= near_centroid_dist - torch.dist(current_features_F,mu_c_dict_train[y_real.item()]).item() 
            features.clear()

    return distance_dict


def show_plot(info_dict, mode, version, dataset_name):
    # Impostare uno stile di base e una palette di colori
    sns.set_style("whitegrid")  # Stile con griglia bianca
    sns.set_palette("Set2")  # Palette di colori accattivante

    # Creare una figura e gli assi per il plot 2x2
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

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
    
    # upper-right: memorization
    sns.lineplot(x=np.arange(1, len(info_dict['mem']) + 1),
                 y=info_dict['mem'],
                 ax=axs[1, 2],
                 marker="d",
                 linewidth=2,
                 color='yellow',
                 label='total memo.')  # Cambia il colore e il marker
    '''
    sns.lineplot(x=np.arange(1, len(info_dict['clean_mem']) + 1),
                 y=info_dict['clean_mem'],
                 ax=axs[1, 2],
                 marker="d",
                 linewidth=2,
                 color='orange',
                 label='clean labels memo.')  # Cambia il colore e il marker
    sns.lineplot(x=np.arange(1, len(info_dict['fake_mem']) + 1),
                 y=info_dict['fake_mem'],
                 ax=axs[1, 2],
                 marker="d",
                 linewidth=2,
                 color='violet',
                 label='noisy labels memo.')  # Cambia il colore e il marker
    axs[1, 2].set_title('Memorization Metrics', fontsize=16, fontweight='bold')
    axs[1, 2].set_xlabel('Epochs', fontsize=14)
    axs[1, 2].set_ylabel('Metric Value', fontsize=14)
    '''
    # Aggiustiamo lo spazio tra i plot per una migliore visualizzazione
    plt.tight_layout()
    #plt.show()
    # Salvare il grafico in un file PNG
    plt.savefig(f'epoch_{len(info_dict["train_acc"])}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def show_noise_samples(track_samples,mode,version,dataset_name):
    # Impostare uno stile di base e una palette di colori
    sns.set_style("whitegrid")  # Stile con griglia bianca

    # Creare una figura e gli assi per il plot 3x4
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))

    custom_palette = sns.color_palette(["#FF0B04", "#4374B3", "#008000", "#FFFF00", "#800080", "#808000",
                                    "#0000FF", "#A52A2A", "#FFD700", "#90EE90", "#F08080", "#ADD8E6"])
    
    row=3
    col=4
    sample_id=0
    for r in range(row):
        for c in range(col):
            sns.lineplot(x=np.arange(1, len(track_samples[str(sample_id)]) + 1),
                y=track_samples[str(sample_id)],
                ax=axs[r, c],
                marker="d",
                linewidth=2,
                color=custom_palette[sample_id],
                ) 
            axs[r, c].set_title(f'Noisy sample {sample_id}', fontsize=16, fontweight='bold')
            axs[r, c].set_xlabel('Epochs', fontsize=14)
            axs[r, c].set_ylabel('relative distances', fontsize=14)
            sample_id+=1

    # Aggiustiamo lo spazio tra i plot per una migliore visualizzazione
    plt.tight_layout()
    # Salvare il grafico in un file PNG
    plt.savefig(f'noise{int(noise_rate*100)}_samples_epoch_{len(track_samples[str(0)])}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def show_clean_samples(track_samples,mode,version,dataset_name):
    # Impostare uno stile di base e una palette di colori
    sns.set_style("whitegrid")  # Stile con griglia bianca

    # Creare una figura e gli assi per il plot 3x4
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))

    custom_palette = sns.color_palette(["#FF0B04", "#4374B3", "#008000", "#FFFF00", "#800080", "#808000",
                                    "#0000FF", "#A52A2A", "#FFD700", "#90EE90", "#F08080", "#ADD8E6"])
    
    row=3
    col=4
    sample_id=0
    for r in range(row):
        for c in range(col):
            sns.lineplot(x=np.arange(1, len(track_samples[str(sample_id)]) + 1),
                y=track_samples[str(sample_id)],
                ax=axs[r, c],
                marker="d",
                linewidth=2,
                color=custom_palette[sample_id],
                ) 
            axs[r, c].set_title(f'Clean sample {sample_id}', fontsize=16, fontweight='bold')
            axs[r, c].set_xlabel('Epochs', fontsize=14)
            axs[r, c].set_ylabel('relative distances', fontsize=14)
            sample_id+=1

    # Aggiustiamo lo spazio tra i plot per una migliore visualizzazione
    plt.tight_layout()
    # Salvare il grafico in un file PNG
    plt.savefig(f'clean_samples_epoch_{len(track_samples[str(0)])}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def show_sample_stability(media_varianza_dizionario,mode,version,dataset_name):
    medie = [v[0] for v in media_varianza_dizionario.values()]
    varianze = [v[1] for v in media_varianza_dizionario.values()]

    # Calcola i range per medie e varianze
    range_medie = np.ptp(medie)
    range_varianze = np.ptp(varianze)

    # Crea un layout 3x1
    fig, axes = plt.subplots(3, 1, figsize=(10, 20))

    # Grafico 1: Lineplot medie vs varianze (verde)
    sns.lineplot(x=medie, y=varianze, ax=axes[0], color='green')
    axes[0].set_title('Means and Variances', fontsize=16)
    axes[0].set_xlabel('Mean', fontsize=14)
    axes[0].set_ylabel('Var', fontsize=14)

    # Grafico 2: Igramma range medie (blu)
    sns.histplot(medie, bins=np.linspace(min(medie), max(medie), 50), 
                kde=True, ax=axes[1], alpha=0.7, color='blue')
    axes[1].set_title('Means Range', fontsize=16)
    axes[1].set_xlabel('Mean Value', fontsize=14)
    axes[1].set_ylabel('Frequency', fontsize=14)

    # Grafico 3: Igramma range varianze (rosso)
    sns.histplot(varianze, bins=np.linspace(min(varianze), max(varianze), 50),
                kde=True, ax=axes[2], alpha=0.7, color='red')
    axes[2].set_title('Variance Range', fontsize=16)
    axes[2].set_xlabel('Variance Value', fontsize=14)
    axes[2].set_ylabel('Frequency', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'mean_variance_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')

##################################################################################################################################################
#define dataset, model, its version
dataset_name='cifar10'
mode='lenet'
version=''

#define model and relative datset to train and collapse
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
train_dataset = cifar10_noisy_trainset 
test_dataset = cifar10_testset 
trainloader = cifar10_trainloader
testloader = cifar10_testloader

#model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model = model.to(device)

############################################################################
#define parameters
epochs=40

# MSE+WD and low LR 
criterion= nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4, betas=(0.9,0.999))

info_dict = {
        #'A': [],
        #'b': [],
        #'H': [],
        'NC1': [],
        'NC2': [],
        'NC3': [],
        'NC4': [],
        'mem': [],
        'clean_mem':[],
        'fake_mem':[],
        #'mu_G_train': [],
        'train_acc': [],
        'eval_acc': [],
        #'mu_G_test': [],
        #'test_acc1': [],
        #'test_acc3': []
}

tracking_noise= {
    '0':[],
    '1':[],
    '2':[],
    '3':[],
    '4':[],
    '5':[],
    '6':[],
    '7':[],
    '8':[],
    '9':[],
    '10':[],
    '11':[]
}

tracking_clean={
    '0':[],
    '1':[],
    '2':[],
    '3':[],
    '4':[],
    '5':[],
    '6':[],
    '7':[],
    '8':[],
    '9':[],
    '10':[],
    '11':[]
}

force, distance, delta_distance =  {i: [] for i in range(epochs)} , {i: [] for i in range(epochs)} , {i: [] for i in range(epochs)} 

for i in range(epochs):
    #TRAIN MODEL at current epoch

    #Evaluate NC properties at current epoch
    mu_G_train, mu_c_dict_train, train_acc, eval_acc, nc4, mu_c_dict_test = compute_epoch_info( model, trainloader, testloader, optimizer, criterion, num_classes, mode=mode)
    #mu_G_test, mu_c_dict_test, test_acc1, test_acc3 = compute_epoch_info( model, testloader, isTrain=False, mode='ResNet')

    #NC1 and stability metrics( f,d and delta_d are vectors of dimension N to append into a dict that store it as epoch result )
    Sigma_W, f, d, delta_d = compute_Sigma_W_and_force( model, mu_c_dict_train, trainloader, mode=mode)
    force[i]=f 
    distance[i]=d
    delta_distance[i]=delta_d
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

    # Total Memorization
    memorization=0
    for c in mu_c_dict_train.keys():
        memorization+= torch.dist(mu_c_dict_train[c], mu_c_dict_test[c]).item()
    memorization/=len(mu_c_dict_train.keys())
    #memorization= compute_total_memorization(model, trainloader, mu_c_dict_test)

    # Divide dataset between real and fake samples
    #clean_labels_dataset, noisy_labels_dataset= train_dataset.divide_samples()
    #clean_dataloader = DataLoader(clean_labels_dataset, batch_size=128, num_workers=2, drop_last=False)
    #noisy_dataloader = DataLoader(noisy_labels_dataset, batch_size=64, num_workers=2, drop_last=False)

    # Noise Memorization
    #clean_label_memorization = compute_memorization(model, clean_dataloader, mu_c_dict_test)
    # Real labels Memorization
    #noise_label_memorization = compute_memorization(model, noisy_dataloader, mu_c_dict_test)

    # Track of n noisy samples 
    features_distance_items = get_distance_n_noisy_saples(12 , model, train_dataset, mu_c_dict_train)
    for key, v in features_distance_items.items():
        tracking_noise[key].append(v)
    
    #track n clean samples
    features_distance_items = get_distance_n_saples(12 , model, train_dataset, mu_c_dict_train)
    for key, v in features_distance_items.items():
        tracking_clean[key].append(v)

################################################################################
    #Store NC properties
    info_dict['NC1'].append(collapse_metric)
    info_dict['NC2'].append(ETF_metric)
    info_dict['NC3'].append(alignment)
    info_dict['NC4'].append(nc4)
    info_dict['mem'].append(memorization)
    #info_dict['clean_mem'].append(clean_label_memorization)
    #info_dict['fake_mem'].append(noise_label_memorization)
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

    print('[epoch:'+ str(i + 1) +' | train top1:' + str(train_acc) +' | eval acc:' + str(eval_acc) +' | NC1:' + str(collapse_metric)+' | NC2:'+ str(ETF_metric)+' | NC3:'+ str(alignment)+' | NC4:'+str(nc4)+' | memorization:'+ str(memorization))
    
    #if (i+1) % 10 == 0:
    show_plot(info_dict, mode, version, dataset_name) 
    show_noise_samples(tracking_noise, mode, version, dataset_name)
    show_clean_samples(tracking_clean, mode, version, dataset_name)

    torch.save(model.state_dict(), folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/epoch_{i+1}_{mode}{version}_{dataset_name}_weights.pth')

    with open(folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_results.pkl', 'wb') as f:
        pickle.dump(info_dict, f)
    
    with open(folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_noise_track.pkl', 'wb') as f:
        pickle.dump(tracking_noise, f)


# compute mean e variace for distance and force:
# invert dict
force = {i: [v[i] for v in force.values()] for i in range(len(force[i]))}
distance = {i: [v[i] for v in distance.values()] for i in range(len(force[i]))}
# perform mu and Var
force_mu_var = {k: [np.mean(v), np.var(v)] for k, v in force.items()}
distance_mu_var = {k: [np.mean(v), np.var(v)] for k, v in distance.items()}

show_sample_stability(distance_mu_var, mode, version, dataset_name)