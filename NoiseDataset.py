import torch
import numpy as np
import pickle
import os

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from Hooks import FeaturesHook
from sample_tracker import SampleTracker
from tsne import EmbeddingVisualizer2D

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg as scilin

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
        
        num_samples = len(self.dataset)
        num_corrupted = int(self.noise_rate * num_samples)
        self.corrupted_indices = np.random.choice(num_samples, size=num_corrupted, replace=False)
        self.inject_noise()

    def inject_noise(self):
        for idx in self.corrupted_indices:
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
    
    def get_corrupted_indices(self):
        return self.corrupted_indices
    
    def take_random_n_noise_samples(self, n):
        # Seleziona casualmente n campioni in cui l'etichetta reale non corrisponde a quella falsa
        matching_indices = np.where(self.real_labels != self.fake_labels)[0]
        selected_indices = np.random.choice(matching_indices, size=n, replace=False)
        
        # Prepara i dati e le etichette per i campioni selezionati
        selected_data = [self.dataset[i][0] for i in selected_indices]
        selected_real_labels = self.real_labels[selected_indices]
        selected_fake_labels = self.fake_labels[selected_indices]
        
        # Converte le etichette in tensori di PyTorch
        selected_real_labels_tensor = torch.tensor(selected_real_labels, dtype=torch.long)
        selected_fake_labels_tensor = torch.tensor(selected_fake_labels, dtype=torch.long)
        
        # Crea un nuovo dataset con i campioni selezionati e le etichette concatenate
        selected_dataset = NoisyLabelsDataset(dataset=selected_data, real_labels=selected_real_labels_tensor, fake_labels=selected_fake_labels_tensor)
        
        return selected_dataset
    
    def take_random_n_clean_samples(self, n):
        # Seleziona casualmente n campioni in cui l'etichetta reale corrisponde a quella falsa
        matching_indices = np.where(self.real_labels == self.fake_labels)[0]
        selected_indices = np.random.choice(matching_indices, size=n, replace=False)
        
        # Prepara i dati e le etichette per i campioni selezionati
        selected_data = [self.dataset[i][0] for i in selected_indices]
        selected_real_labels = self.real_labels[selected_indices]
        selected_fake_labels = self.fake_labels[selected_indices]
        
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
        non_matching_real_labels = self.real_labels[non_matching_indices]

        # Create combined datasets including data and fake labels
        clean_samples = NoiseDataset(matching_data, matching_fake_labels)
        noisy_labels_samples = NoiseDataset(non_matching_data, non_matching_fake_labels)
        complete_noisy_labels_samples = NoisyLabelsDataset(non_matching_data, non_matching_real_labels, non_matching_fake_labels)
        
        return clean_samples, noisy_labels_samples, complete_noisy_labels_samples

    def __len__(self):
        return len(self.dataset)
    
class NoisySubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset

    def __getattr__(self, name):
        # Questo metodo viene chiamato quando cerchi di accedere a un attributo/metodo
        # che non è definito direttamente in CustomSubset, ma potrebbe esserlo in self.dataset
        return getattr(self.dataset, name)

"""CIFAR-10"""

# Caricamento clean datasets
cifar10_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Shuffle the trainset
num_samples = len(cifar10_trainset)
indices = np.arange(num_samples)
np.random.shuffle(indices)
shuffled_train_dataset= Subset(cifar10_trainset,indices)

# Creazione degli oggetti NoisyLabelsDataset
cifar10_noisy_trainset = NoisyLabelsDatasetManager(shuffled_train_dataset, noise_rate=noise_rate)


# Configurazione del DataLoader
cifar10_trainloader = DataLoader(cifar10_noisy_trainset, batch_size=256, shuffle=False, num_workers=2, drop_last=False)
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
##################################################################################################################################################

# Accuracy metric
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
    handle_F = select_model(mode)

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
    handle_F = select_model(mode)

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

def compute_label_coherence_score(model, mu_c_tensor, dataloader, norm_factor):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)
    coherence_count=0
    num_classes = len(mu_c_tensor)


    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            fake_targets = targets[:,1:,:].squeeze().to(device)
            real_targets = targets[:,:1,:].squeeze().to(device)
            outputs = model(inputs)

            features_batch = features.features_F[-1].squeeze()
            batch_size=len(features_batch)

            # Reset delle liste di features per il prossimo batch
            features.features_F = []
            features.clear()

            distances = torch.cdist(features_batch, mu_c_tensor)  # calcola tensore delle distanze [B,classes]
            
            centroid_distance = distances[torch.arange(batch_size), fake_targets]  # take distance from 'fake' label class centoroid
            real_centroid_distances = distances[torch.arange(batch_size), real_targets] # take distance from real label class centoroid
            
            mask = torch.ones(distances.shape, dtype=bool)
            mask[torch.arange(batch_size), fake_targets] = False  # create a mask for deleting extracted label centroid distances
            d_remaining = distances[mask].view(batch_size, num_classes-1) # take the rest of distances
            d_min, _ = torch.min(d_remaining, dim=1)  # find the minimum distance centroid from the remaining ones
        
            coherence_count += (d_min == real_centroid_distances).sum().item()

    return coherence_count / norm_factor

def compute_label_coherence_for_noisy_outliers(model, mu_c_tensor, dataloader):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)
    coherence_count=0
    num_classes = len(mu_c_tensor)
    norm_factor=0

    model.eval()
    for idx, (inputs, targets) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            fake_targets = targets[:,1:,:].squeeze().to(device)
            real_targets = targets[:,:1,:].squeeze().to(device)
            outputs = model(inputs)

            features_batch = features.features_F[-1].squeeze()
            batch_size=len(features_batch)

            # Reset delle liste di features per il prossimo batch
            features.features_F = []
            features.clear()

            distances = torch.cdist(features_batch, mu_c_tensor)  # calcola tensore delle distanze [B,classes]
            
            centroid_distance = distances[torch.arange(batch_size), fake_targets]  # take distance from 'fake' label class centoroid
            real_centroid_distances = distances[torch.arange(batch_size), real_targets] # take distance from real label class centoroid
            
            mask = torch.ones(distances.shape, dtype=bool)
            mask[torch.arange(batch_size), fake_targets] = False  # create a mask for deleting extracted label centroid distances
            d_remaining = distances[mask].view(batch_size, num_classes-1) # take the rest of distances
            d_min, _ = torch.min(d_remaining, dim=1)  # find the minimum distance centroid from the remaining ones

            distance_batch= d_min - centroid_distance

            # Select only samples where the difference is greater than 0
            valid_samples_mask = distance_batch < 0
            valid_real_centroid_distances = real_centroid_distances[valid_samples_mask]

            # Update the coherence count by comparing real centroid distances with the minimum distances
            coherence_count += (d_min[valid_samples_mask] == valid_real_centroid_distances).sum().item()

            # Update the normalization factor based on the number of valid samples
            norm_factor += valid_samples_mask.sum().item()

    # Return the coherence count normalized by the number of valid samples
    return coherence_count / norm_factor if norm_factor > 0 else 0

    return coherence_count / norm_factor


#Update centorid values over batch
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

def compute_weighted_mu_c(features, labels, num_classes, previous_mu_c):
    # Inizializza i tensori di output
    weighted_mu_c_batch = torch.zeros((num_classes, features.size(1)), dtype=features.dtype, device=features.device)
    counter = torch.zeros(num_classes, dtype=torch.int32, device=features.device)
    
    # Itera su ogni classe e somma le feature corrispondenti
    for c in range(num_classes):
        mask = (labels == c)
        # class projection
        class_features = features[mask]
        
        if class_features.size(0) > 0:
            # Perform Distance(mu_c[c],F_i[c]) for every sample i in batch_size
            distances = torch.cdist(class_features, previous_mu_c[c].unsqueeze(0), p=2).squeeze(1)

            # Perform Weighting factor using a specific function
            alpha=3
            beta=0.1
            weights = torch.exp( distances / alpha) + beta

            #Alternative distance metrics:

            #alpha=2.0
            #weights = 1 / (distances ** alpha) 

            #sigma=1.0
            #weights = torch.exp( max(distances,4) / sigma)
        
            # Weighted sum
            weighted_sum = (class_features * weights.unsqueeze(1)).sum(dim=0)
            # Update centroid 
            weighted_mu_c_batch[c] = weighted_sum
            # Update count for normalization factor
            counter[c] = len(class_features)

    return weighted_mu_c_batch, counter

@measure_time
def compute_epoch_info(model, dataloader,eval_loader, optimizer, criterion, num_classes, feature_size, mode='resnet'):
    global features
    features = FeaturesHook()
    features.clear()
    handle_F= select_model(mode)

    mu_G = 0

    mu_c = torch.zeros((num_classes, feature_size), device=device)
    mu_c_test = torch.zeros((num_classes, feature_size), device=device)
    mu_c_clean = torch.zeros((num_classes, feature_size), device=device)
    mu_c_weighted = torch.zeros((num_classes, feature_size), device=device)

    counter = torch.zeros(num_classes, dtype=torch.int32, device=device)
    clean_counter = torch.zeros(num_classes, dtype=torch.int32, device=device)
    weighted_counter = torch.zeros(num_classes, dtype=torch.int32, device=device)
    test_counter= torch.zeros(num_classes, dtype=torch.int32, device=device)

    top1 = []
    eval_top1 = []

    model.train()
    for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = labels[:,1:,:].squeeze().to(device) # take only corrupted labels
            real_targets = labels[:,:1,:].squeeze().to(device) #take true labels, useful to create clean centroids.
            optimizer.zero_grad()
            outputs = model(inputs).logits

            current_features_F = features.features_F[-1].squeeze()
            # Reset delle liste di features per il prossimo batch
            features.clear()

            # Update network
            one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
            loss = criterion(outputs, one_hot_targets)
            loss.backward()
            optimizer.step()

            # Aggiorna mu_G
            mu_G += current_features_F.sum(dim=0)

            # Crea maschere per campioni puliti e corrotti
            clean_mask = targets == real_targets
            # Aggiorna mu_c_dict e counter per campioni puliti
            clean_features = current_features_F[clean_mask]
            clean_targets = targets[clean_mask]

            clean_mu_c_batch, clean_counter_batch= compute_mu_c(clean_features,clean_targets, num_classes)
            mu_c_batch, counter_batch= compute_mu_c(current_features_F, targets, num_classes)

            mu_c_clean += clean_mu_c_batch
            mu_c += mu_c_batch

            clean_counter += clean_counter_batch
            counter += counter_batch
            
            #Compute accuracy
            prec1 = top_k_accuracy(outputs,targets ,1)
            top1.append(prec1)

    # Normalize to obtain final class means and global mean
    mu_G /= len(dataloader.dataset)
    for k in range(len(mu_c)):
          mu_c[k] /= counter[k] 
          mu_c_clean[k] /= clean_counter[k] 
    
    # perform centroid by weight sample based on distance on mu_c normal centroids
    model.eval()
    for idx, (inputs, labels) in enumerate(dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = labels[:,1:,:].squeeze().to(device)
            outputs = model(inputs)

            current_features_F = features.features_F[-1].squeeze()
            features.clear()

            weighted_mu_c_batch, weighted_counter_batch= compute_weighted_mu_c(current_features_F, targets, num_classes, mu_c)

            mu_c_weighted += weighted_mu_c_batch
            weighted_counter += weighted_counter_batch

    for k in range(len(mu_c_weighted)):
          mu_c_weighted[k] /= weighted_counter[k]  
        
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
    
    return mu_G, mu_c, mu_c_clean, mu_c_weighted, sum(top1)/len(top1), sum(eval_top1)/len(eval_top1), nc4_ratio, mu_c_test

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
            targets = targets[:,1:,:].squeeze().to(device)
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

# Track n samples and show relative distance between its real centroid and its fake centroid
'''
    if distance is positive, sample is more near to its real centroid than its fake centorid...
    else it is the contrary, and the sample became far away from its real classification centroid.
    We want a positive tendency value or ZERO, else memorization occur.
    '''
def get_distance_n_noisy_samples(n,model,train_dataset, mu_c_train):

    global features
    features = FeaturesHook()
    features.clear()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)
    
    # Take first n noisy samples from trainining dataset
    tracked_noise_samples= train_dataset.take_random_n_noise_samples(n)
    tracked_noise_samples_loader=DataLoader(tracked_noise_samples, batch_size=1, num_workers=2)

    # foreach sample take its features and compute relative distance ( real - fake )
    distance_dict=dict()
    model.eval()
    for idx, (inputs, targets) in enumerate(tracked_noise_samples_loader):
            y_real = targets[:,:1,:].squeeze().to(device)
            y_fake = targets[:,1:,:].squeeze().to(device)
            if y_fake == y_real:
                raise ValueError("Bad extraction in  'take_first_n_clean_samples' function")
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()
            class_indices=list(range(10))
            del class_indices[y_real.item()]
            dist=dict()
            for i in class_indices:
                dist[i]=torch.dist(current_features_F,mu_c_train[i]).item()
            near_centroid_dist=min(dist, key=lambda x: dist[x])

            distance_dict[str(idx)] = near_centroid_dist - torch.dist(current_features_F,mu_c_train[y_fake.item()]).item()
            
            features.clear()

    return distance_dict

# Track n samples and show relative distance between its real centroid and the second nearest 
'''
    if value is positive, sample is far from its real centroid than other centoroid...
    if distance is negative, the sample resides near its real classification centroid.
    We want a negative value as possible
'''
def get_distance_n_samples(n,model,train_dataset, mu_c_train):
   
    global features
    features = FeaturesHook()
    features.clear()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)
    
    # Take first n noisy samples from trainining dataset
    tracked_clean_samples= train_dataset.take_random_n_clean_samples(n)
    tracked_clean_samples_loader=DataLoader(tracked_clean_samples, batch_size=1, num_workers=2)

    # foreach sample take its features and compute relative distance ( real - fake )
    distance_dict=dict()
    model.eval()
    for idx, (inputs, targets) in enumerate(tracked_clean_samples_loader):
        with torch.no_grad():
            y_real = targets[:,:1,:].squeeze().to(device)
            y_fake = targets[:,1:,:].squeeze().to(device)
            if y_fake != y_real:
                raise ValueError("Bad extraction in  'take_random_n_clean_samples' function")
            inputs = inputs.to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()
            class_indices=list(range(10))
            del class_indices[y_real.item()]
            dist=dict()
            for i in class_indices:
                dist[i]=torch.dist(current_features_F,mu_c_train[i]).item()
            near_centroid_dist=min(dist, key=lambda x: dist[x])

            distance_dict[str(idx)]=  near_centroid_dist - torch.dist(current_features_F,mu_c_train[y_real.item()]).item()
            features.clear()

    return distance_dict

# show acuracy and NC metrics
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
                 color='yellow',
                 label='total memo.')  # Cambia il colore e il marker
    
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

#tracking of n noisy samples relative distance respect its real/fake centorids 
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

# tracking of n samples relative distance respect its label/nearest centorids 
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

def show_mean_var_relevations(tensor, mode, version, dataset_name, noisy_indices, dict_type='delta_distance'):
    # Calcola le medie e le varianze per ogni riga (N,)
    means = tensor.mean(dim=1).cpu().numpy()
    variances = tensor.var(dim=1).cpu().numpy()

    # Determina gli indici per i campioni "clean" (non rumorosi)
    clean_indices = np.setdiff1d(np.arange(tensor.shape[0]), noisy_indices)

    # Estrai medie e varianze per i campioni "clean" e "noisy"
    clean_means = means[clean_indices]
    clean_variances = variances[clean_indices]
    noisy_means = means[noisy_indices]
    noisy_variances = variances[noisy_indices]

    # Impostazioni di Seaborn per un grafico più accattivante
    sns.set_style("whitegrid")
    sns.set_palette('deep')

    # Crea la figura e gli assi (3x3 layout)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Prima colonna: Tutti i campioni
    sns.histplot(means, bins=30, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('All Samples: Histogram of Means')
    axes[0, 0].set_xlabel('Mean')
    axes[0, 0].set_ylabel('Count')

    sns.histplot(variances, bins=30, ax=axes[1, 0], color='lightcoral')
    axes[1, 0].set_title('All Samples: Histogram of Variances')
    axes[1, 0].set_xlabel('Variance')
    axes[1, 0].set_ylabel('Count')

    # Scatter plot con separazione dei campioni rumorosi
    sns.scatterplot(x=means, y=variances, ax=axes[2, 0], color='green', label='Clean Samples')
    sns.scatterplot(x=noisy_means, y=noisy_variances, ax=axes[2, 0], color='red', label='Noisy Samples', edgecolor='black', alpha=0.7)
    sns.kdeplot(x=means, y=variances, ax=axes[2, 0], cmap="Blues_r", fill=True, alpha=0.5)
    axes[2, 0].set_title('All Samples: Scatter plot of Means vs Variances')
    axes[2, 0].set_xlabel('Mean')
    axes[2, 0].set_ylabel('Variance')
    axes[2, 0].legend()

    # Trova i limiti comuni per l'asse Y nei grafici delle medie
    all_counts, _ = np.histogram(means, bins=30)
    clean_counts, _ = np.histogram(clean_means, bins=30)
    noisy_counts, _ = np.histogram(noisy_means, bins=30)
    
    max_count = max(all_counts.max(), clean_counts.max(), noisy_counts.max())

    # Seconda colonna: Campioni puliti (clean)
    sns.histplot(clean_means, bins=30, ax=axes[0, 1], color='skyblue')
    axes[0, 1].set_title('Clean Samples: Histogram of Means')
    axes[0, 1].set_xlabel('Mean')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_ylim(0, max_count)

    sns.histplot(clean_variances, bins=30, ax=axes[1, 1], color='lightcoral')
    axes[1, 1].set_title('Clean Samples: Histogram of Variances')
    axes[1, 1].set_xlabel('Variance')
    axes[1, 1].set_ylabel('Count')

    sns.scatterplot(x=clean_means, y=clean_variances, ax=axes[2, 1], color='green', label='Clean Samples')
    sns.kdeplot(x=clean_means, y=clean_variances, ax=axes[2, 1], cmap="Blues_r", fill=True, alpha=0.5)
    axes[2, 1].set_title('Clean Samples: Scatter plot of Means vs Variances')
    axes[2, 1].set_xlabel('Mean')
    axes[2, 1].set_ylabel('Variance')
    axes[2, 1].legend()

    # Terza colonna: Campioni rumorosi (noisy)
    sns.histplot(noisy_means, bins=30, ax=axes[0, 2], color='skyblue')
    axes[0, 2].set_title('Noisy Samples: Histogram of Means')
    axes[0, 2].set_xlabel('Mean')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_ylim(0, max_count)

    sns.histplot(noisy_variances, bins=30, ax=axes[1, 2], color='lightcoral')
    axes[1, 2].set_title('Noisy Samples: Histogram of Variances')
    axes[1, 2].set_xlabel('Variance')
    axes[1, 2].set_ylabel('Count')

    sns.scatterplot(x=noisy_means, y=noisy_variances, ax=axes[2, 2], color='red', label='Noisy Samples', edgecolor='black', alpha=0.7)
    sns.kdeplot(x=noisy_means, y=noisy_variances, ax=axes[2, 2], cmap="Blues_r", fill=True, alpha=0.5)
    axes[2, 2].set_title('Noisy Samples: Scatter plot of Means vs Variances')
    axes[2, 2].set_xlabel('Mean')
    axes[2, 2].set_ylabel('Variance')
    axes[2, 2].legend()

    plt.tight_layout()
    # Salvare il grafico in un file PNG
    plt.savefig(f'mean_variance_epoch_{len(tensor[0])}_{dict_type}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def show_centroid_distances(weighted_dist, normal_dist,mode,version,dataset_name):
    weighted_dist=weighted_dist[5:] # first five elements are huge and hide the visualization around zero values
    # Impostare uno stile di base e una palette di colori
    sns.set_style("whitegrid")  # Stile con griglia bianca

    # Imposta lo stile di Seaborn
    sns.set(style="whitegrid")

    fig, axs = plt.subplots(2, 5, figsize=(15, 10))  # 2 righe e 5 colonne di grafici

    for i in range(num_classes):
        ax = axs[i // 5, i % 5]  # Seleziona il grafico in base alla classe
        
        # Plot delle distanze usando Seaborn
        sns.lineplot(x=torch.arange(weighted_dist.size(0)), 
                     y=weighted_dist[:, i].numpy(), 
                     color='green', 
                     label=f'W_c', 
                     ax=ax)
        
        sns.lineplot(x=torch.arange(normal_dist.size(0)), 
                     y=normal_dist[:, i].numpy(), 
                     color='red', 
                     label=f'normal_c({i})', 
                     ax=ax)
        
        ax.set_title(f'Class {i}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distance')
        ax.legend()

    # Aggiustiamo lo spazio tra i plot per una migliore visualizzazione
    plt.tight_layout()
    # Salvare il grafico in un file PNG
    plt.savefig(f'centroid_distances_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')

def show_noise_coherence(info_dict, mode, version, dataset_name):
    sns.set_style("whitegrid")  # Stile con griglia bianca
    sns.set_palette("Set2")  # Set palette per i colori

    plt.figure(figsize=(10, 6))  # Definisci una figura di dimensioni appropriate

    # Linea per la coerenza dei campioni rumorosi
    sns.lineplot(x=np.arange(1, len(info_dict['coherence']) + 1),
                 y=info_dict['coherence'],
                 marker="o",
                 linewidth=2,
                 color='blue',
                 label='Coherence')

    # Linea per la coerenza degli outliers rumorosi
    sns.lineplot(x=np.arange(1, len(info_dict['outliers_coherence']) + 1),
                 y=info_dict['outliers_coherence'],
                 marker="o",
                 linewidth=2,
                 color='red',
                 label='Outliers Coherence')

    # Imposta titolo, etichette e legenda
    plt.title(f'Coherence Metrics for Noisy Samples and Outliers')
    plt.xlabel('Epoch')
    plt.ylabel('% Sample Coherent with Real Label')
    plt.legend()

    # Salva il grafico in un file PNG
    plt.tight_layout()
    plt.savefig(f'coherence_epoch_{len(info_dict["coherence"])}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')

    # Chiudi la figura per liberare memoria
    plt.close()

##################################################################################################################################################
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
criterion= nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(), lr=0.001,  betas=(0.9,0.999)) # weight_decay=5e-4,

info_dict = {
        'NC1': [],
        'NC2': [],
        'NC3': [],
        'NC4': [],
        'mem': [],
        'train_acc': [],
        'eval_acc': [],
        'mu_c_train':[],
        'mu_c_weighted':[],
        'mu_c_clean':[],
        'coherence':[],
        'outliers_coherence':[],
        #'clean_mem':[],
        #'fake_mem':[],
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


delta_distance_tracker=SampleTracker()
distance_tracker=SampleTracker()

clean_weighted_centroid_distances = torch.zeros(((epochs), num_classes))
clean_real_centroid_distances = torch.zeros(((epochs) , num_classes))


for i in range(epochs):
    #TRAIN MODEL at current epoch

    #Evaluate NC properties at current epoch
    mu_G_train, mu_c_train, mu_c_clean, mu_c_weighted, train_acc, eval_acc, nc4, mu_c_test = compute_epoch_info( model, trainloader, testloader, optimizer, criterion, num_classes, feature_size, mode=mode)

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

    # Total Memorization
    #memorization= compute_total_memorization(model, trainloader, mu_c_test)
    
    # Divide dataset between real and fake samples
    clean_labels_dataset, noisy_labels_dataset, complete_noisy_labels_dataset = train_dataset.divide_samples()
    clean_dataloader = DataLoader(clean_labels_dataset, batch_size=128, num_workers=2, drop_last=False)
    noisy_dataloader = DataLoader(noisy_labels_dataset, batch_size=128, num_workers=2, drop_last=False)
    complete_noisy_dataloader = DataLoader(complete_noisy_labels_dataset, batch_size=128, num_workers=2, drop_last=False)

    norm_factor=len(complete_noisy_labels_dataset)
    label_coherence_score = compute_label_coherence_score(model, mu_c_train, complete_noisy_dataloader, norm_factor)
    label_coherence_of_nearest_centroid = compute_label_coherence_for_noisy_outliers(model, mu_c_train, complete_noisy_dataloader)
    
    # Noise Memorization
    #clean_label_memorization = compute_memorization(model, clean_dataloader, mu_c_test)
    # Real labels Memorization
    #noise_label_memorization = compute_memorization(model, noisy_dataloader, mu_c_test)

    # Track of n noisy samples 
    features_distance_items = get_distance_n_noisy_samples(12 , model, train_dataset, mu_c_train)
    for key, v in features_distance_items.items():
        tracking_noise[key].append(v)
    
    #track n clean samples
    features_distance_items = get_distance_n_samples(12 , model, train_dataset, mu_c_train)
    for key, v in features_distance_items.items():
        tracking_clean[key].append(v)

################################################################################
    #Store NC properties
    info_dict['NC1'].append(collapse_metric)
    info_dict['NC2'].append(ETF_metric)
    info_dict['NC3'].append(alignment)
    info_dict['NC4'].append(nc4)
    info_dict['mu_c_train'].append(mu_c_train)
    info_dict['mu_c_weighted'].append(mu_c_weighted)
    info_dict['mu_c_clean'].append(mu_c_clean)
    info_dict['coherence'].append(label_coherence_score)
    info_dict['outliers_coherence'].append(label_coherence_of_nearest_centroid)
    #info_dict['mem'].append(memorization)
    #info_dict['clean_mem'].append(clean_label_memorization)
    #info_dict['fake_mem'].append(noise_label_memorization)

    #Store accuracies
    info_dict['train_acc'].append(train_acc)
    info_dict['eval_acc'].append(eval_acc)

    #Compute distances between computed centroids
    clean_weighted = torch.norm(mu_c_weighted - mu_c_clean, dim=1)  
    clean_real = torch.norm(mu_c_train - mu_c_clean, dim=1)  

    # Store distances at current epoch
    clean_weighted_centroid_distances[i, :] = clean_weighted
    clean_real_centroid_distances[i, :] = clean_real

    if (i+1) % 5 == 0:
        show_plot(info_dict, mode, version, dataset_name) 
        show_noise_samples(tracking_noise, mode, version, dataset_name)
        show_clean_samples(tracking_clean, mode, version, dataset_name)

        delta_distances= delta_distance_tracker.tensorize()
        #distances= distance_tracker.tensorize()

        show_mean_var_relevations(delta_distances, mode, version, dataset_name, noisy_indices=cifar10_noisy_trainset.get_corrupted_indices(), dict_type='delta_distance')
        show_noise_coherence(info_dict, mode, version,dataset_name)
        del delta_distances
        
    '''
    if (i+1) % epochs == 0:
        visualizer = EmbeddingVisualizer2D(model,feature_size, mode, version, dataset_name, (i+1), device, trainloader, cifar10_noisy_trainset.corrupted_indices, mu_c_train, mu_c_weighted, mu_c_clean)
        visualizer.run()
    '''

    torch.save(model.state_dict(), folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/epoch_{i+1}_{mode}{version}_{dataset_name}_weights.pth')

    with open(folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_results.pkl', 'wb') as f:
        pickle.dump(info_dict, f)

    with open(folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_noise_track.pkl', 'wb') as f:
        pickle.dump(tracking_noise, f)

    print(f'[epoch:{i + 1} | train top1:{train_acc:.4f} | eval acc:{eval_acc:.4f} | NC1:{collapse_metric:.4f} | NC4:{nc4:.4f} | coherence:{label_coherence_score:.4f}]')

show_centroid_distances(clean_weighted_centroid_distances,clean_real_centroid_distances,mode,version,dataset_name)

