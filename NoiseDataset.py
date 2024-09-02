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
from sample_tracker import SampleTracker
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg as scilin

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Shuffle the trainset
num_samples = len(cifar10_trainset)
indices = np.arange(num_samples)
# Esegui lo shuffle degli indici
np.random.shuffle(indices)
cifar10_trainset = Subset(cifar10_trainset, indices)

# Creazione degli oggetti NoisyLabelsDataset
cifar10_noisy_trainset = NoisyLabelsDatasetManager(cifar10_trainset, noise_rate=noise_rate)

# Configurazione del DataLoader
cifar10_trainloader = DataLoader(cifar10_noisy_trainset, batch_size=256, num_workers=2, drop_last=False)
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

#Training Function

@measure_time
def compute_epoch_info(model, dataloader,eval_loader, optimizer, criterion, num_classes, mode='resnet'):
    global features
    features = FeaturesHook()
    features.clear()
    handle_F= select_model(mode)

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

    return mu_G, mu_c_dict, sum(top1)/len(top1), sum(eval_top1)/len(eval_top1), nc4_ratio, mu_c_dict_test

"""Neural Collapse Properties and personalized metrics"""

def compute_delta_distance(mu_c_tensor,features_batch,targets,batch_size):
    distances = torch.cdist(features_batch, mu_c_tensor)  # calcola tendore delle distanze [B,classes]
    centroid_distance = distances[torch.arange(batch_size), targets]  # take distance from label class centoroid
    mask = torch.ones(distances.shape, dtype=bool)
    mask[torch.arange(batch_size), targets] = False  # create a mask for deleteing extracted label centroid distances
    d_remaining = distances[mask].view(batch_size, num_classes-1) # take the rest of distances
    d_min, _ = torch.min(d_remaining, dim=1)  # find the minimum distance centroid from the remaining ones
            
    distance_batch= d_min - centroid_distance

    return distance_batch

# NC1 (sigmaW)
@measure_time
def compute_Sigma_W_and_distance(model, mu_c_dict, dataloader,delta_distance_tracker,distance_tracker, mode='resnet'):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)

    Sigma_W = 0
    num_classes = len(mu_c_dict)

    mu_c_tensor=torch.stack( [torch.tensor(v) for v in mu_c_dict.values()], dim=0).to(device)

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
            mu_c_batch = torch.stack([mu_c_dict[y.item()] for y in target_indices]).to(device)
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
    del mu_c_tensor

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

# Track n samples and show relative distance between its real centroid and its fake centroid
'''
    if distance is positive, sample is more near to its real centroid than its fake centorid...
    else it is the contrary, and the sample became far away from its real classification centroid.
    We want a positive tendency value or ZERO, else memorization occur.
    '''
def get_distance_n_noisy_samples(n,model,train_dataset, mu_c_dict_train):

    global features
    features = FeaturesHook()
    features.clear()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)
    
    # Take first n noisy samples from trainining dataset
    tracked_noise_samples= train_dataset.take_first_n_noise_samples(n)
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
                dist[i]=torch.dist(current_features_F,mu_c_dict_train[i]).item()
            near_centroid_dist=min(dist, key=lambda x: dist[x])

            distance_dict[str(idx)]=  torch.dist(current_features_F,mu_c_dict_train[y_fake.item()]).item() - near_centroid_dist 
            features.clear()

    return distance_dict

# Track n samples and show relative distance between its real centroid and the second nearest 
'''
    if value is positive, sample is far from its real centroid than other centoroid...
    if distance is negative, the sample resides near its real classification centroid.
    We want a negative value as possible
'''
def get_distance_n_samples(n,model,train_dataset, mu_c_dict_train):
   
    global features
    features = FeaturesHook()
    features.clear()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode)
    
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
            if y_fake != y_real:
                raise ValueError("Bad extraction in  'take_first_n_clean_samples' function")
            inputs = inputs.to(device)
            outputs = model(inputs)
            current_features_F = features.features_F[-1].squeeze()
            class_indices=list(range(10))
            del class_indices[y_real.item()]
            dist=dict()
            for i in class_indices:
                dist[i]=torch.dist(current_features_F,mu_c_dict_train[i]).item()
            near_centroid_dist=min(dist, key=lambda x: dist[x])

            distance_dict[str(idx)]=  torch.dist(current_features_F,mu_c_dict_train[y_real.item()]).item() - near_centroid_dist
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

def show_mean_var_relevations(tensor, mode, version, dataset_name,noisy_indices, dict_type='delta_distance', ):
 
    # Calcola le medie e le varianze per ogni riga (N,)
    means = tensor.mean(dim=1).cpu().numpy()
    variances = tensor.var(dim=1).cpu().numpy()

    # Impostazioni di Seaborn per un grafico più accattivante
    sns.set_style("whitegrid")
    sns.set_palette('deep')

    # Crea la figura e gli assi
    fig, axes = plt.subplots(3, 1, figsize=(8, 15))

    # Istogramma delle medie
    sns.histplot(means, bins=30, ax=axes[0], color='skyblue')
    axes[0].set_title('Histogram of Means')
    axes[0].set_xlabel('Mean')
    axes[0].set_ylabel('Count')

    # Istogramma delle varianze
    sns.histplot(variances, bins=30, ax=axes[1], color='lightcoral')
    axes[1].set_title('Histogram of Variances')
    axes[1].set_xlabel('Variance')
    axes[1].set_ylabel('Count')

    # Scatter plot con KDE
    # Disegna prima i punti non evidenziati
    sns.scatterplot(x=means, y=variances, ax=axes[2], color='green', label='Normal Points')

    # Se noisy_indices è fornito e non è vuoto
    if noisy_indices is not None and len(noisy_indices) > 0:
        # Seleziona i punti da evidenziare e dissegnali sopra quelli normali
        noisy_means = means[noisy_indices]
        noisy_variances = variances[noisy_indices]
        sns.scatterplot(x=noisy_means, y=noisy_variances, ax=axes[2], color='red', label='Noisy Points', edgecolor='black', alpha=0.7)

    # Disegna la KDE plot
    sns.kdeplot(x=means, y=variances, ax=axes[2], cmap="Blues_r", fill=True, alpha=0.5)

    # Aggiungi le etichette e la leggenda
    axes[2].set_title('Scatter plot of Means vs Variances with KDE')
    axes[2].set_xlabel('Mean')
    axes[2].set_ylabel('Variance')
    axes[2].legend()

    plt.tight_layout()
    # Salvare il grafico in un file PNG
    plt.savefig(f'mean_variance_epoch_{len(tensor[0])}_{dict_type}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

##################################################################################################################################################
#define dataset, model, its version
dataset_name='cifar10'
mode='resnet'
version='18'

#define model and relative datset to train and collapse
model = cifar10_resnet18
num_classes = cifar10_classes
feature_size =  resnet_feature_size

#flags
resnet=True
densenet=False
lenet=False
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
epochs=50

# MSE+WD and low LR 
criterion= nn.BCEWithLogitsLoss()
optimizer=optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4, betas=(0.9,0.999))

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

#TODO: angular variation

for i in range(epochs):
    #TRAIN MODEL at current epoch

    #Evaluate NC properties at current epoch
    mu_G_train, mu_c_dict_train, train_acc, eval_acc, nc4, mu_c_dict_test = compute_epoch_info( model, trainloader, testloader, optimizer, criterion, num_classes, mode=mode)

    #NC1 and stability metrics( f,d and delta_d are vectors of dimension N to append into a dict that store it as epoch result )
    Sigma_W = compute_Sigma_W_and_distance( model, mu_c_dict_train, trainloader, delta_distance_tracker, distance_tracker, mode=mode)

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
    #memorization= compute_total_memorization(model, trainloader, mu_c_dict_test)

    # Divide dataset between real and fake samples
    clean_labels_dataset, noisy_labels_dataset= train_dataset.divide_samples()
    #clean_dataloader = DataLoader(clean_labels_dataset, batch_size=128, num_workers=2, drop_last=False)
    #noisy_dataloader = DataLoader(noisy_labels_dataset, batch_size=128, num_workers=2, drop_last=False)

    # Noise Memorization
    #clean_label_memorization = compute_memorization(model, clean_dataloader, mu_c_dict_test)
    # Real labels Memorization
    #noise_label_memorization = compute_memorization(model, noisy_dataloader, mu_c_dict_test)

    # Track of n noisy samples 
    features_distance_items = get_distance_n_noisy_samples(12 , model, train_dataset, mu_c_dict_train)
    for key, v in features_distance_items.items():
        tracking_noise[key].append(v)
    
    #track n clean samples
    features_distance_items = get_distance_n_samples(12 , model, train_dataset, mu_c_dict_train)
    for key, v in features_distance_items.items():
        tracking_clean[key].append(v)

################################################################################
    #Store NC properties
    info_dict['NC1'].append(collapse_metric)
    info_dict['NC2'].append(ETF_metric)
    info_dict['NC3'].append(alignment)
    info_dict['NC4'].append(nc4)
    info_dict['mu_c_train'].append(mu_c_dict_train)
    #info_dict['mu_c_wighted'].append()
    #info_dict['mu_c_clean'].append()
    #info_dict['mem'].append(memorization)
    #info_dict['clean_mem'].append(clean_label_memorization)
    #info_dict['fake_mem'].append(noise_label_memorization)

    #Store accuracies
    info_dict['train_acc'].append(train_acc)
    info_dict['eval_acc'].append(eval_acc)

    print('[epoch:'+ str(i + 1) +' | train top1:' + str(train_acc) +' | eval acc:' + str(eval_acc) +' | NC1:' + str(collapse_metric))
    
    if (i+1) % 10 == 0:
        show_plot(info_dict, mode, version, dataset_name) 
        show_noise_samples(tracking_noise, mode, version, dataset_name)
        show_clean_samples(tracking_clean, mode, version, dataset_name)

        delta_distances= delta_distance_tracker.tensorize()
        distances= distance_tracker.tensorize()

        show_mean_var_relevations(delta_distances,mode,version,dataset_name, noisy_indices=cifar10_noisy_trainset.get_corrupted_indices(), dict_type='delta_distance')

        del delta_distances,distances

    torch.save(model.state_dict(), folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/epoch_{i+1}_{mode}{version}_{dataset_name}_weights.pth')

    with open(folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_results.pkl', 'wb') as f:
        pickle.dump(info_dict, f)
    
    with open(folder_path + f'/noise/noise10/{dataset_name}/{mode}{version}/{mode}{version}_{dataset_name}_noise_track.pkl', 'wb') as f:
        pickle.dump(tracking_noise, f)