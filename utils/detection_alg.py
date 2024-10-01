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

def reset_model(mode, classes):
    model=None
    feature_size=0
    if mode == 'resnet18':
        model = models.resnet18(pretrained=False)
        feature_size=model.fc.in_features
        model.fc = nn.Linear(model.fc.in_features, classes, bias=False)
    elif mode == 'densenet121':
        model = models.densenet121(pretrained=False)
        feature_size=model.classifier.in_features
        model.classifier = nn.Linear(model.classifier.in_features, classes, bias=False)
    elif mode == 'lenet':
        model = models.googlenet(pretrained=False)
        feature_size=model.fc.in_features
        model.fc = nn.Linear(model.fc.in_features, classes, bias=False)
    elif mode== 'regnet400':
        model = models.regnet_y_400mf(pretrained=False)
        feature_size=model.fc.in_features
        model.fc = nn.Linear(model.fc.in_features, classes, bias=False)
    elif mode== 'efficientnet':
        model = models.efficientnet_b0(pretrained=False)
        feature_size=model.classifier[1].in_features
        model.classifier[1]=nn.Linear(model.classifier[1].in_features, classes, bias=False)
    elif mode == 'mnas05':
        model = models.mnasnet0_5(pretrained=False)
        feature_size=model.classifier[1].in_features
        model.classifier[1]=nn.Linear(model.classifier[1].in_features, classes, bias=False)
    elif mode == 'mnas075':
        model = models.mnasnet0_75(pretrained=False)
        feature_size=model.classifier[1].in_features
        model.classifier[1]=nn.Linear(model.classifier[1].in_features, classes, bias=False)
    else:
        raise ValueError("Mode not supported")

    return model , feature_size

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

# Creazione degli oggetti NoisyLabelsDataset
cifar10_noisy_trainset = NoisyLabelsDatasetManager(cifar10_trainset, noise_rate=noise_rate)

cifar10_testloader = DataLoader(cifar10_testset, batch_size=256, num_workers=2, drop_last=True) 

cifar10_classes = 10

# Accuracy metric
def top_k_accuracy(outputs, labels, k=3):

    # Otteniamo i primi k indici di predizioni più alte per ciascun campione
    top_k_preds = torch.topk(outputs, k, dim=1).indices

    # Verifichiamo se l'etichetta corretta è tra i top k predizioni
    correct = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))

    # Calcoliamo la top-k accuracy
    top_k_acc = correct.sum().item() / labels.size(0)

    return top_k_acc

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

            # Aggiorna mu_C
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
    mask[torch.arange(batch_size), targets] = False  # create a mask for deleting extracted label centroid distances
    d_remaining = distances[mask].view(batch_size, num_classes-1) # take the rest of distances
    d_min, _ = torch.min(d_remaining, dim=1)  # find the minimum distance centroid from the remaining ones
            
    distance_batch= d_min - centroid_distance # If high distace from its label centroid, negative value is returned

    return distance_batch

# NC1 (sigmaW)
@measure_time
def compute_Sigma_W_and_distance(model, mu_c_tensor, dataloader,delta_distance_tracker, mode='resnet'):
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

def show_mean_var_relevations(tensor, mode, version, dataset_name, noisy_indices, index_map, label='delta_distance'):
    # Calcola le medie e le varianze per ogni riga (N,)
    means = tensor.mean(dim=1).cpu().numpy()
    variances = tensor.var(dim=1).cpu().numpy()

    # Converti gli indici noisy da indici mescolati a indici originali, se necessario
    if index_map is not None:
        noisy_indices = np.array([index_map[idx] for idx in noisy_indices])

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
    plt.savefig(f'mean_variance_epoch_{len(tensor[0])}_{label}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

##############################################################################################################################

#define dataset, model, its version
dataset_name='cifar10'
mode='lenet'
version=''

#define model and relative dataset to train and collapse
num_classes = cifar10_classes

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

testloader = cifar10_testloader

#############################################################################################################################

# Metric storage
delta_distance_tracker=SampleTracker()


# Algorithm Hyperparameterization
alpha = 0.03  # Threshold
noisy_samples = {}
X = 3
epochs = 20
num_samples = len(train_dataset)
noisy_indices = set(list(train_dataset.get_corrupted_indices()))

for iteration in range(X): 
    info_dict = {
        'NC1': [],
        'NC4': [],
        'train_acc': [],
        'eval_acc': []
    }
    # Shuffle del trainset to inject new temporal dependencies
    indices = torch.randperm(num_samples)
    shuffled_train_dataset = Subset(train_dataset, indices)

    # Take reference between index of the dataset and shuffled version
    index_map = {idx: i for i, idx in enumerate(indices)}

    #RESET MODEL
    model, feature_size = reset_model(mode=mode+version, classes=num_classes)
    model = model.to(device)

    # Model parameters
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4, betas=(0.9,0.999))

    # Configurazione del DataLoader
    trainloader = DataLoader(shuffled_train_dataset, batch_size=256, shuffle=False, num_workers=2, drop_last=False)
    
    for epoch in range(epochs):
        # Train
        mu_G_train, mu_c_train, train_acc, eval_acc, nc4, mu_c_test = compute_epoch_info(model, trainloader, testloader, optimizer, criterion, num_classes, feature_size, mode=mode)

        # NC1 and stability metrics
        Sigma_W = compute_Sigma_W_and_distance(model, mu_c_train, trainloader, delta_distance_tracker, mode=mode)
        Sigma_B = compute_Sigma_B(mu_c_train, mu_G_train)
        collapse_metric = float(np.trace(Sigma_W.cpu() @ scilin.pinv(Sigma_B.cpu().numpy())) / len(mu_c_train))

        print('[model: '+str(iteration+1)+'/'+str(X)+' epoch:'+ str(epoch + 1)+'/'+str(epochs) +' | train top1:' + str(train_acc) +' | eval acc:' + str(eval_acc) +' | NC1:' + str(collapse_metric))


    # Calcola le distanze medie per ogni campione lungo le epoche
    delta_distances = delta_distance_tracker.tensorize()  # Tensor (N_samples, n_epochs)
    show_mean_var_relevations(delta_distances, mode, version, dataset_name, train_dataset.get_corrupted_indices(), index_map)
    delta_distance_tracker.reset_dict() # reset dict for next iteration

    means = delta_distances.mean(dim=1)  # Manteniamo il tutto come tensore di PyTorch (N_samples)

    # Ordina 'means' e ottieni gli indici ordinati
    sorted_indices = torch.argsort(means)
    sorted_means = means[sorted_indices]

    # Prendi i primi int(N_samples * alpha) indici
    boundary = int(num_samples * alpha)
    top_indices = sorted_indices[:boundary]

    # Converti top_indices da indici mescolati a indici originali
    original_top_indices = [indices[idx].item() for idx in top_indices]

    # Filtra i campioni già presenti nel set di noisy_samples
    new_noisy_samples = set(original_top_indices) - noisy_samples.keys()

    # Aggiungi i nuovi campioni noisy al set
    noisy_samples.update({i: train_dataset[i] for i in new_noisy_samples})

    # Crea il dataset finale escludendo i campioni rumorosi
    final_indices = list(set(range(num_samples)) - noisy_samples.keys())

    final_indices = set(final_indices)
    correctly_identified_noisy = set(noisy_samples.keys()).intersection(noisy_indices)


    print("Noisy Sample Detected: "+str(len(correctly_identified_noisy))+"/"+str(boundary))
    print("Cumulative set size: "+str(len(noisy_samples))+" samples")

#Then we can use the final dataset to calculate a smoothed label for noisy indices!

# 1)  Train on clean dataset
# 2) Extract labels from noisy label data and use it for retrain the model on smoothed labels (pseudo-labeling)

