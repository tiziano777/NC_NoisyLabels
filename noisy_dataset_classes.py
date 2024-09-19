import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset

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

    def return_only_noise_dataset(self,labels):
        data = [self.dataset[i][0] for i in self.dataset.indices]
        noise_dataset=NoiseDataset(data, labels)
        return noise_dataset

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
