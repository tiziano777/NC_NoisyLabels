import torch
import time
import numpy as np
import seaborn as sns
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
from tsnecuda import TSNE
from Autoencoder import Autoencoder
from Hooks import FeaturesHook

def save_feature_F(module, input, output):
    features.get_feature_F(module, input, output)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def select_model(model,mode):
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



'''
class EmbeddingVisualizer2D:
    def __init__(self, model, feature_size, mode, version, dataset_name, epoch, device, data_loader, noisy_indices, mu_c, mu_c_weighted, mu_c_clean):
        self.model = model
        self.feature_size = feature_size
        self.mode = mode
        self.version = version
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.device = device
        self.data_loader = data_loader
        self.noisy_indices = noisy_indices
        self.mu_c = mu_c
        self.mu_c_weighted = mu_c_weighted
        self.mu_c_clean = mu_c_clean
        self.embeddings = []
        self.labels = []
        self.colors_map = {
            0: 'brown', 1: 'yellow', 2: 'purple', 3: 'blue', 4: 'green',
            5: 'gray', 6: 'orange', 7: 'lavender', 8: 'lightblue', 9: 'aqua'
        }
        self.noisy_color = 'magenta'  # Colore acceso per i noisy samples
    
    @measure_time
    def extract_embeddings(self):

        global features
        features = FeaturesHook()
        features.clear()
        handle_F= select_model(self.model,self.mode)

        #extract embedding features
        with torch.no_grad():
            self.model.eval()
            for batch in self.data_loader:
                inputs, raw_labels = batch
                inputs = inputs.to(self.device)
                raw_labels = torch.tensor(raw_labels).to(self.device)

                targets = raw_labels[:, 1:, :].squeeze().to(self.device)
                outputs = self.model(inputs)

                self.labels.extend(targets.cpu().numpy())

        # Take embeddings and reduce dimensionality
        autoencoder=Autoencoder(input_dim=self.feature_size, latent_dim=20)

        # Utilizza torch.stack per combinare i tensori lungo una nuova dimensione
        features.features_F=features.features_F[:-1]
        feature_tensor = torch.stack(features.features_F).to(self.device)
        autoencoder.training_phase(features=feature_tensor,device=self.device)
        autoencoder.eval()
        self.embeddings.extend(autoencoder.encoder(feature_tensor).detach().cpu().numpy())

    @measure_time
    def apply_tsne(self):
        embeddings_tensor = torch.tensor(self.embeddings).detach().numpy()
        embeddings_tensor=embeddings_tensor.reshape(-1, embeddings_tensor.shape[-1])
        self.tsne_embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(embeddings_tensor)

    @measure_time
    def visualize_embeddings(self):
        # Prepara i colori per i punti
        self.labels=self.labels[:len(self.tsne_embeddings)]
        print(len(self.labels),self.tsne_embeddings.shape)
        colors = [
            self.noisy_color if i in self.noisy_indices else self.colors_map[label] for i, label in enumerate(self.labels)
        ]

        # Visualizzazione 2D con Seaborn
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=self.tsne_embeddings[:, 0], y=self.tsne_embeddings[:, 1], hue=colors, palette=colors, legend=False)

        # Aggiungi i centroidi mu_c, mu_c_weighted e mu_c_clean
        self.add_centroids_to_plot(self.mu_c.cpu().numpy(), 'black', 'mu_c')
        self.add_centroids_to_plot(self.mu_c_weighted.cpu().numpy(), 'darkblue', 'mu_c_weighted')
        self.add_centroids_to_plot(self.mu_c_clean.cpu().numpy(), 'red', 'mu_c_clean')

        # Titolo e label degli assi
        plt.title(f"2D t-SNE Visualization of Embeddings (Epoch {self.epoch})")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        # Salva la figura
        plt.savefig("embedding_visualization_2d.png")
        plt.show()


    def add_centroids_to_plot(self, centroids, color, name):
        plt.scatter(centroids[:, 0], centroids[:, 1], color=color, s=100, label=name, edgecolor='k')
        plt.legend()

    def run(self):
        self.extract_embeddings()
        self.apply_tsne()
        self.visualize_embeddings()
'''

class EmbeddingVisualizer2D:
    def __init__(self, model, feature_size, mode, version, dataset_name, epoch, device, data_loader, noisy_indices, mu_c, mu_c_weighted, mu_c_clean):
        self.model = model
        self.feature_size = feature_size
        self.mode = mode
        self.version = version
        self.dataset_name = dataset_name
        self.epoch = epoch
        self.device = device
        self.data_loader = data_loader
        self.noisy_indices = noisy_indices
        self.mu_c = mu_c
        self.mu_c_weighted = mu_c_weighted
        self.mu_c_clean = mu_c_clean
        self.embeddings = []
        self.labels = []
        self.real_targets = []
        self.num_classes = 10  # Numero delle classi
        self.noisy_color = 'red'  # Colore per i noisy samples
        self.clean_color = 'green'  # Colore per i clean samples

    @measure_time
    def extract_embeddings(self):
        global features
        features = FeaturesHook()
        features.clear()
        handle_F = select_model(self.model, self.mode)

        # Extract embedding features
        with torch.no_grad():
            self.model.eval()
            for batch in self.data_loader:
                inputs, raw_labels = batch
                inputs = inputs.to(self.device)
                
                # Estrazione dei real targets e fake targets
                real_targets = raw_labels[:, :1, :].squeeze().to(self.device)
                fake_targets = raw_labels[:, 1:, :].squeeze().to(self.device)
                
                # Salvo i real_targets e fake_targets
                self.real_targets.extend(real_targets.cpu().numpy())
                self.labels.extend(fake_targets.cpu().numpy())

                # Passa attraverso il modello per ottenere gli embeddings
                outputs = self.model(inputs)

        # Dimensionality reduction con Autoencoder
        autoencoder = Autoencoder(input_dim=self.feature_size, latent_dim=20)
        features.features_F = features.features_F[:-1]
        feature_tensor = torch.stack(features.features_F).to(self.device)
        autoencoder.training_phase(features=feature_tensor, device=self.device)
        autoencoder.eval()
        self.embeddings.extend(autoencoder.encoder(feature_tensor).detach().cpu().numpy())

    @measure_time
    def apply_tsne(self):
        embeddings_tensor = torch.tensor(self.embeddings).detach().numpy()
        embeddings_tensor = embeddings_tensor.reshape(-1, embeddings_tensor.shape[-1])
        self.tsne_embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(embeddings_tensor)

    @measure_time
    def visualize_embeddings(self):
        self.labels = np.array(self.labels[:len(self.tsne_embeddings)])
        self.real_targets = np.array(self.real_targets[:len(self.tsne_embeddings)])
        self.tsne_embeddings = np.array(self.tsne_embeddings)

        # Creazione della griglia 2x5
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()

        for i in range(self.num_classes):
            ax = axes[i]
            ax.set_title(f"Class {i}")

            # Identificazione dei campioni clean (verde) e noisy (rosso)
            clean_mask = (self.real_targets == i) & (self.labels == i)
            noisy_mask = (self.real_targets == i) & (self.labels != i) & np.isin(np.arange(len(self.labels)), self.noisy_indices)

            # Plot dei clean samples (verde)
            ax.scatter(self.tsne_embeddings[clean_mask, 0], self.tsne_embeddings[clean_mask, 1], 
                       color=self.clean_color, label='Clean', s=20)

            # Plot dei noisy samples (rosso)
            ax.scatter(self.tsne_embeddings[noisy_mask, 0], self.tsne_embeddings[noisy_mask, 1], 
                       color=self.noisy_color, label='Noisy', s=20)

            # Plot dei centroidi della classe corrente
            ax.scatter(self.mu_c[i, 0].cpu(), self.mu_c[i, 1].cpu(), color='black', s=100, edgecolor='k', label='mu_c')
            ax.scatter(self.mu_c_weighted[i, 0].cpu(), self.mu_c_weighted[i, 1].cpu(), color='darkblue', s=100, edgecolor='k', label='mu_c_weighted')
            ax.scatter(self.mu_c_clean[i, 0].cpu(), self.mu_c_clean[i, 1].cpu(), color='red', s=100, edgecolor='k', label='mu_c_clean')

            ax.legend()

        # Layout della griglia
        plt.tight_layout()
        plt.savefig("embedding_visualization_2x5.png")
        plt.show()

    def add_centroids_to_plot(self, centroids, color, name, ax):
        ax.scatter(centroids[:, 0], centroids[:, 1], color=color, s=100, label=name, edgecolor='k')
        ax.legend()

    def run(self):
        self.extract_embeddings()
        self.apply_tsne()
        self.visualize_embeddings()


