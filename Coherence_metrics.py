from Hooks import FeaturesHook
import torch

# Aggiungi l'hook al penultimo layer del modello
def select_model(mode, model):
    def save_feature_F(module, input, output):
        features.get_feature_F(module, input, output)

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

# Coherence score for noisy samples
def compute_label_coherence_score(model, mu_c_tensor, noisy_dataloader, norm_factor, mode, device):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode,model)
    coherence_count=0
    num_classes = len(mu_c_tensor)


    model.eval()
    for idx, (inputs, targets) in enumerate(noisy_dataloader):
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
            
            large_value = 1e6  # Usa un valore molto grande che non interferisca con le distanze reali
            distances[torch.arange(batch_size), fake_targets] = large_value
            d_min, _ = torch.min(distances, dim=1)  # find the minimum distance centroid from the remaining ones

            # take only the points that have as minimum distance centroid the real label centroid
            tolerance = 1e-7
            coherence_count += (torch.abs(d_min - real_centroid_distances) < tolerance).sum().item()

            #coherence_count += (d_min == real_centroid_distances).sum().item()

    return coherence_count / norm_factor

# Coherence for those samples that are far away from its label centroid
def compute_label_coherence_for_noisy_outliers(model, mu_c_tensor, noisy_dataloader, mode, device):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode, model)
    coherence_count=0
    num_classes = len(mu_c_tensor)
    norm_factor=0

    model.eval()
    for idx, (inputs, targets) in enumerate(noisy_dataloader):
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
            
            centroid_distance = distances[torch.arange(batch_size), fake_targets]  # take distance from 'fake' label class centroid
            real_centroid_distances = distances[torch.arange(batch_size), real_targets] # take distance from real label class centroid
            
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

# Coherence for clean samples
def compute_label_coherence_for_clean_samples( model, mu_c_tensor, noisy_dataloader, mode, device):
    global features
    features = FeaturesHook()

    # Aggiungi l'hook al penultimo layer del modello
    handle_F = select_model(mode, model)
    coherence_count = 0
    num_classes = len(mu_c_tensor)
    norm_factor=0

    model.eval()
    for idx, (inputs, targets) in enumerate(noisy_dataloader):
        with torch.no_grad():
            inputs = inputs.to(device)
            fake_targets = targets[:, 1:, :].squeeze().to(device)
            real_targets = targets[:, :1, :].squeeze().to(device)

            outputs = model(inputs)

            features_batch = features.features_F[-1].squeeze()
            batch_size = len(features_batch)

            # Reset delle liste di features per il prossimo batch
            features.features_F = []
            features.clear()

            # Filtra i campioni clean: quelli dove real_targets == fake_targets
            clean_mask = real_targets == fake_targets
            clean_indices = torch.nonzero(clean_mask).squeeze()

            if len(clean_indices) == 0:
                continue  # Salta il batch se non ci sono campioni clean

            # Estrai i clean samples
            features_batch_clean = features_batch[clean_indices]
            real_targets_clean = real_targets[clean_indices]
            norm_factor += len(real_targets_clean)
            distances = torch.cdist(features_batch_clean, mu_c_tensor)  # calcola tensore delle distanze [B_clean, classes]
            
            # Prendi la distanza dal centroide della label reale
            real_centroid_distances = distances[torch.arange(len(clean_indices)), real_targets_clean] 

            # Maschera per escludere il centroide della label reale
            mask = torch.ones(distances.shape, dtype=bool)
            mask[torch.arange(len(clean_indices)), real_targets_clean] = False  
            
            # Prendi le altre distanze escluse quelle del centroide della label reale
            d_remaining = distances[mask].view(len(clean_indices), num_classes - 1) 
            
            # Trova il centroide più vicino tra i rimanenti
            d_min, _ = torch.min(d_remaining, dim=1)  

            # Incrementa il conteggio della coerenza se la distanza al centroide reale è minore della distanza minima
            coherence_count += (real_centroid_distances < d_min).sum().item()

    # Restituisci lo score normalizzato
    return coherence_count / norm_factor


