import torch
import torch.nn.functional as F

# Classical Label Smoothing funzction
def label_smoothing(targets, num_classes, smoothing=0.1):
    # Calculate the smoothing value for each class
    smoothing_value = smoothing / (num_classes - 1)
    
    # Create a one-hot encoded version of the targets
    one_hot_targets = F.one_hot(targets, num_classes=num_classes).float()
    
    # Apply label smoothing
    smooth_targets = one_hot_targets * (1 - smoothing) + smoothing_value
    
    return smooth_targets

#early learning NC based label smoothing
def early_learning_label_smoothing(distances, topk=3):
    """
    :param distances: Tensor (N, C) where N is the number of samples and C is the number of classes (distances from centroids in penultimte Feature layer)
    :param topk: Number of top probabilities to keep
    :return: Tensor (N, C) of smoothed probabilities, where only top-k values are kept and the rest are set to zero
    """
    
    # Step 1: Rescale distances to the range [0, 1] using a linear transformation
    # Subtract distances from max distance and divide by max for each sample
    max_distances = distances.max(dim=1, keepdim=True).values
    rescaled_distances = 1 - distances / max_distances
    
    # Step 2: Select the top-k probabilities for each sample
    top_k_values, top_k_indices = torch.topk(rescaled_distances, k=topk, dim=1)
    
    # Step 3: Create a zero tensor of the same shape as rescaled_distances
    smoothed_probs = torch.zeros_like(rescaled_distances)
    
    # Step 4: Place the top-k values in the corresponding positions
    smoothed_probs.scatter_(1, top_k_indices, top_k_values)
    
    # Step 5: Normalize the top-k probabilities to ensure their sum is 1
    smoothed_probs_sum = smoothed_probs.sum(dim=1, keepdim=True)
    smoothed_probs = smoothed_probs / smoothed_probs_sum
    
    return smoothed_probs

# Early learning + late separation Smoothing


def NC_based_smoothing(early_smoothing, delta_distances, fake_labels, max_adjustment=0.6):
    '''
    :early_smoothing: Tensor(N,C) with smoothed probabilities given by early learning distances from centroids
    :delta_distances: Tensor(N,) that contains a metric between a certain range [x(negative), 0, y(positive)]
    :fake_labels: Tensor(N,) Contains basic noisy labels index [0, C-1]
    :max_adjustment: Maximum adjustment for smoothing
    '''
    def calculate_percentile(tensor, q):
        """
        Calculate the q-th percentile of a tensor.
        :tensor: Tensor(N,) input tensor.
        :q: Percentile to calculate (value between 0 and 100).
        """
        # Ensure tensor is flattened
        tensor = tensor.flatten()
        
        # Sort the tensor
        sorted_tensor = torch.sort(tensor).values
        
        # Calculate the index for the percentile
        k = max(0, min(round(0.01 * q * (tensor.numel() - 1)), tensor.numel() - 1))
        
        # Return the value at the k-th index
        return sorted_tensor[k]

    
    # Standardize delta_distances
    mean = delta_distances.mean()
    std = delta_distances.std()
    standardized_distances = (delta_distances - mean) / std

    # Define percentiles for identifying tails
    left_threshold = calculate_percentile(standardized_distances, 15)
    right_threshold = calculate_percentile(standardized_distances, 75)

    # Initialize smoothed probabilities
    smoothed_probs = early_smoothing.clone()

    # Find outliers
    mask_left = standardized_distances < left_threshold
    mask_right = standardized_distances > right_threshold

    # Process left tail (noisy samples)
    if mask_left.any():
        for i in range(smoothed_probs.size(0)):
            if mask_left[i]:
                target_class = fake_labels[i]
                adjustment = min(max_adjustment, smoothed_probs[i, target_class].item())
                
                # Reduce probability of the target class
                smoothed_probs[i, target_class] = max(0, smoothed_probs[i, target_class] - adjustment)
                
                # Distribute the adjustment among other classes
                non_zero_classes = smoothed_probs[i] > 0
                if non_zero_classes.sum() > 0:
                    smoothed_probs[i, non_zero_classes] += adjustment / non_zero_classes.sum()

    # Process right tail (clean samples)
    if mask_right.any():
        # Assicurati che i tensori siano sulla CPU (se necessario)
        smoothed_probs = smoothed_probs.cpu()
        mask_right = mask_right.cpu()

        # Step 1: Identifica la classe con la probabilità più alta per i campioni in mask_right
        highest_class = torch.argmax(smoothed_probs[mask_right], dim=1)

        # Step 2: Applica la funzione di label smoothing con fattore 0.1
        num_classes = smoothed_probs.size(1)  # Numero di classi

        # Usa la tua funzione di label smoothing per questi campioni
        smoothed_labels = label_smoothing(highest_class, num_classes, smoothing=0.1)

        # Step 3: Aggiorna smoothed_probs con i valori smoothed
        smoothed_probs[mask_right] = smoothed_labels

        print(len(smoothed_probs[mask_right]), smoothed_probs[mask_right])

    # Check for label swaps
    original_labels = torch.argmax(early_smoothing, dim=1).cpu()
    new_labels = torch.argmax(smoothed_probs, dim=1)
    num_swaps = (original_labels != new_labels).sum().item()

    print(f'Number of label swaps: {num_swaps}')

    #TODO: Label swap indices can be converted in a different label, one hot vector with classical label smoothing factor of 0.2.
    # other labels cannot be modificated 

    num_classes = early_smoothing.size(1)  # Number of classes inferred from the shape
    swapped_indices = (original_labels != new_labels).nonzero(as_tuple=True)[0]  # Indices of swapped labels

    if len(swapped_indices) > 0:
        # Apply classical label smoothing to the swapped indices
        swapped_labels = new_labels[swapped_indices]
        
        # Get smoothed probabilities for swapped labels
        smoothed_swaps = label_smoothing(swapped_labels, num_classes, smoothing=0.2)
        
        # Update the smoothed_probs tensor with the smoothed values for swapped indices
        smoothed_probs[swapped_indices] = smoothed_swaps

    return smoothed_probs





