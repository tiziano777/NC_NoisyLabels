import torch.nn as nn
import torch

class SigmoidWeightingBCELoss(nn.Module):
    def __init__(self, min_weight=0.05, max_weight=1.5, scale=1.1, stability_factor=5.0):
        """
        :param min_weight: Peso minimo per i campioni con valori di metrica più negativi
        :param max_weight: Peso massimo per i campioni con valori di metrica più positivi
        :param scale: Scala per la funzione sigmoide, controlla l'effetto della metrica sulla ponderazione
        :stability_factor: evita saturazione sigmoide per i logits
        """
        super(SigmoidWeightingBCELoss, self).__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.scale = scale
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.stability_factor= stability_factor

    def forward(self, outputs, targets, metrics):
        """
        :param outputs: model predictions
        :param targets: Training Labels
        :param metrics: Tensor (N,) with weighting logits
        :return: Our BCE loss with Custom Sigmoid weighting 
        """
        
        metrics = metrics / self.stability_factor
        sigmoid_weights = torch.sigmoid(self.scale * metrics)
        weights = self.min_weight + (self.max_weight - self.min_weight) * sigmoid_weights

        # Calcola la loss 
        loss = self.criterion(outputs, targets)

        # Apply the sample weights: weights is [256], loss is [256, 10]
        # weights are broadcast to [256, 1] when multiplied with the loss
        loss = loss * weights.view(-1, 1)

        # Finally, reduce the loss across all samples and classes
        loss = loss.mean() 

        return loss