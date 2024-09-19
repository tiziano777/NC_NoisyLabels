import torch.nn as nn

class SigmoidWeightingBCELoss(nn.Module):
    def __init__(self, min_weight=0.01, max_weight=1.5, scale=1.1):
        """
        :param min_weight: Peso minimo per i campioni con valori di metrica più negativi
        :param max_weight: Peso massimo per i campioni con valori di metrica più positivi
        :param scale: Scala per la funzione sigmoide, controlla l'effetto della metrica sulla ponderazione
        """
        super(SigmoidWeightingBCELoss, self).__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.scale = scale

    def forward(self, outputs, targets, metrics):
        """
        :param outputs: model predictions
        :param targets: Training Labels
        :param metrics: Tensor (N,) with weighting logits
        :return: Our BCE loss with Custom Sigmoid weighting 
        """
        # BCE with logits Loss standrd implementation
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss = criterion(outputs, targets)
        
        # Sigmoid
        sigmoide_weights = torch.sigmoid(self.scale * metrics)
        # Parameterized Sigmoid
        weights = self.min_weight + (self.max_weight - self.min_weight) * sigmoide_weights
        
        # Apply weights to BCE loss
        weighted_loss = loss * weights
        
        # Restituisce la perdita media
        return weighted_loss.mean()