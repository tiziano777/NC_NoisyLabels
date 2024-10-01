import torch
from collections import defaultdict

class SampleTracker:
    def __init__(self):
        self.result_dict = {}

    def add_epoch_results(self, batch_idx, batch_results):
        # batch_results ha dimensione (N,)
        if batch_idx not in self.result_dict:
            # Se non esiste ancora, aggiungi una dimensione a batch_results per creare (B, 1)
            self.result_dict[batch_idx] = batch_results.unsqueeze(1)
        else:
            # Concatenazione lungo la dimensione 1 per aggiungere una nuova colonna (B, K)
            self.result_dict[batch_idx] = torch.cat(
                [self.result_dict[batch_idx], batch_results.unsqueeze(1)], dim=1)

    def tensorize(self):
        tensorize_dict = [self.result_dict[i] for i in sorted(self.result_dict.keys())]
        return torch.cat(tensorize_dict, dim=0)  # Tensor (N_samples, n_epochs)

    def get_results(self):
        return self.result_dict

    def reset_dict(self):
        self.result_dict = {}
