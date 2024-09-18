import torch
from collections import defaultdict

class SampleLabelTracker:
    def __init__(self):
        self.result_dict = {}

    def add_epoch_results(self, batch_idx, batch_results):
        # batch_results ha dimensione (N, 10)
        if batch_idx not in self.result_dict:
            # Aggiungi una dimensione esplicita per rappresentare l'epoca: (N, 1, 10)
            self.result_dict[batch_idx] = batch_results.unsqueeze(1)
        else:
            # Concatenazione lungo la dimensione 1 per aggiungere una nuova epoca (N, epochs, 10)
            self.result_dict[batch_idx] = torch.cat(
                [self.result_dict[batch_idx], batch_results.unsqueeze(1)], dim=1)

    def tensorize(self):
        tensorize_dict = [self.result_dict[i] for i in sorted(self.result_dict.keys())]
        # Concatenare lungo la dimensione 0 per combinare i batch, mantenendo la dimensione delle epoche
        return torch.cat(tensorize_dict, dim=0)  # Tensor (N_samples_totali, epochs, 10)


    def extract_epoch(self, epoch_idx):
        """
        Estrai tutte le rilevazioni di una specifica epoca.
        
        :param tensorized_data: Tensor (N_samples, epochs,classes)
        :param epoch_idx: Epoch index 
        :return: Tensor (N_samples, classes) fileld with distances of each sample at epoch epoch_idx
        """
        tensorized_data= self.tensorize()
        return tensorized_data[:, epoch_idx, :]

    def get_results(self):
        return self.result_dict

    def reset_dict(self):
        self.result_dict = {}