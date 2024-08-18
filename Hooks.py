
class FeaturesHook():
    def __init__(self):
        self.features_F = []  # Lista per memorizzare le feature F
        self.features_H = []  # Lista per memorizzare le feature H

    def get_feature_F(self, module, input, output):
        self.features_F.append(output.detach().squeeze())
        #self.features_F.append(output.detach().squeeze(1).squeeze(2))  # Rimuove le dimensioni extra 

    def get_feature_H(self, module, input, output):
        self.features_H.append(output.detach())

    def clear(self):  # Metodo per svuotare le liste delle features
        self.features_F = []
        self.features_H = []