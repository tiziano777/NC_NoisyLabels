import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset='cifar10'
model='mnas'
version='075'

with open(f'./NC_experiments/no_noise/{dataset}/{model}{version}/{model}{version}_{dataset}_results.pkl', 'rb') as file:
    info_dict = pickle.load(file)

def show_plot(info_dict, mode, version, dataset_name):
    # Impostare uno stile di base e una palette di colori
    sns.set_style("whitegrid")  # Stile con griglia bianca
    sns.set_palette("Set2")  # Palette di colori accattivante

    # Creare una figura e gli assi per il plot 2x2
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Top-left: Accuracy Top-1
    sns.lineplot(x=np.arange(1, len(info_dict['train_acc']) + 1),
                 y=info_dict['train_acc'],
                 ax=axs[0, 0],
                 marker="o",
                 linewidth=2,
                 color='blue',
                 label='Train Accuracy')  

    sns.lineplot(x=np.arange(1, len(info_dict['eval_acc']) + 1),
             y=info_dict['eval_acc'],
             ax=axs[0, 0],
             marker="o",
             linewidth=2,
             color='orange',
             label='Test Accuracy')
    axs[0, 0].set_title('Accuracy', fontsize=16, fontweight='bold')
    axs[0, 0].set_xlabel('Epochs', fontsize=14)
    axs[0, 0].set_ylabel('Accuracy', fontsize=14)

    

    # center-top: NC1
    sns.lineplot(x=np.arange(1, len(info_dict['NC1']) + 1),
                 y=info_dict['NC1'],
                 ax=axs[0, 1],
                 marker="s",
                 linewidth=2,
                 color='green')  # Cambia il colore e il marker
    axs[0, 1].set_title('NC1 Metric', fontsize=16, fontweight='bold')
    axs[0, 1].set_xlabel('Epochs', fontsize=14)
    axs[0, 1].set_ylabel('Metric Value', fontsize=14)

    # Bottom-right: NC3
    sns.lineplot(x=np.arange(1, len(info_dict['NC2']) + 1),
                 y=info_dict['NC2'],
                 ax=axs[1, 0],
                 marker="^",
                 linewidth=2,
                 color='red',
                 label='NC2')  # Cambia il colore e il marker
    
    sns.lineplot(x=np.arange(1, len(info_dict['NC3']) + 1),
                 y=info_dict['NC3'],
                 ax=axs[1, 0],
                 marker="d",
                 linewidth=2,
                 color='purple',
                 label='NC3')  # Cambia il colore e il marker
    axs[1, 0].set_title('NC2-3 Metric', fontsize=16, fontweight='bold')
    axs[1, 0].set_xlabel('Epochs', fontsize=14)
    axs[1, 0].set_ylabel('Metric Value', fontsize=14)


    # center-low: NC4
    sns.lineplot(x=np.arange(1, len(info_dict['NC4']) + 1),
                 y=info_dict['NC4'],
                 ax=axs[1, 1],
                 marker="d",
                 linewidth=2,
                 color='black')  # Cambia il colore e il marker
    axs[1, 1].set_title('NC4 Metric', fontsize=16, fontweight='bold')
    axs[1, 1].set_xlabel('Epochs', fontsize=14)
    axs[1, 1].set_ylabel('Metric Value', fontsize=14)
    '''
    # upper-right: memorization
    sns.lineplot(x=np.arange(1, len(info_dict['mem']) + 1),
                 y=info_dict['mem'],
                 ax=axs[1, 2],
                 marker="d",
                 linewidth=2,
                 color='yellow')  # Cambia il colore e il marker
    axs[1, 2].set_title('Memorization Metric', fontsize=16, fontweight='bold')
    axs[1, 2].set_xlabel('Epochs', fontsize=14)
    axs[1, 2].set_ylabel('Metric Value', fontsize=14)
    '''

    # Aggiustiamo lo spazio tra i plot per una migliore visualizzazione
    plt.tight_layout()
    plt.show()
    # Salvare il grafico in un file PNG
    plt.savefig(f'epoch_{len(info_dict["train_acc"])}_{mode}{version}_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

show_plot(info_dict, model, version, dataset)