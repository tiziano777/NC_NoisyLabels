import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Assumiamo che tu abbia già il tuo modello, i dati e l'array degli indici dei campioni rumorosi
model = # ... il tuo modello ...
data_loader = # ... il tuo data loader ...
noisy_indices = # ... l'array degli indici dei campioni rumorosi ...

# Estrai gli embedding
embeddings = []
labels = []

with torch.no_grad():
    for batch in data_loader:
        inputs, targets = batch
        outputs = model(inputs)
        embeddings.extend(outputs.cpu().numpy())
        labels.extend(targets.cpu().numpy())

# Applica t-SNE per ridurre a 3 dimensioni
tsne_embeddings = TSNE(n_components=3).fit_transform(embeddings)

# Crea uno scrittore TensorBoard
writer = SummaryWriter('runs/embedding_visualization_3d')

# Prepara i colori per i punti
colors = ['green' if i not in noisy_indices else 'red' for i in range(len(tsne_embeddings))]

# Scrivi gli embedding su TensorBoard
for i, (embedding, label) in enumerate(zip(tsne_embeddings, labels)):
    writer.add_embedding(
        [embedding],
        metadata=[f'Label: {label}'],
        tag=f'Embedding_{i}',
        global_step=i,
        colors=[colors[i]]
    )

# Chiudi lo scrittore
writer.close()

# Visualizzazione 3D con Plotly
fig = go.Figure(data=[
    go.Scatter3d(
        x=tsne_embeddings[:, 0],
        y=tsne_embeddings[:, 1],
        z=tsne_embeddings[:, 2],
        mode='markers',
        marker=dict(color=colors),
        text=[f'Label: {label}' for label in labels]
    )
])

fig.update_layout(
    title="3D t-SNE Visualization of Embeddings",
    scene=dict(
        xaxis_title='Component 1',
        yaxis_title='Component 2',
        zaxis_title='Component 3'),
)

fig.show()