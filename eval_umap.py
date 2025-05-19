


#######################################################################################
#######################################################################################
#################################### eval_umap.py #####################################
#######################################################################################
#######################################################################################



import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from style2vec_model import StyleEmbeddingNet

# 1) Load your trained embedding network
device = torch.device('cpu')
model  = StyleEmbeddingNet(128)
model.load_state_dict(torch.load('style2vec_fmnist_embedding.pth', map_location=device))
model.eval()

# 2) Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
])

# 3) Load data and extract embeddings
dataset = FashionMNIST(root='.', train=True, download=False)
n_samples = 500
embs, labels = [], []
with torch.no_grad():
    for i in range(n_samples):
        img, lbl = dataset[i]
        x = transform(img).unsqueeze(0).to(device)
        e = model(x).cpu().numpy().squeeze()
        embs.append(e); labels.append(lbl)
embs = np.vstack(embs)

# 4) UMAP projection
proj = umap.UMAP(n_components=2, random_state=42).fit_transform(embs)

# 5) Plot
plt.figure(figsize=(8,6))
for c in sorted(set(labels)):
    idx = [i for i, l in enumerate(labels) if l==c]
    plt.scatter(proj[idx,0], proj[idx,1], label=str(c), s=10)
plt.legend(title="Class", bbox_to_anchor=(1,1))
plt.title("UMAP of Fashion-MNIST Triplet Embeddings")
plt.xlabel("UMAP Dim 1"); plt.ylabel("UMAP Dim 2")
plt.tight_layout()
plt.show()
