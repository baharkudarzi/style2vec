

#######################################################################################
#######################################################################################
############################### eval_visualization.py #################################
#######################################################################################
#######################################################################################



import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from style2vec_model import StyleEmbeddingNet

# 1) Load your trained embedding network
device = torch.device('cpu')
embedding_net = StyleEmbeddingNet(embedding_dim=128)
embedding_net.load_state_dict(
    torch.load('style2vec_fmnist_embedding.pth', map_location=device)
)
embedding_net.eval()

# 2) Define the same transforms you used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1-channel → 3-channel
])

# 3) Load the base FashionMNIST (no triplets here)
base = FashionMNIST(root='.', train=True, download=False)

# 4) Sample a subset (e.g. first 500 images)
num_samples = 500
embeddings = []
labels     = []

for i in range(num_samples):
    pil_img, label = base[i]
    img_t = transform(pil_img).unsqueeze(0)  # shape [1,3,224,224]
    with torch.no_grad():
        emb = embedding_net(img_t.to(device)).cpu().numpy().squeeze()
    embeddings.append(emb)
    labels.append(label)

embeddings = np.vstack(embeddings)

# 5) Run t-SNE to project to 2D
tsne = TSNE(n_components=2, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# 6) Plot with Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
for lbl in sorted(set(labels)):
    idxs = [j for j, l in enumerate(labels) if l == lbl]
    ax.scatter(emb_2d[idxs, 0], emb_2d[idxs, 1], label=str(lbl))
ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_title("t-SNE of Fashion-MNIST Triplet Embeddings")
ax.set_xlabel("t-SNE Dim 1")
ax.set_ylabel("t-SNE Dim 2")
plt.tight_layout()
plt.show()
