


#######################################################################################
#######################################################################################
################################ demo_neighbors.py ####################################
#######################################################################################
#######################################################################################



import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from style2vec_model import StyleEmbeddingNet

def main():
    # Configuration
    device = torch.device('cpu')
    embedding_dim = 128
    gallery_size = 500  # how many images to embed for the gallery
    topk = 3            # number of nearest neighbors to display

    # 1) Load your trained embedding network
    model = StyleEmbeddingNet(embedding_dim=embedding_dim)
    model.load_state_dict(
        torch.load('style2vec_fmnist_embedding.pth', map_location=device)
    )
    model.eval()

    # 2) Define the same transforms you used for training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1-channel → 3-channel
    ])

    # 3) Load the FashionMNIST test set
    test_ds = FashionMNIST(root='.', train=False, download=True)
    gallery_images = []
    gallery_labels = []
    embeddings = []

    with torch.no_grad():
        for i in range(gallery_size):
            img, lbl = test_ds[i]
            gallery_images.append(img)
            gallery_labels.append(lbl)
            x = transform(img).unsqueeze(0).to(device)
            emb = model(x).cpu().squeeze()
            embeddings.append(emb)

    embeddings = torch.stack(embeddings)  # shape: [gallery_size, embedding_dim]

    # 4) Pick a random anchor from the gallery
    anchor_idx = np.random.randint(gallery_size)
    anchor_img = gallery_images[anchor_idx]
    anchor_lbl = gallery_labels[anchor_idx]
    anchor_emb = embeddings[anchor_idx].unsqueeze(0)  # shape: [1, D]

    # 5) Compute cosine similarities between anchor and all gallery embeddings
    sims = F.cosine_similarity(anchor_emb, embeddings)  # shape: [gallery_size]
    sims[anchor_idx] = -1.0  # exclude self from neighbors
    _, nn_idxs = sims.topk(topk)

    # 6) Plot the anchor and its top-k nearest neighbors
    fig, axes = plt.subplots(1, topk + 1, figsize=(4*(topk+1), 4))
    axes[0].imshow(anchor_img, cmap='gray')
    axes[0].set_title(f"Anchor\nLabel: {anchor_lbl}")
    axes[0].axis('off')

    for rank, idx in enumerate(nn_idxs, start=1):
        img = gallery_images[idx]
        lbl = gallery_labels[idx]
        sim = sims[idx].item()
        ax = axes[rank]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"NN {rank}\nLabel: {lbl}\nSim: {sim:.2f}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
