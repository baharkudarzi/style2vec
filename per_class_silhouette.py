


#######################################################################################
#######################################################################################
############################## per_class_silhouette.py ################################
#######################################################################################
#######################################################################################



import torch
import numpy as np
from sklearn.metrics import silhouette_samples
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from style2vec_model import StyleEmbeddingNet

def main():
    device = torch.device('cpu')
    model = StyleEmbeddingNet(128)
    model.load_state_dict(torch.load('style2vec_fmnist_embedding.pth', map_location=device))
    model.eval()

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])

    # Collect embeddings and labels
    dataset = FashionMNIST(root='.', train=True, download=False)
    n_samples = 1000
    embs, labels = [], []
    with torch.no_grad():
        for i in range(n_samples):
            img, lbl = dataset[i]
            x = transform(img).unsqueeze(0).to(device)
            e = model(x).cpu().numpy().squeeze()
            embs.append(e); labels.append(lbl)
    embs   = np.vstack(embs)
    labels = np.array(labels)

    # Compute silhouette *per sample*
    sil_vals = silhouette_samples(embs, labels)

    # Aggregate per class
    for cls in np.unique(labels):
        cls_scores = sil_vals[labels == cls]
        print(f"Class {cls:2d} — mean silhouette: {cls_scores.mean():.3f}")

if __name__ == "__main__":
    main()
