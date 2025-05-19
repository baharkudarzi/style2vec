

#######################################################################################
#######################################################################################
################################# eval_metrics.py #####################################
#######################################################################################
#######################################################################################


import torch
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from style2vec_model import StyleEmbeddingNet

def main():
    # Configuration
    device = torch.device('cpu')
    embedding_dim = 128
    num_samples = 500  # number of samples to evaluate

    # 1) Load trained embedding network
    model = StyleEmbeddingNet(embedding_dim=embedding_dim)
    model.load_state_dict(
        torch.load('style2vec_fmnist_embedding.pth', map_location=device)
    )
    model.eval()

    # 2) Define transforms (same as training/eval)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # grayscale → 3-channel
    ])

    # 3) Load FashionMNIST and collect embeddings + labels
    dataset = FashionMNIST(root='.', train=True, download=False)
    embeddings = []
    labels = []

    with torch.no_grad():
        for i in range(num_samples):
            img, label = dataset[i]
            x = transform(img).unsqueeze(0).to(device)
            emb = model(x).cpu().numpy().squeeze()
            embeddings.append(emb)
            labels.append(label)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # 4) Compute cluster quality metrics
    sil_score = silhouette_score(embeddings, labels)
    db_score  = davies_bouldin_score(embeddings, labels)

    print(f"Silhouette Score ({num_samples} samples): {sil_score:.4f}")
    print(f"Davies–Bouldin Index ({num_samples} samples): {db_score:.4f}")

if __name__ == "__main__":
    main()
