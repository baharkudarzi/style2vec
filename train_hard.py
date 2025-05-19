


#######################################################################################
#######################################################################################
#################################### train_hard.py ####################################
#######################################################################################
#######################################################################################



import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from style2vec_model import StyleEmbeddingNet
from torch.nn import TripletMarginLoss

def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance matrix for a batch of embeddings.
    embeddings: Tensor of shape [batch_size, embedding_dim]
    Returns: [batch_size, batch_size] distance matrix.
    """
    # torch.cdist computes pairwise distances
    return torch.cdist(embeddings, embeddings, p=2)

def train_hard_mining():
    # Configuration
    device     = torch.device('cpu')
    batch_size = 64
    epochs     = 5
    margin     = 0.5
    lr         = 1e-3

    # 1) Data transforms and loader (images + labels)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1-channel → 3-channel
    ])
    dataset = FashionMNIST(root='.', train=True, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2) Model, loss, optimizer
    model     = StyleEmbeddingNet(embedding_dim=128).to(device)
    criterion = TripletMarginLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3) Training loop with batch-hard mining
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)  # [B, D]

            # Compute pairwise distance matrix [B, B]
            dist_mat = pairwise_distances(embeddings)

            # Masks for same-class (positive) and different-class (negative)
            labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # [B, B]
            pos_mask = labels_eq.float()
            neg_mask = (~labels_eq).float()

            # Exclude self from positive mask
            pos_mask.fill_diagonal_(0)

            # Hardest positive: maximum distance among same-class
            hardest_pos_dist, hardest_pos_idx = (dist_mat * pos_mask).max(dim=1)

            # Hardest negative: minimum distance among different-class
            # First, mask out positives by adding a large constant
            max_val = dist_mat.max().item() + 1.0
            neg_dist = dist_mat + pos_mask * max_val
            hardest_neg_dist, hardest_neg_idx = neg_dist.min(dim=1)

            # Gather hardest embeddings
            anchor   = embeddings
            positive = embeddings[hardest_pos_idx]
            negative = embeddings[hardest_neg_idx]

            # Compute triplet loss
            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}\n")

    # 4) Save the trained embedding model
    torch.save(model.state_dict(), 'style2vec_fmnist_hard_embedding.pth')
    print("Saved hard-mined embedding model as 'style2vec_fmnist_hard_embedding.pth'")

if __name__ == "__main__":
    train_hard_mining()
