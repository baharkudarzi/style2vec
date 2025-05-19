


#######################################################################################
#######################################################################################
############################## train_hard_weighted.py #################################
#######################################################################################
#######################################################################################



import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from style2vec_model import StyleEmbeddingNet
from torch.nn import TripletMarginLoss

def pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cdist(embeddings, embeddings, p=2)

def train_hard_weighted():
    device     = torch.device('cpu')
    batch_size = 64
    epochs     = 5
    margin     = 0.5
    lr         = 1e-3

    # 1) Transforms + dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ])
    dataset = FashionMNIST(root='.', train=True, download=True, transform=transform)

    # 2) Build sample weights to oversample classes [0,2,4,6]
    #    Give those classes weight=2.0, all others weight=1.0
    class_weights = {cls: (2.0 if cls in [0,2,4,6] else 1.0) for cls in range(10)}
    # FashionMNIST stores labels in `targets`
    labels = dataset.targets.numpy()
    sample_weights = [class_weights[int(lbl)] for lbl in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 3) DataLoader with our sampler
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0
    )

    # 4) Model, loss, optimizer
    model     = StyleEmbeddingNet(embedding_dim=128).to(device)
    criterion = TripletMarginLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) Training loop (batch-hard mining)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)  # [B, D]

            dist_mat = pairwise_distances(embeddings)
            eq_mask  = labels.unsqueeze(1) == labels.unsqueeze(0)
            pos_mask = eq_mask.float()
            pos_mask.fill_diagonal_(0)

            # hardest positives (max dist same class)
            hardest_pos_dist, hardest_pos_idx = (dist_mat * pos_mask).max(dim=1)

            # hardest negatives (min dist different class)
            max_dist = dist_mat.max().item() + 1.0
            neg_dist = dist_mat + pos_mask * max_dist
            hardest_neg_dist, hardest_neg_idx = neg_dist.min(dim=1)

            anchor   = embeddings
            positive = embeddings[hardest_pos_idx]
            negative = embeddings[hardest_neg_idx]

            loss = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}\n")

    # 6) Save
    torch.save(model.state_dict(), 'style2vec_fmnist_hard_weighted.pth')
    print("Saved weighted & hard-mined model as 'style2vec_fmnist_hard_weighted.pth'")

if __name__ == "__main__":
    train_hard_weighted()
