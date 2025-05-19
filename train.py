
#######################################################################################
#######################################################################################
###################################### train.py #######################################
#######################################################################################
#######################################################################################


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset_fmnist import TripletFashionMNIST, repeat3
from style2vec_model import StyleEmbeddingNet, TripletNet
from torchvision import transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 32
lr = 1e-4
epochs = 10
embedding_dim = 128
subset_size = 2000  # limit dataset for faster debugging

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.Lambda(repeat3),
])

# Full dataset
full_dataset = TripletFashionMNIST(root='.', train=True, download=True, transform=transform)
# Create a smaller subset for quick iteration
dataset = Subset(full_dataset, list(range(subset_size)))

# DataLoader (num_workers=0 to avoid multiprocessing issues)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Model, loss, optimizer
embedding_net = StyleEmbeddingNet(embedding_dim=embedding_dim)
model         = TripletNet(embedding_net).to(device)
criterion     = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer     = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for batch_idx, (anc, pos, neg) in enumerate(loader):
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        emb_anc, emb_pos, emb_neg = model(anc, pos, neg)

        loss = criterion(emb_anc, emb_pos, emb_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch}/{epochs}] Average Loss: {avg_loss:.4f}")