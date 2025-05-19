
# Style2Vec: Fashion Item Embeddings via Metric Learning

A complete PyTorch pipeline for learning 128-dimensional embeddings of fashion items using triplet loss, batch-hard negative mining, and weighted sampling. Demonstrated on Fashion-MNIST with tools to train, evaluate, visualize, and demo “complete-the-outfit” recommendations.

## Project Overview

**Goal:** Embed images so that “compatible” items (same class/outfit) lie close in embedding space while “incompatible” items lie far apart.

**Key Features:**
- Triplet training with **random** and **batch-hard** negative mining  
- **Weighted sampling** to focus on hard-to-separate classes  
- **t-SNE** & **UMAP** visualizations of learned embeddings  
- **Silhouette** & **Davies–Bouldin** clustering metrics  
- **Nearest-neighbor demo** showing anchor + top-k matches  
- Modular scripts for easy reproduction  

## Setup & Installation

```bash
git clone <your-repo-url>
cd style2vec
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*If you don’t have `requirements.txt`, install manually:*

```bash
pip install torch torchvision numpy matplotlib scikit-learn umap-learn
```

## Usage

### 1. Training

- **Base triplet training (quick debug subset)**
  ```bash
  python train.py
  ```
  Saves: `style2vec_fmnist_embedding.pth`

- **Batch-hard negative mining**
  ```bash
  python train_hard.py
  ```
  Saves: `style2vec_fmnist_hard_embedding.pth`

- **Weighted sampling + batch-hard mining**
  ```bash
  python train_hard_weighted.py
  ```
  Saves: `style2vec_fmnist_hard_weighted.pth`

### 2. Demo: Nearest-Neighbor Outfit Completion

```bash
python demo_neighbors.py
```

### 3. Embedding Visualization

- **t-SNE**
  ```bash
  python eval_visualization.py
  ```
- **UMAP**
  ```bash
  python eval_umap.py
  ```

### 4. Clustering Metrics

- **Global metrics**
  ```bash
  python eval_metrics.py
  ```
- **Per-class silhouette scores**
  ```bash
  python per_class_silhouette.py
  ```

## Repo Structure

```
style2vec/
├── dataset_fmnist.py
├── test_fmnist_dataset.py
├── style2vec_model.py
├── train.py
├── train_hard.py
├── train_hard_weighted.py
├── demo_neighbors.py
├── eval_visualization.py
├── eval_umap.py
├── eval_metrics.py
├── per_class_silhouette.py
├── requirements.txt
└── README.md
```

## Results

- **Global silhouette score:** ~0.35  
- **Per-class low scores:** Shirt (0.02), Coat (0.20), T-shirt (0.23)

UMAP & t-SNE plots show clear clustering for most classes, with some overlap among upper-body garments.

## Future Work

- Plug in real Polyvore outfit images  
- Add a multi-task classification head  
- Build an interactive Plotly/Bokeh visualizer  
- Provide a Jupyter notebook walkthrough in `notebooks/`


