# PS-DLS: Deep Learning Framework for Incomplete Multi-View Clustering

1. ## Project Structure

```text
PS-DLS/
├── train.py              # Main entry for training, transfer and imputation
├── network.py            # Model architecture (Encoders, Decoders, Autoencoders)
├── metric.py             # Clustering evaluation (ACC, NMI, ARI, PUR) and t-SNE
├── utils/                # Utility functions
│   ├── dataloader.py     # Data loading, preprocessing and missing mechanism
│   └── loss.py           # Implementation of contrastive losses
├── figures/              # Output visualization examples
├── save/                 # Placeholder for t-SNE plot outputs
└── models/               # Placeholder for saved model .pth files


2## Requirements

python>=3.8.0

torch>=1.8.0

torchvision>=0.9.0

numpy>=1.21.6

scipy>=1.7.3

scikit-learn>=1.0.2

matplotlib>=3.5.2

cudatoolkit>=11.3


3 ## Training

python train.py --dataset MNIST --miss_rate 0.7 --cp_type 1

--dataset: Dataset name (e.g., MNIST).

--miss_rate: Ratio of missing instances (e.g., 0.1 to 0.7).










































