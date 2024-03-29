import torch


CUDA = 'cuda'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else 'cpu')
SEED = 1234