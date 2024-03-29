import torch


CUDA = 'cuda'
DEVICE = torch.device(CUDA)
# DEVICE = torch.device(CUDA if torch.cuda.is_available() else 'cpu')
# SEED = 2222
SEEDS = [2222,1792,9334,5354,6789]