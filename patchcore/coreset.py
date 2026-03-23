import torch

def coreset_sampling(memory_bank, sampling_ratio=0.1):

    n_samples = int(memory_bank.shape[0] * sampling_ratio)

    indices = torch.randperm(memory_bank.shape[0])[:n_samples]

    coreset = memory_bank[indices]

    return coreset