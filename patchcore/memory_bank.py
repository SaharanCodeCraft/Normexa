import torch


class MemoryBank:

    def __init__(self):
        self.features = []

    def add(self, patch_embeddings):

        self.features.append(patch_embeddings.cpu())

    def build(self):

        self.features = torch.cat(self.features)

        return self.features