import torch


def compute_anomaly_map(patch_embeddings, memory_bank):
    """
    Computes anomaly map (per patch score)

    Args:
        patch_embeddings: (N, D)
        memory_bank: (M, D)

    Returns:
        min_distances: (N,) anomaly score per patch
    """

    # ensure same device
    memory_bank = memory_bank.to(patch_embeddings.device)

    # compute pairwise distances
    distances = torch.cdist(patch_embeddings, memory_bank)

    # nearest neighbor distance (per patch)
    min_distances = torch.min(distances, dim=1)[0]

    return min_distances


def compute_image_score(min_distances, top_k=50):
    """
    Computes final anomaly score for the image

    Args:
        min_distances: (N,)
        top_k: number of most anomalous patches to consider

    Returns:
        score: scalar
    """

    # sort distances (descending → most anomalous first)
    sorted_distances, _ = torch.sort(min_distances, descending=True)

    # take top K anomalous patches
    k = min(top_k, len(sorted_distances))
    topk = sorted_distances[:k]

    # average → stable score
    score = torch.mean(topk)

    return score


def compute_anomaly_score(patch_embeddings, memory_bank, k=50):

    memory_bank = memory_bank.to(patch_embeddings.device)

    distances = torch.cdist(patch_embeddings, memory_bank)

    min_distances = torch.min(distances, dim=1)[0]

    # 🔥 KEY CHANGE: top-k instead of max
    topk_values, _ = torch.topk(min_distances, k=min(k, len(min_distances)))

    image_score = torch.mean(topk_values)

    return min_distances