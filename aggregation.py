"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(k * hidden_dim,)`` where k is the
        number of selected layers times the number of pooling strategies.
    """
    hidden_states = hidden_states.cpu()
    attention_mask = attention_mask.cpu()

    n_layers = hidden_states.size(0)

    # Last real token position
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    # Last token of the final layer
    feature = hidden_states[-1, last_pos]

    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    n_layers, seq_len, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.cpu()
    attention_mask = attention_mask.cpu()
    mask_float = attention_mask.unsqueeze(1).float()  # (seq_len, 1)
    n_real = attention_mask.sum().clamp(min=1).float()

    geo_features = []

    # 1. Layer-wise L2 norms of mean-pooled representations
    for layer_idx in range(n_layers):
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        mean_pooled = (layer * mask_float).sum(dim=0) / n_real
        geo_features.append(torch.norm(mean_pooled, p=2).unsqueeze(0))

    # 2. Inter-layer cosine similarities (representation drift)
    # Compare consecutive layers
    mean_pools = []
    for layer_idx in range(n_layers):
        layer = hidden_states[layer_idx]
        mean_pooled = (layer * mask_float).sum(dim=0) / n_real
        mean_pools.append(mean_pooled)

    for i in range(n_layers - 1):
        cos_sim = torch.nn.functional.cosine_similarity(
            mean_pools[i].unsqueeze(0), mean_pools[i + 1].unsqueeze(0)
        )
        geo_features.append(cos_sim.reshape(1))

    # 3. Variance of token representations per layer (activation spread)
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        layer = hidden_states[layer_idx]  # (seq_len, hidden_dim)
        real_layer = layer[attention_mask.bool()]  # (n_real, hidden_dim)
        var_per_dim = real_layer.var(dim=0)
        geo_features.append(var_per_dim.mean().unsqueeze(0))

    # 4. Last-token norm per layer (3 selected layers)
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    for layer_idx in [0, n_layers // 2, n_layers - 1]:
        last_vec = hidden_states[layer_idx, last_pos]
        geo_features.append(torch.norm(last_vec, p=2).unsqueeze(0))

    # 5. Sequence length (normalized)
    geo_features.append((n_real / seq_len).unsqueeze(0))

    return torch.cat(geo_features, dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
