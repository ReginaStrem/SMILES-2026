# Изменения в aggregation.py

## Функция `aggregate()` — строки 40–52

### Оригинал (GitHub):
```python
    # Default: last real token of the final transformer layer.
    layer = hidden_states[-1]          # (seq_len, hidden_dim)

    # Find the index of the last real (non-padding) token.
    real_positions = attention_mask.nonzero(as_tuple=False)  # (n_real, 1)
    last_pos = int(real_positions[-1].item())                 # scalar index

    feature = layer[last_pos]          # (hidden_dim,)

    return feature
```

### Наша версия:
```python
    hidden_states = hidden_states.cpu()
    attention_mask = attention_mask.cpu()

    n_layers = hidden_states.size(0)

    # Last real token position
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())

    # Last token of the final layer
    feature = hidden_states[-1, last_pos]

    return feature
```

### Что изменено:
- Добавлены строки `hidden_states = hidden_states.cpu()` и `attention_mask = attention_mask.cpu()` — предотвращают ошибку device mismatch (тензоры на GPU, а агрегация на CPU)
- Убрана промежуточная переменная `layer`, обращение напрямую `hidden_states[-1, last_pos]`
- Добавлена `n_layers` (используется в `extract_geometric_features`)

---

## Функция `extract_geometric_features()` — строки 78–123

### Оригинал (GitHub):
```python
    # Placeholder: returns an empty tensor (no geometric features).
    return torch.zeros(0)
```

### Наша версия:
```python
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
```

### Что изменено:
- Заглушка `torch.zeros(0)` заменена на 56 hand-crafted признаков:
  - **25 признаков**: L2 нормы mean-pooled представлений по каждому слою
  - **24 признака**: косинусные сходства между соседними слоями (representation drift)
  - **3 признака**: средняя дисперсия активаций по токенам для 3 слоёв
  - **3 признака**: L2 нормы last-token для 3 слоёв
  - **1 признак**: нормализованная длина последовательности
- Добавлены `.cpu()` вызовы для предотвращения device mismatch
- `cos_sim.reshape(1)` вместо `cos_sim.squeeze(0)` — предотвращает RuntimeError при скалярном тензоре
