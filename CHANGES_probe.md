# Изменения в probe.py

## Импорты — строки 13–18

### Оригинал (GitHub):
```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
```

### Наша версия:
```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
```

### Что изменено:
- Добавлен `from sklearn.linear_model import LogisticRegression` — заменили MLP на логистическую регрессию

---

## Класс `HallucinationProbe.__init__()` — строки 28–32

### Оригинал (GitHub):
```python
def __init__(self) -> None:
    super().__init__()
    self._net: nn.Sequential | None = None  # built lazily in fit()
    self._scaler = StandardScaler()
    self._threshold: float = 0.5  # tuned by fit_hyperparameters()
```

### Наша версия:
```python
def __init__(self) -> None:
    super().__init__()
    self._scaler = StandardScaler()
    self._clf: LogisticRegression | None = None
    self._threshold: float = 0.5
```

### Что изменено:
- Убран `self._net` (MLP сеть) — заменён на `self._clf` (LogisticRegression)
- Убраны `self._pca`, `self._pca_n_components` — PCA не нужен для LogReg с L2

---

## Метод `_build_network()` — удалён

### Оригинал (GitHub):
```python
def _build_network(self, input_dim: int) -> None:
    self._net = nn.Sequential(
        nn.Linear(input_dim, 1),
    )
```

### Наша версия:
Метод полностью удалён — LogisticRegression строит модель внутри себя.

---

## Метод `forward()` — строки 34–35

### Оригинал (GitHub):
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self._net is None:
        raise RuntimeError("Network has not been built yet...")
    return self._net(x).squeeze(-1)
```

### Наша версия:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1)
```

### Что изменено:
- Убрана проверка на `self._net` — заглушка, т.к. реальная логика в sklearn

---

## Метод `fit()` — строки 37–47

### Оригинал (GitHub):
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
    X_scaled = self._scaler.fit_transform(X)

    self._build_network(X_scaled.shape[1])

    X_t = torch.from_numpy(X_scaled).float()
    y_t = torch.from_numpy(y.astype(np.float32))

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    self.train()
    for _ in range(200):
        optimizer.zero_grad()
        logits = self(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()

    self.eval()
    return self
```

### Наша версия:
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
    X_scaled = self._scaler.fit_transform(X)
    self._clf = LogisticRegression(
        C=0.05,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",
        penalty="l2",
    )
    self._clf.fit(X_scaled, y)
    return self
```

### Что изменено:
- Весь PyTorch training loop (Adam, 200 epochs, BCEWithLogitsLoss) заменён на одну строку `self._clf.fit()`
- `C=0.05` — сильная L2-регуляризация (обратная величина: меньше = сильнее)
- `class_weight="balanced"` — автоматическая балансировка классов (заменяет ручной pos_weight)
- `max_iter=2000` — увеличен лимит итераций для сходимости lbfgs
- `solver="lbfgs"` — оптимален для малых dense-матриц

---

## Метод `fit_hyperparameters()` — строки 49–63

### Без изменений
Код порогового поиска идентичен оригиналу.

---

## Метод `predict()` — строки 65–67

### Оригинал (GitHub):
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)
```

### Наша версия:
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    probs = self.predict_proba(X)[:, 1]
    return (probs >= self._threshold).astype(int)
```

### Что изменено:
- Добавлена промежуточная переменная `probs` — косметическое изменение, логика идентична

---

## Метод `predict_proba()` — строки 69–71

### Оригинал (GitHub):
```python
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    X_scaled = self._scaler.transform(X)
    X_t = torch.from_numpy(X_scaled).float()
    with torch.no_grad():
        logits = self(X_t)
        prob_pos = torch.sigmoid(logits).numpy()
    return np.stack([1.0 - prob_pos, prob_pos], axis=1)
```

### Наша версия:
```python
def predict_proba(self, X: np.ndarray) -> np.ndarray:
    X_scaled = self._scaler.transform(X)
    return self._clf.predict_proba(X_scaled)
```

### Что изменено:
- Убран PyTorch inference (torch.from_numpy, no_grad, sigmoid) — заменён на sklearn `predict_proba()`
- sklearn автоматически возвращает матрицу (n, 2) с вероятностями классов
