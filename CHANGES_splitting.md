# Изменения в splitting.py

## Импорт — строка 21

### Оригинал (GitHub):
```python
from sklearn.model_selection import train_test_split
```

### Наша версия:
```python
from sklearn.model_selection import StratifiedKFold, train_test_split
```

### Что изменено:
- Добавлен `StratifiedKFold` для k-fold кросс-валидации

---

## Сигнатура функции `split_data()` — строки 24–30

### Оригинал (GitHub):
```python
def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
```

### Наша версия:
```python
def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
```

### Что изменено:
- Параметр `test_size: float = 0.15` заменён на `n_splits: int = 5` — вместо доли тестовой выборки указываем число фолдов
- `val_size` сохранён — доля train-части под валидацию

---

## Тело функции `split_data()` — строки 48–63

### Оригинал (GitHub) — Option A (single stratified split):
```python
    idx = np.arange(len(y))

    idx_train_val, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=y[idx_train_val],
    )
    return [(idx_train, idx_val, idx_test)]
```

### Наша версия — StratifiedKFold:
```python
    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []
    for idx_train_val, idx_test in skf.split(idx, y):
        relative_val = val_size / (1.0 - val_size)
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits
```

### Что изменено:
- `train_test_split` для выделения test заменён на `StratifiedKFold.split()` — каждый фолд становится тестовой выборкой по очереди
- `shuffle=True` — перемешивание перед разбиением (оригинальный Option B в GitHub имел shuffle=False)
- Результат — список из 5 кортежей вместо 1 (5-fold CV)
- Внутренний `train_test_split` для val сохранён, но `relative_val` считается от `val_size / (1.0 - val_size)` вместо `val_size / (1.0 - test_size)` — корректная формула для доли от train-части после выделения test фолдом
