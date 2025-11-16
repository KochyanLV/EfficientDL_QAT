# SASRec with Quantization-Aware Training (QAT)

Реализация модели **SASRec** (Self-Attentive Sequential Recommendation) с различными методами **квантизации (QAT)** для INT8.

## Структура проекта

```
sasrec_new/
├── src/
│   ├── model/
│   │   ├── sasrec.py              # Базовая модель SASRec с хуками для квантизации
│   │   ├── sasrec_lsq.py          # SASRec + LSQ
│   │   ├── sasrec_pact.py         # SASRec + PACT
│   │   ├── sasrec_adaround.py     # SASRec + AdaRound
│   │   ├── sasrec_apot.py         # SASRec + APoT
│   │   ├── sasrec_dorefa.py       # SASRec + DoReFa
│   │   ├── sasrec_ste.py          # SASRec + Fake STE
│   │   ├── embedding_layer.py     # Слой эмбеддингов
│   │   ├── self_attn_block.py     # Блок self-attention
│   │   └── ...
│   ├── dataset.py                 # Загрузка и подготовка данных
│   ├── trainer.py                 # Цикл обучения и оценки
│   ├── main_simple.py             # Упрощённый скрипт запуска (без mlflow)
│   └── utils/
│       ├── arguments.py           # Аргументы командной строки
│       └── ...
├── data/
│   └── movie-lens_1m.txt          # MovieLens-1M датасет
├── training.py                    # Скрипт для обучения всех моделей
├── experiments.ipynb              # Jupyter ноутбук с экспериментами
└── README.md                      # Этот файл
```

## Методы квантизации

Все методы используют **fake quantization** (симуляция INT8) во время обучения:

1. **Base SASRec**: Без квантизации (FP32 baseline)
2. **LSQ** (Learned Step Size Quantization): Обучаемые шаги квантизации
3. **PACT** (Parameterized Clipping Activation): Обучаемые пороги клиппинга
4. **AdaRound** (Adaptive Rounding): Адаптивное округление весов
5. **APoT** (Additive Powers-of-Two): Неравномерная квантизация степенями двойки
6. **DoReFa**: Квантизация активаций и весов с нормализацией
7. **Fake STE**: Простой Straight-Through Estimator

## Установка

```bash
# Установите зависимости
pip install -r ../requirements.txt
```

## Использование

### 1. Обучение всех моделей (Python скрипт)

```bash
cd /path/to/EfficientDL_QAT/sasrec_new
python training.py
```

Этот скрипт обучит все 7 моделей (Base + 6 квантизованных) и сохранит результаты в папку `outputs/`.

### 2. Обучение через Jupyter Notebook

```bash
cd /path/to/EfficientDL_QAT/sasrec_new
jupyter notebook experiments.ipynb
```

Ноутбук содержит:
- Пошаговое обучение всех моделей
- Визуализацию результатов
- Сравнительную таблицу метрик

### 3. Обучение отдельной модели

```python
from src.dataset import Dataset
from src.model import QuantSASRecLSQ
from src.trainer import Trainer
import torch

# Загрузка данных
dataset = Dataset(
    batch_size=128,
    max_seq_len=50,
    data_filepath='data/movie-lens_1m.txt',
    debug=False,
)

# Создание модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QuantSASRecLSQ(
    num_items=dataset.num_items,
    num_blocks=2,
    hidden_dim=50,
    max_seq_len=50,
    dropout_p=0.5,
    share_item_emb=False,
    device=str(device),
    bits=8,
)

# Обучение
# ... (см. training.py для полного примера)
```

## Конфигурация

Все гиперпараметры вынесены в глобальный файл `config.py`:

```python
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset parameters
BATCH_SIZE = 128
MAX_SEQ_LEN = 50

# Model architecture
HIDDEN_DIM = 50
NUM_BLOCKS = 2
DROPOUT = 0.5

# Training parameters
EPOCHS = 20
LR = 1e-3
EARLY_STOP_EPOCH = 10

# Evaluation
EVALUATE_K = 10

# Quantization parameters
BITS = 8
PACT_INIT_ALPHA = 6.0
# ... и другие
```

Чтобы изменить параметры, просто отредактируйте `config.py`.

## Метрики

Модели оцениваются по следующим метрикам:

- **nDCG@10**: Normalized Discounted Cumulative Gain (основная метрика)
- **Hit@10**: Hit Rate (recall в top-10)

## Датасет

Используется **MovieLens-1M** в формате:
```
user_id item_id
1 1
1 2
...
```

Данные автоматически разбиваются на train/valid/test:
- **Train**: все взаимодействия кроме последних двух
- **Valid**: предпоследний элемент последовательности
- **Test**: последний элемент последовательности

## Результаты

После обучения результаты сохраняются в:
- `outputs/<model_name>/best_checkpoint.pt` - лучшая модель
- `results_summary.csv` - сводная таблица метрик
- `results_comparison.png` - визуализация сравнения

## Отличия от оригинальной реализации

По сравнению с исходным кодом в `sasrec_new/src/`:

1. ✅ **Убран MLflow** - упрощённое логирование
2. ✅ **Добавлены хуки квантизации** в базовую модель SASRec
3. ✅ **Созданы дочерние классы** для всех методов QAT
4. ✅ **Упрощён dataset.py** - убрана условная логика для debug
5. ✅ **Создан training.py** - аналогично `lstm/training.py`
6. ✅ **Создан experiments.ipynb** - для интерактивных экспериментов
7. ✅ **Создан config.py** - глобальная конфигурация вместо `arguments.py`

## Архитектура квантизации

Квантизация применяется к:
- **Активациям после embedding layer** (per-tensor)
- **Активациям после каждого attention block** (per-tensor)
- **Активациям после final layer norm** (per-tensor)

Пример хуков в базовой модели:

```python
def quant_embed_out(self, x: torch.Tensor) -> torch.Tensor:
    """Quantization hook after embedding layer"""
    return x  # no-op в базовой модели

def quant_attn_out(self, x: torch.Tensor, block_idx: int = 0) -> torch.Tensor:
    """Quantization hook after each attention block"""
    return x

def quant_final_out(self, x: torch.Tensor) -> torch.Tensor:
    """Quantization hook after final layer norm"""
    return x
```

## Ссылки

- **SASRec**: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- **LSQ**: [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)
- **PACT**: [PACT: Parameterized Clipping Activation](https://arxiv.org/abs/1805.06085)
- **AdaRound**: [Up or Down? Adaptive Rounding](https://arxiv.org/abs/2004.10568)
- **APoT**: [Additive Powers-of-Two Quantization](https://arxiv.org/abs/1909.13144)
- **DoReFa**: [DoReFa-Net](https://arxiv.org/abs/1606.06160)

## Лицензия

См. `../LICENSE`

