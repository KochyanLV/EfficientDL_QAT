import inspect
from pathlib import Path
import torch
import re

def record_init(cls):
    """
    Декоратор класса: перехватывает __init__, сохраняет все переданные
    параметры (с учётом дефолтов) в self._init_args (dict).
    """
    orig_init = cls.__init__
    sig = inspect.signature(orig_init)

    def __init__(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        orig_init(self, *args, **kwargs)
        self._init_args = {k: v for k, v in bound.arguments.items() if k != "self"}

    cls.__init__ = __init__
    return cls


def _short_class_name(obj) -> str:
    return obj.__class__.__name__.lower()


def _sanitize(v):
    if isinstance(v, (int, float, bool)):
        return str(v).replace(".", "p")
    if isinstance(v, str):
        v = v.strip().lower()
        v = re.sub(r"[^a-z0-9._-]+", "-", v)
        return v[:32]
    return type(v).__name__.lower()


def build_ckpt_name(model, extra: dict | None = None,
                    with_values: bool = True) -> str:
    """
    Собирает имя файла на основе:
      - имени класса модели
      - всех параметров __init__ (model._init_args)
      - опционально дополнительных полей (epochs, batch_size и т.п.)
    """
    base = _short_class_name(model)

    parts = []
    if hasattr(model, "_init_args"):
        for k, v in model._init_args.items():
            if not with_values:
                parts.append(k)
            else:
                parts.append(f"{k}{_sanitize(v)}")

    if extra:
        for k, v in extra.items():
            if not with_values:
                parts.append(k)
            else:
                parts.append(f"{k}{_sanitize(v)}")

    name = "_".join([base] + parts)
    name = re.sub(r"_+", "_", name)
    return name + ".pt"


def save_ckpt_from_init(model,
                        folder: str = "sasrec/checkpoints",
                        extra: dict | None = None,
                        with_values: bool = True) -> str:
    ckpt_dir = Path(folder)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fname = build_ckpt_name(model, extra=extra, with_values=with_values)
    path = ckpt_dir / fname
    torch.save(model.state_dict(), path)
    return str(path)

