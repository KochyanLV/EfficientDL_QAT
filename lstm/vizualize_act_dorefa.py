# viz_dorefa.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ---------- вспомогательные функции (как в DoReFa) ----------
def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x - x.detach()) + x.round().detach()

def quantize_kbits_unit(x01: torch.Tensor, k: int) -> torch.Tensor:
    """Квантование x∈[0,1] в k бит (равномерная сетка из 2^k уровней)."""
    n = (1 << k) - 1  # 2^k - 1
    y = ste_round(x01 * n) / n
    return y.clamp(0.0, 1.0)

# ---------- разложение активаций DoReFaActQuant ----------
@torch.no_grad()
def dorefa_act_decompose(x: torch.Tensor, module) -> dict:
    """
    Восстанавливает внутренние шаги DoReFaActQuant для тензора x.
    Возвращает:
      bits, signed, preproc,
      x_raw,
      x_unit  — «нормализованная» величина в [0,1] (unsigned) ИЛИ
                xsigned в [-1,1] (signed) — используется как baseline (без округления),
      x_ref_for_err — та же baseline-величина, приведённая к диапазону x_deq для подсчёта ошибки,
      codes, x_deq — де-квантованное значение (в том же домене, что baseline_ref).
    """
    bits = int(module.bits_a)
    signed = bool(module.signed)
    preproc = module.preproc
    eps = float(getattr(module, "eps", torch.tensor(1e-8)).item())

    x = x.detach().float()
    out = {"bits": bits, "signed": signed, "preproc": preproc}

    if not signed:
        # unsigned [0,1]
        if preproc == "clip":
            x01 = x.clamp(0., 1.)
            x_ref_for_err = x01
        else:  # 'tanh'
            xt = torch.tanh(x)
            amax = xt.abs().max().clamp(min=eps)
            x01 = xt / (2 * amax) + 0.5
            out["x_tanh"] = xt.cpu()
            x_ref_for_err = x01

        codes = torch.round(x01 * ((1 << bits) - 1))
        deq01 = quantize_kbits_unit(x01, bits)  # [0,1]

        out.update({
            "x_raw": x.cpu(),
            "x_unit": x01.cpu(),
            "codes": codes.cpu(),
            "x_deq": deq01.cpu(),               # [0,1]
            "x_ref_for_err": x_ref_for_err.cpu()
        })
    else:
        # signed [-1,1] (через tanh-нормализацию)
        xt = torch.tanh(x)
        amax = xt.abs().max().clamp(min=eps)
        xsigned = xt / amax                      # [-1,1]
        x01 = (xsigned + 1.0) * 0.5              # [0,1]

        codes = torch.round(x01 * ((1 << bits) - 1))
        deq01 = quantize_kbits_unit(x01, bits)   # [0,1]
        deq_signed = 2.0 * deq01 - 1.0           # [-1,1]

        out.update({
            "x_raw": x.cpu(),
            "x_tanh": xt.cpu(),
            "x_signed": xsigned.cpu(),           # baseline в [-1,1]
            "x_unit": x01.cpu(),
            "codes": codes.cpu(),
            "x_deq": deq_signed.cpu(),           # deq в [-1,1]
            "x_ref_for_err": xsigned.cpu(),      # сравниваем xsigned vs x_deq
        })

    out["qmin"] = 0
    out["qmax"] = (1 << bits) - 1
    return out

# ---------- разложение весов DoReFaWeightQuant ----------
@torch.no_grad()
def dorefa_weight_decompose(w: torch.Tensor, module) -> dict:
    """
    Разворачивает шаги DoReFaWeightQuant:
      w -> tanh -> unit[0,1] -> codes -> deq[-1,1]
    Возвращает также baseline веса без округления (w_ref = 2*unit-1).
    """
    bits = int(module.bits_w)
    eps = float(getattr(module, "eps", torch.tensor(1e-8)).item())

    w = w.detach().float()
    w_t = torch.tanh(w)
    amax = w_t.abs().max().clamp(min=eps)
    w_unit = w_t / (2 * amax) + 0.5
    codes = torch.round(w_unit * ((1 << bits) - 1))
    w_deq01 = quantize_kbits_unit(w_unit, bits)     # [0,1]
    w_deq = 2.0 * w_deq01 - 1.0                     # [-1,1]
    w_ref = 2.0 * w_unit - 1.0                      # baseline без округления

    return {
        "bits": bits,
        "qmin": 0,
        "qmax": (1 << bits) - 1,
        "w_raw": w.cpu(),
        "w_tanh": w_t.cpu(),
        "w_unit": w_unit.cpu(),
        "codes": codes.cpu(),
        "w_deq": w_deq.cpu(),
        "w_ref": w_ref.cpu(),
    }

# ---------- утилиты рисования ----------
def _hist(ax, arr, title, bins=50, color=None):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    ax.hist(arr, bins=bins, alpha=0.85, edgecolor="black", color=color)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)

def _int_code_bins(qmin, qmax):
    # Бины по целым значениям кодов
    # np.arange даёт края бинов; для корректной гистограммы добавим +1
    return np.arange(qmin, qmax + 2, 1, dtype=np.int32)

# ---------- главный визуализатор ----------
@torch.no_grad()
def visualize_dorefa_qat_lstm(
    model,
    input_ids: torch.Tensor,
    out_dir: str = "./qat_debug",
    title: str = "BERT-like QAT view for DoReFa LSTM (step XXX)"
) -> str:
    """
    Делает один forward двумя путями:
      • FP′: те же нормализации DoReFa, но БЕЗ округления (baseline)
      • Quant: с округлением (как в твоём forward)
    И строит 12-панельную фигуру.
    Требуемые атрибуты модели:
      model.emb, model.lstm, model.head (Linear),
      model.dorefa_act_in, model.dorefa_act_lstm, model.dorefa_w_head
    """
    os.makedirs(out_dir, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()
    input_ids = input_ids.to(device)

    # 1) Эмбеддинги
    x_emb = model.emb(input_ids)                 # (B,T,E)

    # FP путь до LSTM (для инфо): out_fp не используется в сравнении
    out_fp, _ = model.lstm(x_emb)                # (B,T,H)
    pooled_fp = out_fp.mean(dim=1)               # (B,H)

    # 2) Активации после Embedding: разложение + квант-путь
    act_in = dorefa_act_decompose(x_emb, model.dorefa_act_in)
    xq_emb = model.dorefa_act_in(x_emb)          # квантованная активация

    # 3) Пропускаем LSTM на квантованных эмбеддингах
    out_q, _ = model.lstm(xq_emb)                # (B,T,H)

    # 4) Активации после LSTM: разложение + квант-путь
    act_lstm = dorefa_act_decompose(out_q, model.dorefa_act_lstm)
    out_q2 = model.dorefa_act_lstm(out_q)
    pooled_q = out_q2.mean(dim=1)                # (B,H) квантованный путь

    # 5) Веса головы
    W = model.head.weight
    B = model.head.bias
    winfo = dorefa_weight_decompose(W, model.dorefa_w_head)
    Wq   = model.dorefa_w_head(W)                # квантованные веса
    Wref = winfo["w_ref"].to(W.dtype).to(W.device)  # baseline веса без округления

    # 6) Собираем baseline FP′ (те же нормализации, без округления)
    #    act_lstm["x_ref_for_err"] уже в нужном домене:
    #      signed=True -> [-1,1], signed=False -> [0,1]
    #    нам нужна агрегация по T, затем Linear с Wref
    x_ref = act_lstm["x_ref_for_err"].to(pooled_q.dtype).to(pooled_q.device)
    # act_lstm["x_ref_for_err"] соответствует тензору размерности как out_q:
    pooled_fp_prime = x_ref.mean(dim=1)          # (B,H) baseline активации
    y_fp_prime = F.linear(pooled_fp_prime, Wref, B)
    y_quant    = F.linear(pooled_q,        Wq,   B)

    # 7) Подготовка чисел для панелей
    # Верхний ряд — активации (emb)
    x_raw = act_in["x_raw"].numpy().ravel()
    x_deq = act_in["x_deq"].numpy().ravel()
    codes_act = act_in["codes"].numpy().ravel().astype(np.int32)
    qmin_a, qmax_a = int(act_in["qmin"]), int(act_in["qmax"])
    clipped_a_min = (codes_act == qmin_a).sum()
    clipped_a_max = (codes_act == qmax_a).sum()
    clipped_a_pct_min = 100.0 * clipped_a_min / codes_act.size
    clipped_a_pct_max = 100.0 * clipped_a_max / codes_act.size
    ref_a = act_in["x_ref_for_err"].numpy().ravel()
    err_a = ref_a - x_deq

    # Средний ряд — веса
    w_raw = winfo["w_raw"].numpy().ravel()
    w_deq = winfo["w_deq"].numpy().ravel()
    codes_w = winfo["codes"].numpy().ravel().astype(np.int32)
    qmin_w, qmax_w = int(winfo["qmin"]), int(winfo["qmax"])
    clipped_w_min = (codes_w == qmin_w).sum()
    clipped_w_max = (codes_w == qmax_w).sum()
    clipped_w_pct_min = 100.0 * clipped_w_min / codes_w.size
    clipped_w_pct_max = 100.0 * clipped_w_max / codes_w.size
    # ошибка по unit (до и после округления)
    w_unit = winfo["w_unit"].numpy().ravel()
    deq_unit = codes_w / ((1 << winfo["bits"]) - 1)
    err_w_unit = w_unit - deq_unit

    # Нижний ряд — выходы
    yfp  = y_fp_prime.detach().cpu().numpy().ravel()
    yq   = y_quant.detach().cpu().numpy().ravel()
    eout = yfp - yq
    mse  = float(np.mean(eout ** 2))

    # 8) Рисуем 12 панелей
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(title, fontweight="bold")

    # 1. Raw Activation (emb)
    ax1 = plt.subplot(3, 4, 1)
    _hist(ax1, x_raw, f"1) Raw Activation (emb)\nmean={x_raw.mean():.4f}, std={x_raw.std():.4f}", bins=60, color="C0")

    # 2. Activation Codes (8-bit)
    ax2 = plt.subplot(3, 4, 2)
    ax2.hist(codes_act, bins=_int_code_bins(qmin_a, qmax_a), alpha=0.85, edgecolor="black", color="C1")
    ax2.set_title(f"2) Activation Codes ({act_in['bits']}-bit)\nrange:[{qmin_a},{qmax_a}] | "
                  f"clipped: {clipped_a_pct_min:.2f}%@min, {clipped_a_pct_max:.2f}%@max", fontsize=9)
    ax2.axvline(qmin_a, color="r", linestyle="--"); ax2.axvline(qmax_a, color="r", linestyle="--")
    ax2.grid(True, alpha=0.3)

    # 3. Dequantized Activation
    ax3 = plt.subplot(3, 4, 3)
    _hist(ax3, x_deq, "3) Dequantized Activation", bins=60, color="C2")

    # 4. Activation Quant Error
    ax4 = plt.subplot(3, 4, 4)
    _hist(ax4, err_a, f"4) Act Quant Error\nMAE={np.abs(err_a).mean():.4f}", bins=60, color="C3")

    # 5. Raw Weights
    ax5 = plt.subplot(3, 4, 5)
    _hist(ax5, w_raw, f"5) Raw Weights\nmean={w_raw.mean():.4f}, std={w_raw.std():.4f}", bins=60, color="C4")

    # 6. Weight Codes (8-bit)
    ax6 = plt.subplot(3, 4, 6)
    ax6.hist(codes_w, bins=_int_code_bins(qmin_w, qmax_w), alpha=0.85, edgecolor="black", color="C5")
    ax6.set_title(f"6) Weight Codes ({winfo['bits']}-bit)\nrange:[{qmin_w},{qmax_w}] | "
                  f"clipped: {clipped_w_pct_min:.2f}%@min, {clipped_w_pct_max:.2f}%@max", fontsize=9)
    ax6.axvline(qmin_w, color="r", linestyle="--"); ax6.axvline(qmax_w, color="r", linestyle="--")
    ax6.grid(True, alpha=0.3)

    # 7. Dequantized Weights
    ax7 = plt.subplot(3, 4, 7)
    _hist(ax7, w_deq, "7) Dequantized Weights", bins=60, color="C6")

    # 8. Weight Quant Error (unit)
    ax8 = plt.subplot(3, 4, 8)
    _hist(ax8, err_w_unit, f"8) Weight Quant Error (unit)\nMAE={np.abs(err_w_unit).mean():.4f}", bins=60, color="C3")

    # 9. FP′ Output (baseline без округления)
    ax9 = plt.subplot(3, 4, 9)
    _hist(ax9, yfp, f"9) FP′ Output\nmean={yfp.mean():.4f}, std={yfp.std():.4f}", bins=60, color="C0")

    # 10. Quantized Output
    ax10 = plt.subplot(3, 4, 10)
    _hist(ax10, yq, f"10) Quantized Output\nmean={yq.mean():.4f}, std={yq.std():.4f}", bins=60, color="C1")

    # 11. FP′ vs Quantized
    ax11 = plt.subplot(3, 4, 11)
    n_sc = min(3000, yfp.size)
    ax11.scatter(yfp[:n_sc], yq[:n_sc], s=4, alpha=0.5)
    lo = float(min(yfp.min(), yq.min())); hi = float(max(yfp.max(), yq.max()))
    ax11.plot([lo, hi], [lo, hi], 'r--', lw=2)
    ax11.set_title("11) FP′ vs Quantized"); ax11.set_xlabel("FP′"); ax11.set_ylabel("Quant")
    ax11.grid(True, alpha=0.3)

    # 12. Output Error
    ax12 = plt.subplot(3, 4, 12)
    _hist(ax12, eout, f"12) Output Error\nMSE={mse:.6f}", bins=60, color="C3")

    plt.tight_layout()
    save_path = os.path.join(out_dir, "dorefa_qat_lstm_12panels.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {save_path}")
    return save_path

from lstm.model.LstmDorefa import QuantLSTMDoReFa
from lstm.train_utils.loaders import make_loaders


if __name__ == "__main__":
    model_best = QuantLSTMDoReFa(
        vocab_size=5000,
        emb_dim=128,
        hidden_dim=256,
        num_classes=1,
        bits_a=8,
        bits_w=8,
        act_signed=True,
        act_preproc='tanh'
    )

    weights_best = torch.load("lstm/checkpoints/quantlstmdorefa_vocab_size5000_emb_dim128_hidden_dim256_num_classes1_bits_a8_bits_w8_act_signedTrue_act_preproctanh_epochs5_bs128_lr0p001.pt")
    model_best.load_state_dict(weights_best)
    
    train_loader, test_loader, _ = make_loaders(tokenizer_path="lstm/own_tokenizer", batch_size=8, max_len=512)

    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]

    # отрисовать 12 графиков
    visualize_dorefa_qat_lstm(
        model=model_best,
        input_ids=input_ids,
        out_dir="lstm/qat_act",
        title="QAT view for DoReFa LSTM"
    )
    
    