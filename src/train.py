"""Обучение и оценка моделей, запуск экспериментов."""
from __future__ import annotations

import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn

from . import config
from .models import TrafficSignCNN, count_parameters


def set_seed(seed: int = config.SEED):
    """Фиксирует все генераторы случайных чисел для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()          # обратное распространение ошибки
        optimizer.step()         # шаг оптимизатора
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, return_outputs: bool = False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        if return_outputs:
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(labels.cpu())
    result = {"loss": total_loss / total, "accuracy": correct / total}
    if return_outputs:
        result["probs"] = torch.cat(all_probs).numpy()
        result["labels"] = torch.cat(all_labels).numpy()
    return result


def fit(model, train_loader, validation_loader, epochs=config.EPOCHS, lr=config.LEARNING_RATE,
        weight_decay=0.0, scheduler: str | None = None, device=None, verbose=True):
    """Обучает модель оптимизатором Adam. Возвращает историю обучения.
    В конце в модель загружаются веса эпохи с лучшей точностью на валидации."""
    device = device or get_device()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                    if scheduler == "cosine" else None)

    history = {k: [] for k in ["train_loss", "train_accuracy", "val_loss", "val_accuracy",
                               "epoch_time_sec", "lr"]}
    best_accuracy, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start
        val = evaluate(model, validation_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["epoch_time_sec"].append(epoch_time)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if lr_scheduler:
            lr_scheduler.step()
        if val["accuracy"] > best_accuracy:
            best_accuracy, best_state = val["accuracy"], copy.deepcopy(model.state_dict())
        if verbose:
            print(f"Эпоха {epoch:2d}/{epochs}: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} | {epoch_time:.1f} c")
    model.load_state_dict(best_state)
    history["best_val_accuracy"] = best_accuracy
    history["best_epoch"] = int(np.argmax(history["val_accuracy"]) + 1)
    return history


def run_experiment(name: str, train_loader, validation_loader, model_kwargs: dict,
                   epochs=config.EPOCHS, lr=config.LEARNING_RATE, weight_decay=0.0,
                   scheduler=None, seed=config.SEED, verbose=False):
    """Один эксперимент: фиксируем seed, создаём модель, обучаем, сохраняем историю в JSON."""
    set_seed(seed)
    model = TrafficSignCNN(**model_kwargs)
    print(f"\n=== {name} | параметров: {count_parameters(model):,} ===")
    history = fit(model, train_loader, validation_loader, epochs=epochs, lr=lr,
                  weight_decay=weight_decay, scheduler=scheduler, verbose=verbose)
    summary = {
        "name": name,
        "model_kwargs": model_kwargs,
        "augmentation": getattr(train_loader.dataset, "augmentation", None) is not None,
        "epochs": epochs, "lr": lr, "weight_decay": weight_decay, "scheduler": scheduler,
        "parameters": count_parameters(model),
        "best_val_accuracy": history["best_val_accuracy"],
        "best_epoch": history["best_epoch"],
        "final_train_accuracy": history["train_accuracy"][-1],
        "mean_epoch_time_sec": float(np.mean(history["epoch_time_sec"])),
        "history": history,
    }
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.EXPERIMENTS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Лучшая точность на валидации: {summary['best_val_accuracy']:.4f} "
          f"(эпоха {summary['best_epoch']}), время эпохи: {summary['mean_epoch_time_sec']:.1f} c")
    return model, summary
