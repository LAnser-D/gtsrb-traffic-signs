"""Архитектура свёрточной нейросети и расчёт её параметров и вычислительной сложности."""
from __future__ import annotations

import pandas as pd
import torch
from torch import nn

from . import config

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


class TrafficSignCNN(nn.Module):
    """Свёрточная сеть для классификации дорожных знаков 32x32.

    Сеть состоит из нескольких свёрточных блоков:
        Conv2d 3x3 (padding=1) -> [BatchNorm2d] -> функция активации -> MaxPool2d 2x2
    Каждый блок уменьшает пространственный размер в 2 раза (32 -> 16 -> 8 -> 4)
    и увеличивает число каналов (признаков). Затем классификатор:
        Flatten -> Linear -> активация -> [Dropout] -> Linear (43 выхода, логиты классов)
    Softmax явно не применяется: он входит в функцию потерь CrossEntropyLoss.
    """

    def __init__(self, channels=(32, 64, 128), activation: str = "relu",
                 batch_norm: bool = False, dropout: float = 0.0,
                 hidden_units: int = 256, num_classes: int = config.NUM_CLASSES,
                 image_size: int = config.IMAGE_SIZE):
        super().__init__()
        activation_cls = ACTIVATIONS[activation]
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_cls())
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        feature_map_size = image_size // (2 ** len(channels))
        flattened_size = in_channels * feature_map_size * feature_map_size
        classifier = [nn.Flatten(), nn.Linear(flattened_size, hidden_units), activation_cls()]
        if dropout > 0:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(hidden_units, num_classes))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_summary(model: nn.Module, image_size: int = config.IMAGE_SIZE) -> pd.DataFrame:
    """Таблица по слоям: размер выхода, число параметров и число операций
    умножения-сложения (MAC) на одно изображение.

    Формулы:
      Conv2d:  params = (C_in * k * k + 1) * C_out
               MAC    = H_out * W_out * C_out * C_in * k * k
      Linear:  params = (N_in + 1) * N_out
               MAC    = N_in * N_out
      BatchNorm2d: params = 2 * C (масштаб gamma и сдвиг beta)
    """
    rows = []
    hooks = []

    def hook(module, inputs, output):
        params = sum(p.numel() for p in module.parameters(recurse=False))
        macs = 0
        if isinstance(module, nn.Conv2d):
            c_in, k = module.in_channels, module.kernel_size[0]
            _, c_out, h_out, w_out = output.shape
            macs = h_out * w_out * c_out * c_in * k * k
        elif isinstance(module, nn.Linear):
            macs = module.in_features * module.out_features
        rows.append({
            "Слой": module.__class__.__name__,
            "Размер выхода": "x".join(str(d) for d in output.shape[1:]),
            "Параметры": params,
            "Операции MAC": macs,
        })

    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=next(model.parameters()).device))
    for h in hooks:
        h.remove()
    table = pd.DataFrame(rows)
    total = {"Слой": "ИТОГО", "Размер выхода": "",
             "Параметры": table["Параметры"].sum(), "Операции MAC": table["Операции MAC"].sum()}
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)
