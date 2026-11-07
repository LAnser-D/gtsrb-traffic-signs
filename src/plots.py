"""Графики для отчёта. Каждый график подписан, оси подписаны, файл сохраняется
в results/figures/ в разрешении 200 dpi, чтобы вставить его в отчёт."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from . import config

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, filename):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, bbox_inches="tight")
    plt.show()


def to_displayable(image_uint8_chw):
    return image_uint8_chw.permute(1, 2, 0).numpy()


# ----------------------------- Разведочный анализ ---------------------------
def plot_class_distribution(train_counts, test_counts, filename="01_class_distribution.png"):
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(config.NUM_CLASSES)
    ax.bar(x - 0.2, train_counts, width=0.4, label="Обучающая часть")
    ax.bar(x + 0.2, test_counts, width=0.4, label="Тестовая часть")
    ax.set_xticks(x)
    ax.set_xlabel("Номер класса")
    ax.set_ylabel("Количество изображений")
    ax.set_title("Распределение изображений по классам дорожных знаков (GTSRB)")
    ax.legend()
    _save(fig, filename)


def plot_class_examples(images_uint8, labels, filename="02_class_examples.png"):
    """По одному случайному примеру каждого из 43 классов."""
    rng = np.random.default_rng(config.SEED)
    labels = np.asarray(labels)
    fig, axes = plt.subplots(5, 9, figsize=(16, 10))
    for class_id, ax in enumerate(axes.flat):
        ax.axis("off")
        if class_id >= config.NUM_CLASSES:
            continue
        index = rng.choice(np.flatnonzero(labels == class_id))
        ax.imshow(to_displayable(images_uint8[index]))
        ax.set_title(f"{class_id}: {config.CLASS_NAMES_RU[class_id]}", fontsize=7)
    fig.suptitle("Примеры изображений каждого класса (после приведения к 32x32)")
    _save(fig, filename)


def plot_size_and_brightness(original_sizes, mean_brightness, filename="03_sizes_brightness.png"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(original_sizes[:, 0], bins=50, color="tab:blue")
    axes[0].set_title("Ширина исходных изображений")
    axes[0].set_xlabel("Ширина, пикселей")
    axes[0].set_ylabel("Количество изображений")
    axes[1].scatter(original_sizes[:, 0], original_sizes[:, 1], s=2, alpha=0.2)
    axes[1].set_title("Ширина и высота исходных изображений")
    axes[1].set_xlabel("Ширина, пикселей")
    axes[1].set_ylabel("Высота, пикселей")
    axes[2].hist(mean_brightness, bins=50, color="tab:orange")
    axes[2].set_title("Средняя яркость изображений")
    axes[2].set_xlabel("Средняя яркость (0 — чёрный, 255 — белый)")
    axes[2].set_ylabel("Количество изображений")
    _save(fig, filename)


def plot_track_example(images_uint8, indices, title, filename="04_track_example.png"):
    """Кадры одного трека — один и тот же физический знак с разного расстояния."""
    indices = list(indices)[:15]
    fig, axes = plt.subplots(1, len(indices), figsize=(1.2 * len(indices), 1.8))
    for ax, index in zip(np.atleast_1d(axes), indices):
        ax.imshow(to_displayable(images_uint8[index]))
        ax.axis("off")
    fig.suptitle(title)
    _save(fig, filename)


def plot_before_after_normalization(raw_uint8, normalized, filename="05_before_after_normalization.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["red", "green", "blue"]
    names = ["R", "G", "B"]
    raw = raw_uint8.float().div(255.0)
    for c in range(3):
        axes[0].hist(raw[:, c].flatten().numpy(), bins=60, alpha=0.5, color=colors[c], label=names[c])
        axes[1].hist(normalized[:, c].flatten().numpy(), bins=60, alpha=0.5, color=colors[c], label=names[c])
    axes[0].set_title("До стандартизации (значения пикселей 0..1)")
    axes[1].set_title("После стандартизации (среднее 0, ст. отклонение 1)")
    for ax in axes:
        ax.set_xlabel("Значение пикселя")
        ax.set_ylabel("Количество пикселей")
        ax.legend(title="Канал")
    _save(fig, filename)


def plot_augmentation_examples(images_uint8, augmentation, indices, filename="06_augmentation.png"):
    fig, axes = plt.subplots(len(indices), 6, figsize=(9, 1.6 * len(indices)), squeeze=False)
    for row, index in enumerate(indices):
        axes[row, 0].imshow(to_displayable(images_uint8[index]))
        axes[row, 0].set_title("Исходное" if row == 0 else "", fontsize=8)
        for col in range(1, 6):
            axes[row, col].imshow(to_displayable(augmentation(images_uint8[index])))
            if row == 0:
                axes[row, col].set_title(f"Аугментация {col}", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Примеры аугментации: поворот, сдвиг, масштаб, яркость и контраст")
    _save(fig, filename)


# ----------------------------- Обучение -------------------------------------
def plot_training_history(history, title, filename):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(epochs, history["train_loss"], "o-", label="Обучающая выборка")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Валидационная выборка")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Функция потерь (кросс-энтропия)")
    axes[0].set_title("Функция потерь")
    axes[1].plot(epochs, history["train_accuracy"], "o-", label="Обучающая выборка")
    axes[1].plot(epochs, history["val_accuracy"], "s-", label="Валидационная выборка")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Доля верных ответов (accuracy)")
    axes[1].set_title("Точность")
    for ax in axes:
        ax.legend()
    fig.suptitle(title)
    _save(fig, filename)


def plot_experiment_comparison(summaries: dict, title, filename, metric="val_accuracy"):
    """Сравнение нескольких экспериментов по кривой валидационной точности/потерь."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, summary in summaries.items():
        values = summary["history"][metric]
        ax.plot(np.arange(1, len(values) + 1), values, "o-", markersize=3, label=label)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Точность на валидации" if metric == "val_accuracy" else "Потери на валидации")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)


def plot_depth_cost(depth_table, filename="10_depth_vs_time.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(depth_table["Свёрточных блоков"].astype(str), depth_table["Параметры"])
    axes[0].set_xlabel("Число свёрточных блоков")
    axes[0].set_ylabel("Число обучаемых параметров")
    axes[0].set_title("Размер модели")
    axes[1].plot(depth_table["Операции MAC, млн"], depth_table["Время эпохи, c"], "o-")
    for _, row in depth_table.iterrows():
        axes[1].annotate(f"{int(row['Свёрточных блоков'])} бл.",
                         (row["Операции MAC, млн"], row["Время эпохи, c"]))
    axes[1].set_xlabel("Операций умножения-сложения на изображение, млн")
    axes[1].set_ylabel("Время одной эпохи обучения, с")
    axes[1].set_title("Время обучения в зависимости от сложности сети")
    _save(fig, filename)


# ----------------------------- Итоговая оценка ------------------------------
def plot_confusion(y_true, y_pred, filename="20_confusion_matrix.png"):
    matrix = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES), normalize="true")
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Нормированная матрица ошибок на тестовой выборке (доли по строкам)")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Доля изображений истинного класса")
    _save(fig, filename)
    return matrix


def plot_per_class_f1(f1_scores, filename="21_per_class_f1.png"):
    order = np.argsort(f1_scores)
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.barh([f"{i}: {config.CLASS_NAMES_RU[i]}" for i in order], f1_scores[order])
    ax.set_xlabel("F1-мера")
    ax.set_title("F1-мера по классам на тестовой выборке (по возрастанию)")
    ax.set_xlim(min(0.8, f1_scores.min() - 0.02), 1.0)
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, filename)


def plot_roc(y_true, probs, worst_classes, filename="22_roc_curves.png"):
    """ROC-кривые «один против всех»: усреднённая micro-кривая и кривые
    для классов с наименьшей F1-мерой."""
    y_bin = label_binarize(y_true, classes=range(config.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
    ax.plot(fpr, tpr, lw=2.5, label=f"Micro-усреднение (AUC = {auc(fpr, tpr):.4f})")
    for class_id in worst_classes:
        fpr, tpr, _ = roc_curve(y_bin[:, class_id], probs[:, class_id])
        ax.plot(fpr, tpr, lw=1.2,
                label=f"Класс {class_id} «{config.CLASS_NAMES_RU[class_id]}» (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Случайный классификатор")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_xlabel("Доля ложноположительных (FPR), логарифмическая шкала")
    ax.set_ylabel("Доля истинноположительных (TPR)")
    ax.set_title("ROC-кривые на тестовой выборке")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, filename)


def plot_misclassified(images_uint8, y_true, y_pred, probs, n=24, filename="23_misclassified.png"):
    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        print("Ошибок нет")
        return
    # самые «уверенные» ошибки — самые показательные
    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n]
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.6 * rows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, index in zip(np.atleast_1d(axes).flat, wrong):
        ax.imshow(to_displayable(images_uint8[index]))
        ax.set_title(f"Истина: {config.CLASS_NAMES_RU[y_true[index]]}\n"
                     f"Модель: {config.CLASS_NAMES_RU[y_pred[index]]} ({probs[index, y_pred[index]]:.0%})",
                     fontsize=7)
    fig.suptitle("Примеры ошибок модели на тестовой выборке (наиболее уверенные ошибки)")
    _save(fig, filename)
