"""Загрузка GTSRB, разбиение на выборки, предобработка и загрузчики данных."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

from . import config


# ---------------------------------------------------------------------------
# 1. Список файлов датасета
# ---------------------------------------------------------------------------
def load_gtsrb_file_lists(data_dir: Path = config.DATA_DIR, download: bool = True):
    """Скачивает GTSRB (если ещё не скачан) и возвращает списки (путь, класс)
    для обучающей и тестовой частей."""
    train_set = datasets.GTSRB(root=str(data_dir), split="train", download=download)
    test_set = datasets.GTSRB(root=str(data_dir), split="test", download=download)
    return list(train_set._samples), list(test_set._samples)


def track_id_from_path(path: str) -> str:
    """В обучающей части GTSRB файлы называются <трек>_<кадр>.ppm.
    Один трек — это ~30 снимков ОДНОГО физического знака с разного расстояния."""
    return Path(path).name.split("_")[0]


# ---------------------------------------------------------------------------
# 2. Разбиение train / validation по трекам
# ---------------------------------------------------------------------------
def split_train_validation_by_track(samples, validation_fraction=config.VALIDATION_FRACTION,
                                    seed=config.SEED):
    """Делит обучающую часть на train и validation так, чтобы все кадры одного
    трека попали в одну выборку. Иначе почти одинаковые снимки одного знака окажутся
    и в train, и в validation, и точность на валидации будет завышена (утечка данных).
    Разбиение делается внутри каждого класса, чтобы сохранить пропорции классов."""
    rng = random.Random(seed)
    tracks_by_class: dict[int, dict[str, list[int]]] = {}
    for index, (path, label) in enumerate(samples):
        tracks_by_class.setdefault(label, {}).setdefault(track_id_from_path(path), []).append(index)

    train_indices, validation_indices = [], []
    for label in sorted(tracks_by_class):
        track_ids = sorted(tracks_by_class[label])
        rng.shuffle(track_ids)
        n_validation_tracks = max(1, round(len(track_ids) * validation_fraction))
        for i, track in enumerate(track_ids):
            target = validation_indices if i < n_validation_tracks else train_indices
            target.extend(tracks_by_class[label][track])
    return sorted(train_indices), sorted(validation_indices)


# ---------------------------------------------------------------------------
# 3. Чтение изображений в память
# ---------------------------------------------------------------------------
@dataclass
class ImageArrays:
    images: torch.Tensor         # uint8, N x 3 x H x W, уже приведены к IMAGE_SIZE
    labels: torch.Tensor         # int64, N
    original_sizes: np.ndarray   # N x 2 (ширина, высота) исходных изображений
    mean_brightness: np.ndarray  # N, средняя яркость исходного изображения (0..255)


def read_images(samples, image_size: int = config.IMAGE_SIZE) -> ImageArrays:
    """Открывает каждое изображение, запоминает исходный размер и яркость,
    затем приводит к размеру image_size x image_size (билинейная интерполяция)."""
    n = len(samples)
    images = torch.empty((n, 3, image_size, image_size), dtype=torch.uint8)
    labels = torch.empty(n, dtype=torch.int64)
    original_sizes = np.empty((n, 2), dtype=np.int32)
    mean_brightness = np.empty(n, dtype=np.float32)
    for i, (path, label) in enumerate(samples):
        with Image.open(path) as img:
            img = img.convert("RGB")
            original_sizes[i] = img.size
            mean_brightness[i] = np.asarray(img.convert("L"), dtype=np.float32).mean()
            resized = img.resize((image_size, image_size), Image.BILINEAR)
        images[i] = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)
        labels[i] = label
    return ImageArrays(images, labels, original_sizes, mean_brightness)


def load_or_cache_arrays(samples, cache_file: Path) -> ImageArrays:
    """Чтение ~50 тыс. файлов занимает пару минут, поэтому результат кэшируется."""
    if cache_file.exists():
        stored = torch.load(cache_file, weights_only=False)
        return ImageArrays(**stored)
    arrays = read_images(samples)
    torch.save(arrays.__dict__, cache_file)
    return arrays


def compute_channel_mean_std(images_uint8: torch.Tensor):
    """Среднее и стандартное отклонение по каждому каналу RGB (по обучающей выборке)."""
    pixels = images_uint8.float().div(255.0)
    mean = pixels.mean(dim=(0, 2, 3))
    std = pixels.std(dim=(0, 2, 3))
    return mean.tolist(), std.tolist()


# ---------------------------------------------------------------------------
# 4. Dataset и аугментация
# ---------------------------------------------------------------------------
def build_augmentation():
    """Аугментация имитирует реальные условия съёмки: небольшой поворот камеры,
    смещение и масштаб знака в кадре, разное освещение.
    Горизонтальное отражение НЕ используется: «поворот налево» после отражения
    превратился бы в «поворот направо» — это сменило бы класс."""
    return v2.Compose([
        v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    ])


class TrafficSignDataset(Dataset):
    def __init__(self, images_uint8, labels, mean, std, augmentation=None):
        self.images = images_uint8
        self.labels = labels
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augmentation is not None:
            image = self.augmentation(image)
        image = image.float().div(255.0)
        image = (image - self.mean) / self.std   # стандартизация каналов
        return image, self.labels[index]


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, shuffle: bool, batch_size: int = config.BATCH_SIZE,
                seed: int = config.SEED, num_workers: int = 2):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, worker_init_fn=_seed_worker,
                      generator=generator, pin_memory=torch.cuda.is_available())
