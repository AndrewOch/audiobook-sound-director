"""
Модуль для настройки путей кэша.

Если проект находится на внешнем диске (например, /Volumes/ADATA/...),
кэш будет перенаправлён в корень этого диска.

Если внешний диск определить не удалось, кэш будет размещён
в каталоге .cache внутри проекта или в пути, указанном
в переменной окружения AUDIOBOOK_CACHE_ROOT.
"""

import os
from pathlib import Path


ENV_CACHE_ROOT = "AUDIOBOOK_CACHE_ROOT"


def get_project_root(project_root: Path | None = None) -> Path:
    """
    Возвращает корень проекта. Если не передан явно — определяется
    относительно текущего файла (на один уровень вверх).
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return project_root


def detect_external_disk_root(path: Path) -> Path | None:
    """
    Пытается определить корень внешнего диска по пути проекта.
    Если диск похож на внешний (Volumes, mnt, media), возвращает его корень,
    иначе — None.
    """
    parts = path.resolve().parts

    # macOS: /Volumes/DISK_NAME/...
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "Volumes":
        return Path("/") / parts[1] / parts[2]

    # Linux: /mnt/DISK_NAME/...
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt":
        return Path("/") / parts[1] / parts[2]

    # Linux: /media/USER/DISK_NAME/...
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "media":
        return Path("/") / parts[1] / parts[2] / parts[3]

    # Не похоже на внешний диск
    return None


def get_cache_root(project_root: Path | None = None) -> Path:
    """
    Возвращает корневую директорию для кэша.

    Приоритет:
    1. Переменная окружения AUDIOBOOK_CACHE_ROOT (если указана).
    2. Корень внешнего диска (если проект на нём).
    3. Каталог .cache внутри проекта.
    """
    project_root = get_project_root(project_root)

    # 1. Явно переопределённый путь через env
    env_root = os.environ.get(ENV_CACHE_ROOT)
    if env_root:
        return Path(env_root).expanduser().resolve()

    # 2. Попытка определить внешний диск
    external_root = detect_external_disk_root(project_root)
    if external_root is not None:
        return external_root / ".cache" / "audiobook-sound-director"

    # 3. Fallback: локальный .cache в проекте
    return project_root / ".cache"


def setup_cache_directories(project_root: Path | None = None) -> Path:
    """
    Настраивает переменные окружения для перенаправления кэша.
    Возвращает путь к корневой директории кэша.
    """
    project_root = get_project_root(project_root)
    cache_root = get_cache_root(project_root)

    # Создаём основную директорию кэша
    cache_root.mkdir(parents=True, exist_ok=True)

    # HuggingFace кэш (для transformers, diffusers, MusicGen, AudioLDM2)
    hf_cache = cache_root / "huggingface"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")

    # Whisper кэш
    whisper_cache = cache_root / "whisper"
    whisper_cache.mkdir(parents=True, exist_ok=True)
    os.environ["WHISPER_CACHE_DIR"] = str(whisper_cache)

    # PyTorch кэш (для torch.hub)
    torch_cache = cache_root / "torch"
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)

    # Общий кэш (XDG_CACHE_HOME для Unix-систем)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)

    # Временные файлы Python (tempfile)
    temp_dir = cache_root / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(temp_dir))

    return cache_root


def get_whisper_download_root(project_root: Path | None = None) -> Path:
    """
    Возвращает путь для загрузки моделей Whisper.
    Гарантированно лежит в том же корне кэша, что и остальной кэш.
    """
    project_root = get_project_root(project_root)
    cache_root = get_cache_root(project_root)
    whisper_cache = cache_root / "whisper"
    whisper_cache.mkdir(parents=True, exist_ok=True)
    return whisper_cache
