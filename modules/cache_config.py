"""
Модуль для настройки путей кэша на внешний диск.

Этот модуль перенаправляет все кэши моделей (HuggingFace, Whisper, PyTorch)
в корень внешнего диска, чтобы избежать заполнения системного диска
и сделать кэш доступным для всех проектов на этом диске.
"""

import os
from pathlib import Path


def get_external_disk_root(project_root: Path = None) -> Path:
    """
    Определяет корень внешнего диска на основе пути проекта.
    
    Args:
        project_root: Корневая директория проекта. Если None, определяется автоматически.
        
    Returns:
        Path к корню внешнего диска (например, /Volumes/ADATA)
    """
    if project_root is None:
        # Определяем корень проекта на основе расположения этого файла
        project_root = Path(__file__).resolve().parent.parent
    
    # Получаем абсолютный путь
    abs_path = project_root.resolve()
    parts = abs_path.parts
    
    # Ищем корень внешнего диска
    # Для macOS: /Volumes/DISK_NAME
    # Для Linux: /media/USER/DISK_NAME или /mnt/DISK_NAME
    # Для Windows: D:\ или другой диск
    
    if len(parts) >= 3 and parts[0] == '/' and parts[1] == 'Volumes':
        # macOS: /Volumes/DISK_NAME/...
        return Path('/') / parts[1] / parts[2]
    elif len(parts) >= 2 and parts[0] == '/' and parts[1] == 'mnt':
        # Linux: /mnt/DISK_NAME/...
        if len(parts) >= 3:
            return Path('/') / parts[1] / parts[2]
    elif len(parts) >= 4 and parts[0] == '/' and parts[1] == 'media':
        # Linux: /media/USER/DISK_NAME/...
        return Path('/') / parts[1] / parts[2] / parts[3]
    
    # Если не удалось определить, используем первый уровень после корня
    # или возвращаем корень проекта как fallback
    if len(parts) >= 2:
        return Path('/') / parts[1]
    
    # Fallback: используем корень проекта
    return project_root


def setup_cache_directories(project_root: Path = None):
    """
    Настраивает переменные окружения для перенаправления кэша в корень внешнего диска.
    
    Args:
        project_root: Корневая директория проекта. Если None, определяется автоматически.
    """
    if project_root is None:
        # Определяем корень проекта на основе расположения этого файла
        project_root = Path(__file__).resolve().parent.parent
    
    # Определяем корень внешнего диска
    disk_root = get_external_disk_root(project_root)
    
    # Создаем директорию для кэша в корне внешнего диска
    cache_dir = disk_root / ".cache"
    cache_dir.mkdir(exist_ok=True)
    
    # HuggingFace кэш (для transformers, diffusers, MusicGen, AudioLDM2)
    hf_cache = cache_dir / "huggingface"
    hf_cache.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    
    # Whisper кэш
    whisper_cache = cache_dir / "whisper"
    whisper_cache.mkdir(exist_ok=True)
    os.environ["WHISPER_CACHE_DIR"] = str(whisper_cache)
    
    # PyTorch кэш (для torch.hub)
    torch_cache = cache_dir / "torch"
    torch_cache.mkdir(exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)
    
    # Общий кэш (XDG_CACHE_HOME для Unix-систем)
    # Это влияет на многие библиотеки, которые используют стандартные пути кэша
    os.environ["XDG_CACHE_HOME"] = str(cache_dir)
    
    # Временные файлы Python (tempfile)
    # Перенаправляем на внешний диск, если возможно
    temp_dir = cache_dir / "tmp"
    temp_dir.mkdir(exist_ok=True)
    # Примечание: TMPDIR/TMP влияет на tempfile, но не все библиотеки его используют
    
    return cache_dir


def get_whisper_download_root(project_root: Path = None) -> Path:
    """
    Возвращает путь для загрузки моделей Whisper.
    
    Args:
        project_root: Корневая директория проекта. Если None, определяется автоматически.
        
    Returns:
        Path к директории для моделей Whisper в корне внешнего диска
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    
    # Используем корень внешнего диска
    disk_root = get_external_disk_root(project_root)
    whisper_cache = disk_root / ".cache" / "whisper"
    whisper_cache.mkdir(parents=True, exist_ok=True)
    return whisper_cache

