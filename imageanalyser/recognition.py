"""Lazy inference using the pinned official Recognize Anything RAM++ package."""
import json
import os
from pathlib import Path
import threading

from PIL import Image, ImageOps

SOURCE_REVISION = '7cb804a8609e9f4b1a50b7f31436d2df40bb9481'
CHECKPOINT_REPO = 'xinyu1205/recognize-anything-plus-model'
CHECKPOINT_FILE = 'ram_plus_swin_large_14m.pth'
CHECKPOINT_REVISION = '84d4aee3a0265c4e0df1f714f0572011d1bf2ec3'


class RecognitionError(RuntimeError):
    pass


def recognition_key(config):
    checkpoint = config.get('recognize_checkpoint', '')
    if checkpoint:
        path = Path(checkpoint).expanduser().resolve()
        try:
            stat = path.stat()
            identity = [str(path), stat.st_size, stat.st_mtime_ns]
        except OSError:
            identity = [str(path), 'unavailable']
    else:
        identity = [CHECKPOINT_REPO, CHECKPOINT_FILE, CHECKPOINT_REVISION]
    return 'ram-plus-v1:' + json.dumps([SOURCE_REVISION, identity], separators=(',', ':'))


def validate_recognition_data(data):
    if not isinstance(data, dict) or any(not isinstance(data.get(field), list) or
            any(not isinstance(tag, str) for tag in data[field]) for field in ('tags', 'tags_chinese')):
        raise ValueError('RAM++ data must contain tags and tags_chinese string arrays')
    return data


class RecognizeAnything:
    def __init__(self):
        self.lock = threading.RLock()
        self.loaded = None
        self.model = self.transform = None

    def load(self, config):
        key = (recognition_key(config), config.get('recognize_device', 'auto'))
        if self.loaded == key:
            return
        try:
            import torch
            from ram import get_transform
            from ram.models import ram_plus
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RecognitionError('Install RAM++ with uv sync --extra openclip --extra recognize, then restart') from exc
        except Exception as exc:
            raise RecognitionError(f'Could not initialize RAM++ dependencies: {exc}') from exc
        device = key[1]
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise RecognitionError('RAM++ GPU inference needs compatible CUDA/ROCm PyTorch; select CPU or auto')
        if device == 'cpu':
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        try:
            checkpoint = config.get('recognize_checkpoint') or hf_hub_download(
                CHECKPOINT_REPO, CHECKPOINT_FILE, revision=CHECKPOINT_REVISION)
            model = ram_plus(pretrained=str(checkpoint), image_size=384, vit='swin_l')
            self.model = model.eval().to(device)
            self.transform = get_transform(image_size=384)
        except Exception as exc:
            raise RecognitionError(f'Could not load RAM++: {exc}') from exc
        self.device, self.loaded = device, key

    def image(self, path, config):
        with self.lock:
            self.load(config)
            import torch
            from ram import inference_ram
            try:
                with Image.open(path) as original:
                    image = self.transform(ImageOps.exif_transpose(original).convert('RGB')).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    english, chinese = inference_ram(image, self.model)
            except RuntimeError as exc:
                raise RecognitionError(f'RAM++ inference failed: {exc}') from exc
            return {'tags': [tag.strip().lower() for tag in english.split('|') if tag.strip()],
                    'tags_chinese': [tag.strip() for tag in chinese.split('|') if tag.strip()]}
