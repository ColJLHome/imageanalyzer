"""Lazy OpenCLIP inference; the base application does not require PyTorch."""
import json
import os
import threading

from PIL import Image, ImageOps


class OpenClipError(RuntimeError):
    pass


def model_key(config):
    return 'openclip-v1:' + json.dumps([config.get('openclip_model', 'ViT-B-32'),
                                      config.get('openclip_pretrained', 'laion2b_s34b_b79k')],
                                     separators=(',', ':'))


class OpenClipEncoder:
    def __init__(self):
        self.lock = threading.RLock()
        self.loaded = None
        self.model = self.preprocess = self.tokenizer = None
        self.text_cache = {}

    def load(self, config):
        key = (model_key(config), config.get('openclip_device', 'auto'))
        if self.loaded == key:
            return
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise OpenClipError('Install OpenCLIP support with uv sync --extra openclip, then restart the backend') from exc
        device = key[1]
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda' and not torch.cuda.is_available():
            raise OpenClipError('GPU inference requires a compatible CUDA/ROCm PyTorch installation; select CPU or auto')
        if device == 'cpu':
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                config.get('openclip_model', 'ViT-B-32'),
                pretrained=config.get('openclip_pretrained', 'laion2b_s34b_b79k'), device=device)
            model.eval()
            tokenizer = open_clip.get_tokenizer(config.get('openclip_model', 'ViT-B-32'))
        except Exception as exc:
            raise OpenClipError(f'Could not load OpenCLIP model: {exc}') from exc
        self.model, self.preprocess, self.tokenizer = model, preprocess, tokenizer
        self.device, self.loaded = device, key
        self.text_cache.clear()

    @staticmethod
    def vector(features):
        import numpy as np
        vector = features.detach().float().cpu().numpy().reshape(-1).astype('<f4')
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise OpenClipError('OpenCLIP returned an invalid embedding')
        return (vector / norm).astype('<f4').tobytes()

    def image(self, path, config):
        with self.lock:
            self.load(config)
            import torch
            with Image.open(path) as original:
                image = self.preprocess(ImageOps.exif_transpose(original).convert('RGB')).unsqueeze(0).to(self.device)
            try:
                with torch.inference_mode():
                    return self.vector(self.model.encode_image(image))
            except RuntimeError as exc:
                raise OpenClipError(f'OpenCLIP image inference failed: {exc}') from exc

    def text(self, query, config):
        with self.lock:
            self.load(config)
            if query in self.text_cache:
                return self.text_cache[query]
            import torch
            try:
                with torch.inference_mode():
                    result = self.vector(self.model.encode_text(self.tokenizer([query]).to(self.device)))
            except RuntimeError as exc:
                raise OpenClipError(f'OpenCLIP text inference failed: {exc}') from exc
            if len(self.text_cache) >= 100:
                self.text_cache.clear()
            self.text_cache[query] = result
            return result


def rank_vectors(rows, query_vector):
    import numpy as np
    query = np.frombuffer(query_vector, dtype='<f4')
    if not query.size or not np.isfinite(query).all():
        raise OpenClipError('Invalid query embedding')
    valid = []
    vectors = []
    for row in rows:
        raw = row['image_embedding']
        if not isinstance(raw, bytes) or len(raw) != len(query_vector):
            continue
        vector = np.frombuffer(raw, dtype='<f4')
        if np.isfinite(vector).all():
            valid.append(row['id'])
            vectors.append(vector)
    if not vectors:
        return []
    scores = np.stack(vectors) @ query
    return sorted(zip(valid, map(float, scores)), key=lambda item: (-item[1], item[0]))
