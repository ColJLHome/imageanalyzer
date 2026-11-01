from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    model: str = Field(min_length=1, max_length=250)
    llama_url: str = Field(min_length=1, max_length=2000)
    roots: list[str] = Field(min_length=1, max_length=100)
    request_timeout_seconds: int = Field(default=180, ge=1, le=3600)
    max_output_tokens: int = Field(default=2048, ge=128, le=32768)
    enable_thinking: bool = False
    analysis_backend: Literal['llama', 'openclip', 'hybrid'] = 'llama'
    openclip_model: str = Field(default='ViT-B-32', min_length=1, max_length=250)
    openclip_pretrained: str = Field(default='laion2b_s34b_b79k', min_length=1, max_length=500)
    openclip_device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    recognize_enabled: bool = False
    recognize_device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    recognize_checkpoint: str = Field(default='', max_length=2000)
    scan_sort_by: Literal['created', 'size', 'name'] = 'name'
    scan_sort_direction: Literal['asc', 'desc'] = 'asc'

    @field_validator('recognize_checkpoint')
    @classmethod
    def validate_checkpoint(cls, value):
        if value and (not Path(value).is_absolute() or '\x00' in value):
            raise ValueError('Use an absolute local checkpoint path, or leave empty for official weights')
        return value

    @field_validator('llama_url')
    @classmethod
    def validate_url(cls, value):
        url = urlsplit(value)
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment):
            raise ValueError('Use an HTTP(S) base URL without credentials, query, or fragment, including /v1')
        # Validate ports as well as the hostname.
        _ = url.port
        return value.rstrip('/')

    @field_validator('roots')
    @classmethod
    def validate_roots(cls, values):
        roots = []
        for value in values:
            value = value.strip()
            if not value or '\x00' in value or not Path(value).is_absolute():
                raise ValueError('Each scan folder must be an absolute Linux path')
            value = str(Path(value).resolve())
            if value not in roots:
                roots.append(value)
        return roots
