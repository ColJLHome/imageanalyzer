"""Read embedded AUTOMATIC1111 generation text without sending it to the vision model."""
import re

from PIL import Image


def decode_comment(value):
    if isinstance(value, str):
        return value.strip('\x00 \r\n')
    if not isinstance(value, bytes):
        return ''
    if value.startswith(b'UNICODE\x00'):
        payload = value[8:]
        encoding = 'utf-16' if payload.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-16-be'
    elif value.startswith(b'ASCII\x00\x00\x00'):
        payload, encoding = value[8:], 'utf-8'
    else:
        payload, encoding = value, 'utf-8'
    return payload.decode(encoding, errors='replace').strip('\x00 \r\n')


def read_a1111_metadata(path):
    with Image.open(path) as image:
        text = decode_comment(image.info.get('parameters'))
        if not text:
            exif = image.getexif()
            comment = exif.get_ifd(0x8769).get(37510) if 0x8769 in exif else exif.get(37510)
            text = decode_comment(comment or image.info.get('comment'))
            if not re.search(r'(?m)^Steps:\s*\d+', text):
                return None
    if not text:
        return None
    text = text[:128_000]
    match = re.search(r'(?m)^Steps:\s*\d+', text)
    prompts = text[:match.start()].rstrip() if match else text
    parameters = text[match.start():].strip() if match else ''
    positive, separator, negative = prompts.partition('\nNegative prompt:')
    if prompts.startswith('Negative prompt:'):
        positive, negative = '', prompts[len('Negative prompt:'):]
    elif not separator:
        negative = ''
    return {'raw': text, 'prompt': positive.strip(), 'negative_prompt': negative.strip(), 'parameters': parameters}
