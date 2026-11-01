"""Filter complete log records, including tracebacks, across retained rotations."""
from collections import deque
import logging
from pathlib import Path
import re

HEADER = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+) (DEBUG|INFO|WARNING|ERROR|CRITICAL) (?:\[(error|warning|success|activity)\] )?(.*)$')
SUCCESS = re.compile(r'^(Indexed image in |Reused analysis by SHA-256:|OpenCLIP (?:indexed|hash reused):|Database backup created|Database restored;|Settings saved;|Image data manually corrected:|Removed image from index:|Removed \d+ records from hidden folders)')


def category_for(level, message, explicit=None):
    if level in ('ERROR', 'CRITICAL'):
        return 'error'
    if level == 'WARNING':
        return 'warning'
    return explicit or ('success' if SUCCESS.match(message) else 'activity')


class CategoryFilter(logging.Filter):
    def filter(self, record):
        record.category = category_for(record.levelname, record.getMessage(), getattr(record, 'category', None))
        return True


def parse_events(text):
    event = None
    for line in text.splitlines():
        match = HEADER.match(line)
        if match:
            if event:
                yield event
            timestamp, level, explicit, message = match.groups()
            event = {'timestamp': timestamp, 'level': level,
                     'category': category_for(level, message, explicit), 'text': line}
        elif event:
            event['text'] += '\n' + line
    if event:
        yield event


def read_events(path, limit=300, category='all', query=''):
    entries = deque(maxlen=limit)
    matched = total = 0
    truncated = False
    query = query.casefold()
    for suffix in ('.3', '.2', '.1', ''):
        candidate = Path(str(path) + suffix)
        try:
            with candidate.open('rb') as stream:
                size = stream.seek(0, 2)
                # Four files, bounded to about 8.4 MB total. Ignore any orphaned
                # traceback fragment when starting partway through a large file.
                start = max(0, size - 2_100_000)
                stream.seek(start)
                if start:
                    stream.readline()
                    truncated = True
                text = stream.read(2_100_000).decode('utf-8', errors='replace')
        except FileNotFoundError:
            continue  # Rotation may have renamed a file while we were reading.
        for event in parse_events(text):
            total += 1
            selected = (category == 'all' or event['category'] == category or
                        (category == 'problems' and event['category'] in ('error', 'warning')))
            if selected and (not query or query in event['text'].casefold()):
                matched += 1
                entries.append(event)
    result = list(entries)
    # Retain the previous line-based API for clients using only the lines field.
    lines = '\n'.join(event['text'] for event in result).splitlines()[-limit:]
    return {'entries': result, 'lines': lines, 'matched': matched, 'total_events': total,
            'read_limit_reached': truncated}
