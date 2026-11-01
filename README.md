# Image Analyser

Search a collection of images on Linux using natural language, tags, colours, and generation metadata. The app scans mounted folders recursively and stores results in SQLite. Source images are never modified.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). For detailed descriptions, run a vision-capable llama-server with the matching multimodal projector.

```bash
uv sync --all-extras
cp config.example.json config.json
# Set your mounted folder paths, llama-server URL, and model alias.
uv run --all-extras imageanalyser --config config.json
```

Open http://127.0.0.1:8090 (or your configured address). Scanning starts automatically and repeats hourly by default. Model weights download on first use.

In `config.json`:

- `roots`: absolute paths to mounted image folders.
- `data_dir`: writable directory on local disk for SQLite and logs.
- `llama_url`: server URL including `/v1`, such as `http://127.0.0.1:8080/v1`.
- `model`: the name or alias served by llama-server.
- `host` / `port`: web interface address; defaults to loopback on port 8090.

Set `LLAMA_API_KEY` in the environment if needed. See [config.example.json](config.example.json) for other options. Settings saved through the UI override their corresponding config values.

## Analysis and search

Choose **Both** in Settings for three collection-wide passes, in this order:

1. **OpenCLIP + A1111 metadata**: visual embeddings for natural-language search and saved generation prompts/settings.
2. **Recognize Anything (RAM++)**: optional object and scene tags.
3. **llama-server**: descriptions, colours, objects, clothing, and visible people details.

OpenCLIP-only and llama-server-only modes are also available. The supplied PyTorch build uses CPU; llama-server can use its own GPU acceleration. RAM++ uses the [official Recognize Anything model](https://github.com/xinyu1205/recognize-anything). Configure models, devices, folders, and analysis order in **Settings**.

- **Natural language**: try “a person wearing a blue jacket on a beach”. Results rank by OpenCLIP visual similarity; scores are not probabilities. **Natural language + metadata** also searches saved text. Optional object/scene/colour filters match saved tags.
- **Sort**: browse by last tagged/indexed, creation date, or name in either direction. Natural-language queries automatically show best matches first. Reset clears search fields and restores the previous browsing order unless you changed it manually.
- **Preview**: hover over a thumbnail for one second to see a larger image occupying roughly half the screen. Move away or press Escape to close it. Keyboard focus also opens the preview.
- **Scan badges**: hover, focus, or tap OpenCLIP/A1111/RAM++/llama-server badges to inspect stored results. “A1111 (none)” means metadata was checked but absent.
- **Edit data**: correct descriptions, tags, people details, or metadata. Corrections survive automatic analysis of unchanged image content.

## Managing the index

**Scan for new images** checks configured folders immediately. Progress shows the current pass, checked/remaining counts, and estimated time. Completed work resumes after restarts. SHA-256 matching reuses analysis when identical files move or are copied.

**Remove from index** excludes an individual path from future scans. **Settings → Clear image index** clears records and exclusions after confirmation, then pauses scanning. Resume to rebuild. Neither action deletes source files.

**Settings → Database backup and restore** downloads a consistent SQLite snapshot or restores one after confirmation. Backups include indexed results, embeddings, corrections, exclusions, and saved settings; originals, logs, `config.json`, and API keys are separate. Restore first creates a safety backup in `data_dir/backups/` and leaves scanning paused.

**Scanner details** shows current activity and folder problems. **Logs** supports severity/success filters, text search, and copying matching events. Logs are stored at `data_dir/imageanalyser.log` with rotating backups. **Settings → Restart backend** restarts the app when launched with the command above; it does not restart llama-server.

Supported images: JPEG, PNG, WebP, BMP, TIFF, and GIF (first frame). Hidden folders and symlinks are skipped. Unavailable or deleted files retain their indexed results; previews require accessible files inside configured roots. Creation-date sorting falls back to modification time when necessary.

Run one application process per database. The web interface has no login: keep it on a trusted network or use an authenticated reverse proxy. For remote access without exposing the service, use `ssh -L 8090:127.0.0.1:8090 server`.

## Linux service

After installing, adapt these paths, mount points, and user in `/etc/systemd/system/imageanalyser.service`:

```ini
[Unit]
Description=Image Analyser
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/mnt/photos

[Service]
User=imageanalyser
WorkingDirectory=/opt/imageanalyser
ExecStart=/opt/imageanalyser/.venv/bin/imageanalyser --config /opt/imageanalyser/config.json
Restart=on-failure
RestartSec=10
UMask=0077

[Install]
WantedBy=multi-user.target
```

Ensure the service user can read the image folders and write `data_dir`, then run `sudo systemctl daemon-reload` and `sudo systemctl enable --now imageanalyser`.

## Tests

```bash
uv run --all-extras python -m unittest discover -s tests -q
```

Tests use temporary images and mocked inference; no GPU or llama-server is required.
