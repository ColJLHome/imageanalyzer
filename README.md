# Image Analyser

A Linux service that recursively scans mounted image folders, sends images to a vision model in llama-server, and provides a browser search interface. Stores descriptions, scenes, objects with colour attributes, named colours, and a palette measured from image pixels. Originals are never modified.

## Run

Requires Python 3.11+ and a running, vision-capable llama-server with its matching multimodal projector. The application does not download or launch the model. Configure llama-server with an alias such as `--alias local-vision` that matches `model` below.

```bash
uv sync
cp config.example.json config.json
# Edit config.json with your mount paths and llama-server URL.
uv run imageanalyser --config config.json
```

Open http://127.0.0.1:8090. Scanning starts automatically and repeats hourly by default. Search results refresh when you submit the search form. All search terms must match; object, scene, and colour filters combine. Filters use substring matching against the model's tags; search is not semantic and does not guarantee object/colour relationships beyond the generated tags. Palettes are displayed for reference; the colour filter searches named model tags.

`llama_url` should include `/v1`, for example `http://127.0.0.1:8080/v1`. Set `LLAMA_API_KEY` in the service environment if your llama-server requires authentication. The `model` value must match its served model name or alias. Start with one request at a time; benchmark a representative folder before scanning everything. Requests use a JSON schema and ask the chat template to disable thinking (`enable_thinking: false`, the default). The output budget defaults to `max_output_tokens: 2048`. If a response is empty, invalid, or truncated, logs include its finish reason, a bounded preview of final content, reasoning character count, and usage metadata. Thinking/template support depends on the model and llama-server build; check the server configuration if it ignores the request. An older server may need upgrading for schema output support.

For Ryzen AI Max hardware, use a llama.cpp build with working GPU acceleration (Vulkan or compatible ROCm). Confirm GPU use in llama-server's logs. Select the model and GPU memory allocation for your machine; the indexer works independently of the model choice.

Use **Scan for new images** on the Images tab to check all configured folders immediately. New and changed images are analysed; unchanged indexed images are skipped. This also resumes scanning if paused. If a scan is already running, the existing scan continues. Search results refresh as newly analysed images become available.

## Clearing the index and diagnosing issues

Every image card has a **Remove from index** button. It removes that image's analysis and search data without modifying the original file. SQLite retains its file identity with an exclusion marker so automatic scans and restarts do not add it again, even if the file changes. In-flight analysis cannot restore a removed image. **Clear image index** also resets these exclusions, allowing a full rebuild to include those files again.

On the Images tab, **Clear image index** removes all stored image records and pauses scanning. It does not delete or modify source images. The pause persists across restarts. An analysis already in progress may finish its request, but its result is discarded. Select **Resume scanning** to rebuild the index from your configured folders.

The **Scanner details** tab shows current activity, folder status, and recent errors. **Clear scanner details** dismisses these diagnostics without deleting image records, changing retry schedules, pausing scanning, or removing logs. New activity and errors appear as scanning continues.

The **Logs** tab refreshes every five seconds while open and shows recent scanner events, image analysis durations, unavailable roots, and exception tracebacks. Logs are written to `data_dir/imageanalyser.log`, with three rotating backups of up to 2 MB each. The tab displays up to 300 recent lines from the current file; older entries are available in the backups. Clearing images retains diagnostic logs. These are application logs; llama-server's own GPU and inference logs remain in its console or service journal.

## Settings screen

Open the **Settings** tab to change the llama-server base URL, model name/alias, scan folders (one absolute Linux path per line), request timeout, output-token budget, and thinking option. Folder paths refer to the server's filesystem; unavailable mounts may be saved and will be reported by the scanner. Saving queues a scan with the new settings unless scanning is paused. An in-flight request finishes but its result is discarded when settings change. Failed records become eligible for retry.

Settings saved through the screen are stored in the local SQLite database and override the corresponding `config.json` values on subsequent starts. Other configuration, including data directory, bind address, port, and scan interval, stays in `config.json`. The settings screen does not store or expose API keys; authentication continues to use `LLAMA_API_KEY` in the application server environment.

Changing the model field selects the name/alias sent with requests; model installation and loading are managed by llama-server. Existing completed analyses are retained. To rebuild them with the new model, clear the image index and scan again. Records from removed folders remain searchable, while their previews are blocked because they are outside the configured roots.

## Configuration

See `config.example.json`. Roots are absolute Linux paths to already mounted SMB/NFS/local folders; mount them read-only where practical. `data_dir` is a writable **local disk** directory for SQLite. Relative data paths resolve against the process working directory. Supported files: JPEG, PNG, WebP, BMP, TIFF, and GIF (first frame only). HEIC and camera RAW are not supported. EXIF rotation is applied, and analysis copies are resized to `max_image_edge` without changing originals.

Size and nanosecond modification time identify changed files. Successful unchanged images are skipped across restarts. Failed files are retried on later scans with a delay of up to one hour. One worker limits inference pressure; run exactly one application process against a database. Model responses are validated before indexing. Changing models does not automatically reanalyse completed images; use a new data directory to build a fresh index.

Unavailable folders and traversal errors appear in scanner details. Records are retained when mounts disappear. Deleted files also remain searchable until you rebuild the index, with unavailable previews. No automatic deletion avoids losing the index when a network share appears empty. Ensure mounts are present before launch; periodic scans pick up mounts that arrive later. Symlink files and directories are skipped.

## Linux service

Install the project with `uv sync` first, then adapt the paths and user in this unit:

```ini
[Unit]
Description=Image Analyser
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/mnt/photos /mnt/archive/images

[Service]
Type=simple
User=imageanalyser
WorkingDirectory=/opt/imageanalyser
ExecStart=/opt/imageanalyser/.venv/bin/imageanalyser --config /opt/imageanalyser/config.json
Restart=on-failure
RestartSec=10
UMask=0077

[Install]
WantedBy=multi-user.target
```

Save as `/etc/systemd/system/imageanalyser.service`, ensure the service user can read the configured roots and write `data_dir`, then run:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now imageanalyser
journalctl -u imageanalyser -f
```

The web service binds to loopback by default. Use an SSH tunnel for remote access (`ssh -L 8090:127.0.0.1:8090 server`). If you change `host` to expose it on the LAN, restrict access or place it behind an authenticated reverse proxy: the application has no built-in login and displays image previews and full file paths. Images are sent to the configured llama-server; point it at your own trusted instance.

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

Tests use a fake analysis function and temporary images; they do not require a GPU or llama-server.
