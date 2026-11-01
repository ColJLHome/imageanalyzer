# Image Analyser

A Linux service that recursively scans mounted image folders, sends images to a vision model in llama-server, and provides a browser search interface. Stores descriptions, scenes, objects with colour attributes, named colours, and a palette measured from image pixels. Originals are never modified.

## Run

Requires Python 3.11+ and a running, vision-capable llama-server with its matching multimodal projector. The application does not download or launch the model. Configure llama-server with an alias such as `--alias local-vision` that matches `model` below.

```bash
uv sync --extra openclip
cp config.example.json config.json
# Edit config.json with your mount paths and llama-server URL.
uv run --extra openclip imageanalyser --config config.json
```

Open http://127.0.0.1:8090. Scanning starts automatically and repeats hourly by default. Search results refresh when you submit the search form. All search terms must match; object, scene, and colour filters combine. Filters use substring matching against the model's tags; search is not semantic and does not guarantee object/colour relationships beyond the generated tags. Palettes are displayed for reference; the colour filter searches named model tags.

`llama_url` should include `/v1`, for example `http://127.0.0.1:8080/v1`. Set `LLAMA_API_KEY` in the service environment if your llama-server requires authentication. The `model` value must match its served model name or alias. Start with one request at a time; benchmark a representative folder before scanning everything. Requests use a JSON schema and ask the chat template to disable thinking (`enable_thinking: false`, the default). The output budget defaults to `max_output_tokens: 2048`. If a response is empty, invalid, or truncated, logs include its finish reason, a bounded preview of final content, reasoning character count, and usage metadata. Thinking/template support depends on the model and llama-server build; check the server configuration if it ignores the request. An older server may need upgrading for schema output support.

For Ryzen AI Max hardware, use a llama.cpp build with working GPU acceleration (Vulkan or compatible ROCm). Confirm GPU use in llama-server's logs. Select the model and GPU memory allocation for your machine; the indexer works independently of the model choice.

The progress display first shows image discovery, then the number checked and remaining for the current scan. Beside the remaining count, an estimated time uses the mean duration of completed model analysis attempts in that scan, allowing for time already spent on the active image. It shows “calculating” before the first timed attempt and is hidden while paused. Cached and skipped files do not dilute the analysis-time average, so the estimate may be high when many remaining files can be skipped or reused. Every supported file counts as checked after processing, including unchanged, excluded, cached, and failed images. The total covers files discovered on available roots; mount and traversal errors are reported separately. Progress is live for the current process and resets on restart.

Use **Scan for new images** on the Images tab to check all configured folders immediately. New and changed images are analysed; unchanged indexed images are skipped. This also resumes scanning if paused. If a scan is already running, the existing scan continues. Search results refresh as newly analysed images become available.

## Clearing the index and diagnosing issues

Every image card has a **Remove from index** button. It removes that image's analysis and search data without modifying the original file. SQLite retains its file identity with an exclusion marker so automatic scans and restarts do not add it again, even if the file changes. In-flight analysis cannot restore a removed image. **Clear image index** also resets these exclusions, allowing a full rebuild to include those files again.

On the Settings tab, **Clear image index** removes all stored image records and pauses scanning. It does not delete or modify source images. The pause persists across restarts. An analysis already in progress may finish its request, but its result is discarded. Select **Resume scanning** to rebuild the index from your configured folders.

The **Scanner details** tab shows current activity, folder status, and recent errors. **Clear scanner details** dismisses these diagnostics without deleting image records, changing retry schedules, pausing scanning, or removing logs. New activity and errors appear as scanning continues.

The **Logs** tab refreshes every five seconds while open and shows scanner events, image analysis durations, unavailable roots, and exception tracebacks. Filter by **Problems** (errors and warnings), **Errors**, **Warnings**, **Successes**, or **General activity**, and search by message text or image path. Filters search the retained rotations before selecting the latest 300 matching events, so noisy success messages do not hide older failures. Tracebacks remain attached to their error event. **Copy logs** copies only the displayed matching events. Event colours distinguish severity and success. Logs are written to `data_dir/imageanalyser.log`, with three rotating backups of up to 2 MB each. The tab displays up to 300 complete matching events across the current file and three backups, reading at most about 8.4 MB per refresh. Clearing images retains diagnostic logs. These are application logs; llama-server's own GPU and inference logs remain in its console or service journal.

## Correcting image data

Select **Edit data** on an image card to correct its description, scenes, object/clothing tags, colours, individual people details, and saved A1111 metadata. The form provides ordinary fields for each person; no JSON editing is required. Saving updates search immediately, retains OpenCLIP embeddings, and marks the entry **Manually corrected**. These corrections only change the database, not source images or embedded metadata.

Automatic analysis of unchanged content, including an in-flight caption request, cannot overwrite manual corrections. Corrected data is also preserved during embedding rebuilds and hash reuse after moves/copies. Replacing an image's bytes permits fresh analysis for the new content. Clearing the image index removes corrections along with the other records, so include them in database backups.

## OpenCLIP and search backends

Install OpenCLIP support with `uv sync --extra openclip` and launch with `uv run --extra openclip imageanalyser --config config.json`. The extra installs a CPU PyTorch build so it runs on Linux without GPU setup. CPU inference can run alongside llama-server on the GPU. Advanced users can provide a compatible CUDA/ROCm PyTorch environment and select the GPU device; the supplied CPU build does not provide GPU acceleration. Use the installed `.venv/bin/imageanalyser` command if you manage PyTorch packages separately, so a normal `uv sync` does not replace your chosen build.

In Settings choose **llama-server**, **OpenCLIP**, or **Both**. Both performs a full pass extracting image embeddings and A1111 metadata across every discovered image before beginning the llama-server pass. The scanner order applies to each pass. OpenCLIP-only never calls llama-server; it preserves existing descriptions and adds searchable image vectors. Embeddings are stored as normalized float32 vectors in SQLite, keyed by image SHA-256 and OpenCLIP model/weights identity. Identical images reuse embeddings, and changing models builds a separate set for the new model. Completed embeddings resume after restarts. Removed images remain excluded. A model-loading failure stops the pass rather than starting llama-server without OpenCLIP coverage; individual unreadable/corrupt files are logged and skipped.

The initial default model is `ViT-B-32` with `laion2b_s34b_b79k` weights. First use downloads weights into the Hugging Face cache. To pre-download them use:

```bash
hf download laion/CLIP-ViT-B-32-laion2B-s34B-b79K open_clip_pytorch_model.bin open_clip_config.json
```

Hover over an image's **OpenCLIP** badge (or focus it with the keyboard) to see a scrollable tooltip with the full stored vector, model/weights identity, dimensions, image hash, validation flags, and the current search similarity score when available. Data is loaded on demand rather than included in every search response. The tooltip stays open while you move into it to scroll, and Escape dismisses it.

Search modes are **Visual similarity** (rank image embeddings against the query), **Keywords and metadata** (the existing text match), and **Combined search** (reciprocal-rank fusion of both). Visual scores are cosine similarity, not probabilities or confirmed detections. Semantic ranking returns indexed images by relevance, including low-scoring results; it does not require literal tag matches. Scene/object/colour filters still use stored llama-server tags, so they may exclude images that only have embeddings. Combined search falls back to keywords when the selected backend does not enable OpenCLIP. Existing captions, people details, metadata, backups, and exclusions are retained. OpenCLIP does not generate new person descriptions; those appear during the later llama-server pass.

## A1111 generation metadata

Embedded AUTOMATIC1111 `parameters` text in PNG files and EXIF UserComment text in JPEG/WebP files is stored in SQLite and included in the main text search. Search can match positive and negative prompts, LoRA names, model names, seeds, samplers, and other saved generation parameters. Expand **A1111 generation metadata** on an image card to view the original text. Prompt text is kept separate from visual descriptions and is not sent to the vision model; generation prompts, especially negative prompts, do not establish what is visible in the image. Sidecar `.txt` files are not read.

On upgrading to this version, older completed analyses are automatically refreshed once during scanning, adding metadata and current people details. Hash reuse only uses results from the current analysis version, so an old result cannot bypass the refresh. Individual exclusions remain excluded, and later scans skip the refreshed images. Missing or unmounted originals wait until their folders become available. A paused scanner can be resumed with **Scan for new images**.

## People details

New analyses include up to ten prominent people, describing their position in the image, visible clothing and footwear (garment types, colours, patterns, layers, and fit), and accessories. Expand **People** on an image card to see the details. Approximate age groups and visible gender presentation are labelled as estimates, with an unclear option; these do not establish exact age, sex, or gender identity. People details are included in the main text search, for example `blue striped jacket`, `young adult`, or `round glasses`.

Previously indexed images retain their existing analysis, including when reused by hash. To generate the additional details for your collection, back up the database, use **Settings → Clear image index**, then **Images → Scan for new images**. This also resets individual exclusions as described below. For crowded images, a higher output-token limit may be needed to fit the additional detail.

## Settings screen

Open the **Settings** tab to change the llama-server base URL, model name/alias, scan folders (one absolute Linux path per line), request timeout, output-token budget, and thinking option. Folder paths refer to the server's filesystem; unavailable mounts may be saved and will be reported by the scanner. Saving queues a scan with the new settings unless scanning is paused. An in-flight request finishes but its result is discarded when settings change. Failed records become eligible for retry.

Settings saved through the screen are stored in the local SQLite database and override the corresponding `config.json` values on subsequent starts. Other configuration, including data directory, bind address, port, and scan interval, stays in `config.json`. The settings screen does not store or expose API keys; authentication continues to use `LLAMA_API_KEY` in the application server environment.

Changing the model field selects the name/alias sent with requests; model installation and loading are managed by llama-server. Existing completed analyses are retained. To rebuild them with the new model, clear the image index and scan again. Records from removed folders remain searchable, while their previews are blocked because they are outside the configured roots.

### Restarting the backend

In **Settings → Backend**, select **Restart backend**, then confirm. The interface waits for a new backend instance and reconnects automatically. The database, saved settings, exclusions, and persisted scanner pause state are retained. An unfinished analysis is discarded and becomes eligible for retry after startup. Unsaved form edits are not retained. This restarts the image analyser only; it does not restart llama-server or the machine.

Launch through `uv run --extra openclip imageanalyser --config config.json` (or the equivalent installed command). That command runs a supervisor which starts a fresh backend process on a confirmed web restart, reloading code and `config.json`. It also works with the systemd unit below. Direct custom ASGI launches do not enable the restart button. After first installing this feature, stop and start the old application once to activate the supervisor.

### Analysis order

Settings includes **Analyse files in order of** (name, date created, or file size) and **Order** (ascending or descending). The default is name ascending. Name sorting is case-insensitive; date ascending means oldest first and size ascending means smallest first. Ordering is global across all configured roots, with filenames and full paths breaking ties deterministically. Unreadable files are checked last and reported in the logs. Saving a new order switches the scanner to that order after any active analysis finishes; completed files remain indexed and are skipped.

Date created uses the filesystem birth timestamp, including Linux `statx` where supported. Some network shares and filesystems do not supply one; those files use modification time, and the fallback count is logged. Linux inode-change time (`ctime`) is never used as creation time.

## Database backup and restore

In **Settings → Database backup and restore**, select **Download database backup** to save a consistent SQLite snapshot, even while scanning. Backups include image records, analyses, hashes, exclusions, and database settings. Original images, logs, `config.json`, and environment API keys are not included.

Choose a backup file and select **Restore database**, then confirm in the dialog. Uploads are limited to 256 MB and must be a valid Image Analyser database from the current schema version (including SHA-256 support). The server checks SQLite integrity, schema, saved settings, and completed analysis data before changing the active database. It first saves a safety backup at `data_dir/backups/before-restore-*.sqlite3`. Restore replaces the index and database settings, discards in-flight results, and leaves scanning paused. Check the restored folder paths and server settings, then select **Scan for new images** to resume. Safety backups are retained until you remove them manually; they can also be uploaded to restore the previous state.

## Configuration

See `config.example.json`. Roots are absolute Linux paths to already mounted SMB/NFS/local folders; mount them read-only where practical. `data_dir` is a writable **local disk** directory for SQLite. Relative data paths resolve against the process working directory. Supported files: JPEG, PNG, WebP, BMP, TIFF, and GIF (first frame only). HEIC and camera RAW are not supported. EXIF rotation is applied, and analysis copies are resized to `max_image_edge` without changing originals.

The database stores a SHA-256 hash of each image file's bytes, with an index for fast matching. New or changed files are hashed in streaming chunks; a matching successfully indexed image supplies its stored analysis without another model call. This handles moved, renamed, and byte-identical copied files, retaining the model that produced the reused analysis. Each path remains a separate record; old unavailable paths are retained under the existing retention policy. Metadata changes alone also reuse analysis if the bytes match. Visually identical files with different encoding or embedded metadata have different hashes.

Existing databases migrate automatically. Unchanged, accessible legacy records acquire hashes on their next scan without reanalysis; files moved before their first hash was stored cannot be matched to those legacy records. Removed records never supply cached analysis. Clearing the index removes the stored hashes and analyses as part of the full rebuild.

Size and nanosecond modification time identify changed files. Successful unchanged images are skipped across restarts. Failed files are retried on later scans with a delay of up to one hour. One worker limits inference pressure; run exactly one application process against a database. Model responses are validated before indexing. Changing models does not automatically reanalyse completed images; use a new data directory to build a fresh index.

Unavailable folders and traversal errors appear in scanner details. Records are retained when mounts disappear. Deleted files also remain searchable until you rebuild the index, with unavailable previews. No automatic deletion avoids losing the index when a network share appears empty. Ensure mounts are present before launch; periodic scans pick up mounts that arrive later. Symlink files and directories are skipped. All hidden folders (any directory component beginning with a dot, including `.imageviewer-thumbnails`, `.cache`, and `.git`) are excluded, even when explicitly configured as scan roots. Existing records under hidden folders are removed automatically on startup and after restoring a backup, including their analyses and embeddings. Source files are never deleted. A dot-prefixed image filename in an otherwise visible folder remains eligible; only hidden folders are excluded.

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
uv run --extra openclip python -m unittest discover -s tests -v
```

Tests use a fake analysis function and temporary images; they do not require a GPU or llama-server.
