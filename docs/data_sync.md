# Sharing data between machines (rclone + Google Drive)

Code and the benchmark (`data/benchmark/`) travel through git. Everything else that the experiments need is git-ignored and travels through Google Drive with [rclone](https://rclone.org):

| Directory | Content | Can it be rebuilt? |
|---|---|---|
| `data/raw/` | Frozen OSM snapshots | **No.** OSM changes daily; a new download gives a different map. Never re-download an existing area id. |
| `data/processed/` | Processed graph, text corpus with embeddings, LightRAG index | The graph and corpus text are deterministic. The embeddings and the LightRAG index are not reproduced exactly (the index is LLM-extracted and takes hours), so they are copied. |
| `runs/` | Immutable run outputs | No; runs are never recomputed. |
| `.cache/` | LLM response cache | Lets identical requests be replayed without calling the model. |

Commands to run, by machine:

- On the machine that produced the data: `scripts/push_data.py` uploads these directories to `gdrive:graph_llm_data`. It also writes `data/data_manifest.json`, which records the size and SHA-256 of every file and must be committed to git.
- On the other machine: `scripts/setup_environment.py` downloads the data and checks every file against that manifest. The code version and the data version are therefore tied together through git.

Uploads use `rclone copy`, never `rclone sync`, so nothing on Drive is ever deleted or overwritten by a missing local file.

## 1. Install rclone (once per machine, no admin rights needed)

### macOS (Apple Silicon)

```bash
mkdir -p ~/bin && cd /tmp
curl -LO https://downloads.rclone.org/rclone-current-osx-arm64.zip
unzip -o rclone-current-osx-arm64.zip
cp rclone-*-osx-arm64/rclone ~/bin/
xattr -d com.apple.quarantine ~/bin/rclone 2>/dev/null
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
rclone version
```

### Linux server (x86-64)

```bash
mkdir -p ~/bin && cd ~
curl -LO https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip -o rclone-current-linux-amd64.zip
cp rclone-*-linux-amd64/rclone ~/bin/
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
rclone version
```

The server is shared: rclone is installed only in your home directory, and nothing outside this project is touched.

## 2. Connect rclone to Google Drive (once per machine)

The remote must be named `gdrive`; the scripts use `gdrive:graph_llm_data`.

### On a machine with a browser (the Mac)

```bash
rclone config create gdrive drive scope=drive
```

A browser window opens. Log in with the Google account that owns the Drive folder and click **Allow**.

### On a machine without a browser (the server)

```bash
rclone config
```

Answer: `n` (new remote), name `gdrive`, storage `drive`, empty `client_id` and `client_secret`, scope `1` (full access), empty service account file, `n` for advanced config, and **`n` for "Use web browser to automatically authenticate?"**. rclone prints a command of the form:

```bash
rclone authorize "drive" "eyJzY29wZSI6..."
```

Run that exact command **on the Mac**, log in in the browser that opens, then copy the token the Mac prints and paste it into the waiting prompt on the server. Answer `n` to "Shared Drive" and `y` to keep the remote.

### Check the connection

```bash
rclone lsd gdrive:                     # lists top-level Drive folders
rclone mkdir gdrive:graph_llm_data     # only needed the first time
rclone ls gdrive:graph_llm_data        # lists the shared data files
```

**Security.** The configuration, including the OAuth token, is stored in `~/.config/rclone/rclone.conf`. Keep it private (`chmod 600`), never commit it, and never paste it into logs or chats. Anyone with this file can read and write your whole Drive.

## 3. Upload data (on the machine that produced it)

```bash
uv run python scripts/push_data.py                            # all four directories
uv run python scripts/push_data.py --directories data/raw     # only some directories
git add data/data_manifest.json && git commit -m "data: update data manifest"
```

Do not push while a long job is still writing into a directory (for example a LightRAG index being built), or the manifest will describe incomplete files.

## 4. Set up another machine

```bash
git pull
uv sync
uv run python scripts/setup_environment.py
```

What it does:

1. Reads `data/data_manifest.json` and checks that free disk space is at least twice the data size.
2. Downloads the listed directories from the remote and verifies every file's SHA-256. It stops and lists the files that are missing or changed. Local files that are not in the manifest, such as newer runs, are left alone.
3. With `--verify-rebuild`, it rebuilds every processed graph in `configs/graphs/` from its snapshot in a temporary directory, and checks that the rebuilt files are byte-identical to the downloaded ones. This proves that the pipeline is reproducible.
4. With `--llm-config <file>`, it sends one short chat request to that endpoint, bypassing the response cache.
5. Runs the test suite.

Full check on the office server:

```bash
export VLLM_API_KEY=...    # the --api-key of the running vllm serve; never commit it
uv run python scripts/setup_environment.py --verify-rebuild \
    --llm-config configs/llm/vllm_server_qwen3_32b.yaml
```

Other options: `--remote <remote:path>` overrides the remote recorded in the manifest.

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `rclone is not installed` | Install it into `~/bin` (Section 1). The scripts also look in `~/bin` when it is not on `PATH`. |
| `NOTICE: ... shared Google Drive client_id, which is being retired` | rclone's built-in client id is being retired during 2026. When it stops working, create your own client id as described at <https://rclone.org/drive/#making-your-own-client-id>, then run `rclone config`, edit the `gdrive` remote, enter the new `client_id` and `client_secret`, and authorize again. |
| `token expired` or `invalid_grant` | Run `rclone config reconnect gdrive:`. |
| `changed: <file>` after download | The Drive copy differs from what the manifest in git describes. Usually the data was pushed again without committing the new manifest, or the manifest was committed without pushing. Rerun `push_data.py` on the source machine and commit the manifest. |
| `rebuilt graph differs` with `--verify-rebuild` | The library versions differ between machines (for example OSMnx). Check `uv.lock` is the same and run `uv sync`. |
