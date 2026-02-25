# craft-vani

AI Voice system with Whisper transcription.

## 1. Download Model

Before running the server, you need to download the Whisper model. You can use the provided downloader tool:

```bash
#Install uv from https://docs.astral.sh/uv/getting-started/installation/
cd tools/model_downloader
# Install requirements (uv is recommended)
uv sync
# Download the default small model
uv run main.py --model openai/whisper-small
```

The model will be downloaded to the root directory as `whisper-small`.

## 2. Lanuch using bolt.sh

The `bolt.sh` script provides a convenient way to manage the whole project.

### Setup
Install all necessary dependencies (Rust, Protobuf, etc.):
```bash
./bolt.sh setup
```

### Run Server
```bash
./bolt.sh serve
```

### Launch Frontend
```bash
./bolt.sh launch
```
The frontend will be available at `http://localhost:8080`.

## 3. Docker Deployment

### Prerequisites
- Docker and Docker Compose installed.
- NVIDIA Container Toolkit (for GPU support).

### Quick Start (Default)
By default, the backend uses `whisper-tiny.en` for fast startup.

```bash
docker-compose up --build
```

### Running with a Specific Model
You can specify the model using the `MODEL_NAME` environment variable. Ensure the model is already downloaded in your root directory.

```bash
# Example: Running with whisper-small
MODEL_NAME=whisper-small docker-compose up --build
```
The system will be accessible via Caddy at `http://localhost`.
Modify Caddyfile to enable HTTPS and point to the domain.

### Notes for Hugging Face custom Whisper models
- The backend accepts model weights as either `model.safetensors` or `pytorch_model.bin`.
- Keep the model directory in the project root and pass its directory name via `MODEL_NAME`.
- Example:
```bash
MODEL_NAME=whisper-large-v2-marathi docker-compose up --build
```

### Quality tuning (Whisper)
- Set `CRAFT_VANI_LANGUAGE` to force Whisper language token (example: `mr` for Marathi).
- Increase `CRAFT_VANI_MIN_DECODE_SECONDS` to give each decode more context (example: `4` or `6`).
- Set `CRAFT_VANI_OVERLAP_SECONDS` (for example `1`) to keep trailing context between decode windows and improve continuity.
- Example:
```bash
MODEL_NAME=whisper-large-v2-marathi \
CRAFT_VANI_LANGUAGE=mr \
CRAFT_VANI_MIN_DECODE_SECONDS=4 \
CRAFT_VANI_OVERLAP_SECONDS=1 \
docker-compose up --build
```
