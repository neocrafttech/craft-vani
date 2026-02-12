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
