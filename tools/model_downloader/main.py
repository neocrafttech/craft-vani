import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="openai/whisper-small",
        help="Hugging Face model repository ID (default: openai/whisper-small)",
    )
    args = parser.parse_args()

    repo_id = args.model
    # Extract the model name from the repo_id (e.g., 'openai/whisper-small' -> 'whisper-small')
    model_name = repo_id.split("/")[-1]
    local_dir = os.path.join("../../", model_name)

    print(f"Downloading model '{repo_id}' to '{local_dir}'...")

    snapshot_download(
        repo_id=repo_id,
        allow_patterns=["*.json", "*.safetensors"],
        local_dir=local_dir,
    )


if __name__ == "__main__":
    main()
