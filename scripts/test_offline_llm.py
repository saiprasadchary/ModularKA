# scripts/test_offline_llm.py
"""
Quick health-check for Ollama offline mode.

What it does:
1. Loads .env to get OLLAMA_BASE_URL and OLLAMA_GENERAL_MODEL.
2. Pings /api/tags to confirm the Ollama server is reachable.
3. Sends a tiny /api/generate request to the general model.
4. Prints a short snippet of the response for sanity.

Run with:
    cd /path/to/modularka
    python scripts/test_offline_llm.py
"""

import os
import sys
import json
import time
from pathlib import Path

import requests
from dotenv import load_dotenv


def main():
    # Ensure we're running from project root if script is called from elsewhere
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent
    os.chdir(project_root)

    # Load .env from project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"[WARN] .env not found at {env_path}, relying on process env.")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = os.getenv("OLLAMA_GENERAL_MODEL", "llama3:8b")

    print("=== Ollama Offline Health Check ===")
    print(f"Base URL : {base_url}")
    print(f"Model    : {model_name}")
    print("-----------------------------------")

    # 1) Ping /api/tags
    tags_url = f"{base_url}/api/tags"
    try:
        print(f"[1/2] Checking Ollama server at {tags_url} ...")
        resp = requests.get(tags_url, timeout=3)
        resp.raise_for_status()
        tags_json = resp.json()
        available_models = [m.get("name") for m in tags_json.get("models", [])]
        print(f"  ✓ Ollama server reachable. Models: {available_models}")
    except Exception as e:
        print(f"  ✗ Failed to reach Ollama server: {e}")
        print("  Hint: ensure `ollama serve` is running.")
        sys.exit(1)

    # 2) Small /api/generate call
    gen_url = f"{base_url}/api/generate"
    prompt = "Briefly say: 'Ollama offline mode is working.'"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,  # get a single JSON response instead of a stream
    }

    try:
        print(f"[2/2] Testing generate endpoint at {gen_url} ...")
        start = time.time()
        resp = requests.post(gen_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "") or data.get("output", "")
        elapsed = time.time() - start

        if not text:
            print("  ✗ Generate call succeeded but response body is empty/unknown.")
        else:
            snippet = text[:400].replace("\n", " ")
            print(f"  ✓ Generate OK in {elapsed:.2f}s")
            print(f"  Response snippet: {snippet!r}")

    except Exception as e:
        print(f"  ✗ Generate call failed: {e}")
        print("  Hint: ensure the model is pulled, e.g. `ollama pull llama3:8b`.")
        sys.exit(1)

    print("\n✅ Ollama offline health-check completed successfully.")


if __name__ == "__main__":
    main()