"""
Shared config for local_run_openevolve.py, run_openevolve.py, run_judge.py and viewer.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables from .env file for API_KEY
load_dotenv(os.getenv("CCD_DOTENV", ".env"))

# "cpp" or "rust"
LANGUAGE = os.getenv("CCD_LANGUAGE", "") # Set the default value locally when you do not want to use environment variable 
if LANGUAGE not in ["cpp", "rust"]:
    raise ValueError("LANGUAGE is not set")

ENABLE_FORCE_GCC = os.getenv("CCD_ENABLE_FORCE_GCC", '0') != '0'

# Set the LLM Provider Mode
# Options: see the if clause below
LLM_PROVIDER = os.getenv("CCD_LLM_PROVIDER", "") # Set the default value locally when you do not want to use environment variable 


# --- CONFIGURATION START ---
PROBLEMS_DIR = Path(__file__).parent / "dmoj_problems"
RESULTS_DIR = Path(__file__).parent / f"results_{LLM_PROVIDER}_{LANGUAGE}"


if LLM_PROVIDER == "qwen-480B-Q4":
    host = ''
    with open("../vllm_model_serve/qwen_server_addr.txt", 'r') as f:
        host = f.read().strip('\n ')
    MODEL_NAME: str = "Qwen3-Coder-480B-A35B-Instruct-AWQ"
    API_BASE = f"http://{host}/v1"
    API_KEY = "vllm"

elif LLM_PROVIDER == "gemini-2.5-pro-finetuned":
    # Use this when the vertex ai server at `../vertexai-finetune/server` has started!
    MODEL_NAME: str = "gemini-2.5-pro-tuned"

    # host = os.environ.get("HOST", "localhost")
    host = "localhost"
    with open(Path(os.path.dirname(__file__)) / "../vertexai-finetune/server/port.txt", 'r') as f:
        port = f.read().strip(' \n')

    API_BASE = f"http://{host}:{port}/v1"
    API_KEY = "vertexai"

elif LLM_PROVIDER == "ollama":
    MODEL_NAME: str = "qwen2.5:32b" # Example ollama tag
    API_BASE = "http://localhost:11434/v1"
    API_KEY = "ollama"

elif LLM_PROVIDER == "gpt-oss":
    MODEL_NAME: str = "gpt-oss:20b"
    API_BASE = "http://localhost:11434/v1"
    API_KEY = "ollama"

elif LLM_PROVIDER == "deepseek-finetuned-6.7b":
    host = ''
    with open("../vllm_model_serve/deepseek_server_addr.txt", 'r') as f:
        host = f.read().strip('\n ')
    MODEL_NAME: str = "deepseek-6.7b-finetuned-code-contest-merged"
    API_BASE = f"http://{host}/v1"
    API_KEY = "vllm"

elif LLM_PROVIDER == "openai-gpt-5.2-pro":
    # Use this when the vertex ai server at `../vertexai-finetune/server` has started!
    MODEL_NAME: str = "gpt-5.2-pro"
    
    host = "localhost"
    with open(Path(os.path.dirname(__file__)) / "../vertexai-finetune/server/port.txt", 'r') as f:
        port = f.read().strip(' \n')

    API_BASE = f"http://{host}:{port}/v1"
    API_KEY = os.getenv('OPENAI_API_KEY')

elif LLM_PROVIDER == "gemini-2.5-pro":
    MODEL_NAME: str = "gemini-2.5-pro"
    API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
    API_KEY = os.getenv('GEMINI2_API_KEY')

elif LLM_PROVIDER == "gemini":
    MODEL_NAME: str = "gemini-3-pro-preview"
    API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
    API_KEY = os.getenv('GEMINI3_API_KEY')
# --- CONFIGURATION END ---


if __name__ == "__main__":
    print(f"{LANGUAGE=}, {LLM_PROVIDER=}")
