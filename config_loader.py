# config_loader.py
"""
Config Loader
Reads saas_config.yaml OR saas_config.json (auto-detect).
Single source of truth for the entire application.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# =========================
# Auto-detect config file
# =========================
CONFIG_DIR = Path(__file__).parent

# Priority: YAML first, then JSON
YAML_PATH = CONFIG_DIR / "saas_config.yaml"
JSON_PATH = CONFIG_DIR / "saas_config.json"


def load_config() -> Dict[str, Any]:
    """Load config from YAML or JSON — auto-detect."""

    # Try YAML first
    if YAML_PATH.exists():
        try:
            import yaml
            with open(YAML_PATH, "r") as f:
                config = yaml.safe_load(f)
            print(f"[Config] ✅ Loaded from {YAML_PATH.name}")
            return config
        except ImportError:
            print("[Config] ⚠️ YAML file found but pyyaml not installed. Trying JSON...")
        except Exception as e:
            print(f"[Config] ⚠️ Failed to load YAML: {e}. Trying JSON...")

    # Fall back to JSON
    if JSON_PATH.exists():
        with open(JSON_PATH, "r") as f:
            config = json.load(f)
        print(f"[Config] ✅ Loaded from {JSON_PATH.name}")
        return config

    raise FileNotFoundError(
        f"No config file found. Create either {YAML_PATH.name} or {JSON_PATH.name}"
    )


# Load config once at startup
FULL_CONFIG = load_config()

APP_CONFIG = FULL_CONFIG.get("app", {})
SERVER_CONFIG = FULL_CONFIG.get("server", {})
LLM_CONFIG = FULL_CONFIG.get("llm", {})
VECTOR_CONFIG = FULL_CONFIG.get("vector_store", {})
SAAS_CONFIG = FULL_CONFIG.get("saas_backend", {})


# =========================
# App Config
# =========================
def get_app_name() -> str:
    return APP_CONFIG.get("name", "Assistant")


def get_entity_field() -> str:
    return APP_CONFIG.get("entity_field", "entity")


def get_skip_intents() -> list:
    return APP_CONFIG.get("skip_intents", ["unclear", "general", "greeting"])


def get_intent_fields() -> Dict[str, Dict]:
    return APP_CONFIG.get("intent_fields", {})


# =========================
# Server Config
# =========================
def get_debug_mode() -> bool:
    return SERVER_CONFIG.get("debug", False)


def get_cors_origins() -> List[str]:
    return SERVER_CONFIG.get("cors_origins", ["http://localhost:5173"])


def get_allowed_upload_types() -> List[str]:
    return SERVER_CONFIG.get("allowed_upload_types", [".pdf", ".docx", ".txt"])


def get_default_category() -> str:
    return SERVER_CONFIG.get("default_category", "general")


# =========================
# LLM Config
# =========================
def get_llm_base_url() -> str:
    return LLM_CONFIG.get("base_url", "http://localhost:11434")


def get_chat_model() -> str:
    return LLM_CONFIG.get("chat_model", "llama3")


def get_embed_model() -> str:
    return LLM_CONFIG.get("embed_model", "nomic-embed-text")


def get_chat_url() -> str:
    base = get_llm_base_url()
    endpoint = LLM_CONFIG.get("chat_endpoint", "/api/chat")
    return f"{base}{endpoint}"


def get_embed_url() -> str:
    base = get_llm_base_url()
    endpoint = LLM_CONFIG.get("embed_endpoint", "/api/embed")
    return f"{base}{endpoint}"


def get_llm_timeout() -> int:
    return LLM_CONFIG.get("timeout", 120)


# =========================
# Vector Store Config
# =========================
def get_chroma_dir() -> Path:
    dir_name = VECTOR_CONFIG.get("persist_dir", "chroma_data")
    return Path(__file__).parent / dir_name


def get_collection_name() -> str:
    return VECTOR_CONFIG.get("collection_name", "knowledge_base")


def get_similarity_metric() -> str:
    return VECTOR_CONFIG.get("similarity_metric", "cosine")


def get_chunk_size() -> int:
    return VECTOR_CONFIG.get("chunk_size", 500)


def get_chunk_overlap() -> int:
    return VECTOR_CONFIG.get("chunk_overlap", 80)


def get_kb_path() -> Path:
    kb_file = VECTOR_CONFIG.get("kb_file", "kb_docs.json")
    return Path(__file__).parent / kb_file


def get_default_top_k() -> int:
    return VECTOR_CONFIG.get("default_top_k", 3)


# =========================
# System Prompt Builder
# =========================
def build_response_format() -> str:
    fields = get_intent_fields()
    lines = ['  "reply": "<short human-friendly response>"']

    for field_name, field_config in fields.items():
        desc = field_config.get("description", field_name)
        field_type = field_config.get("type", "string")

        if field_type == "number":
            lines.append(f'  "{field_name}": <{desc} or null>')
        else:
            lines.append(f'  "{field_name}": "<{desc} or null>"')

    return "{\n" + ",\n".join(lines) + "\n}"


def build_system_prompt(intent_block: str) -> str:
    template = APP_CONFIG.get("system_prompt", "You are an assistant. Respond in JSON.")
    response_format = build_response_format()

    prompt = template.replace("{app_name}", get_app_name())
    prompt = prompt.replace("{intent_block}", intent_block)
    prompt = prompt.replace("{response_format}", response_format)

    return prompt


# =========================
# Intent Parsing
# =========================
def extract_parsed_fields(parsed_dict: Dict) -> Dict[str, Any]:
    fields = get_intent_fields()
    result = {}

    for field_name, field_config in fields.items():
        default = field_config.get("default")
        value = parsed_dict.get(field_name, default)

        field_type = field_config.get("type", "string")
        if value is not None:
            try:
                if field_type == "number":
                    value = float(value)
                elif field_type == "string":
                    value = str(value)
            except (ValueError, TypeError):
                value = default

        result[field_name] = value

    return result


def is_actionable_intent(intent: str) -> bool:
    skip = get_skip_intents()
    return intent not in skip and intent is not None and intent != ""