# saas_bridge.py
"""
Generic SaaS Bridge (v3 — Fully Config-Driven)
Zero application-specific code.
Everything driven by saas_config.json.
"""

import json
import re
import requests
from typing import Optional, Dict, Any
from config_loader import SAAS_CONFIG, get_entity_field

BASE_URL = SAAS_CONFIG["base_url"]
ENDPOINTS = SAAS_CONFIG["endpoints"]
ENTITY_MAP = SAAS_CONFIG.get("entity_map", {})
INTENT_ALIASES = SAAS_CONFIG.get("intent_aliases", {})


def resolve_entity(name: str) -> Optional[str]:
    """Map entity name variations to entity_id using config."""
    if not name:
        return None
    normalized = name.lower().strip()
    return ENTITY_MAP.get(normalized)


def resolve_variable(value: Any, intent_data: Dict) -> Any:
    """Resolve $variable placeholders to actual values."""
    if not isinstance(value, str):
        return value

    if value == "$entity":
        entity_field = get_entity_field()
        entity_name = intent_data.get(entity_field) or intent_data.get("entity", "")
        return resolve_entity(entity_name)

    if value.startswith("$"):
        field_name = value[1:]
        resolved = intent_data.get(field_name)
        if isinstance(resolved, str):
            try:
                resolved = float(resolved)
                if resolved == int(resolved):
                    resolved = int(resolved)
            except (ValueError, TypeError):
                pass
        return resolved

    return value


def resolve_body(body_template: Optional[Dict], intent_data: Dict) -> Optional[Dict]:
    """Resolve all variables in request body."""
    if not body_template:
        return None
    resolved = {}
    for key, value in body_template.items():
        resolved_value = resolve_variable(value, intent_data)
        if resolved_value is not None:
            resolved[key] = resolved_value
    return resolved


def resolve_path_params(params_template: Dict, intent_data: Dict) -> Dict:
    """Resolve all variables in path parameters."""
    resolved = {}
    for key, value in params_template.items():
        resolved_value = resolve_variable(value, intent_data)
        if resolved_value is not None:
            resolved[key] = resolved_value
    return resolved


def format_message(template: str, intent_data: Dict, api_response: Any = None) -> str:
    """
    Format message template with actual values.
    Dynamically resolves ANY {field} from intent_data
    and ANY {data.field} from API response.
    """
    result = template

    # Replace {entity_name} with the entity field value
    entity_field = get_entity_field()
    entity_name = intent_data.get(entity_field) or intent_data.get("entity", "unknown")
    result = result.replace("{entity_name}", str(entity_name).title())

    # Replace ANY {field} from intent_data — fully dynamic
    for key, value in intent_data.items():
        placeholder = f"{{{key}}}"
        if placeholder in result and value is not None:
            result = result.replace(placeholder, str(value))

    # Replace {data.field} from API response
    if api_response and isinstance(api_response, dict):
        data_fields = re.findall(r'\{data\.(\w+)\}', result)
        for field in data_fields:
            field_value = api_response.get(field, "N/A")
            result = result.replace(f"{{data.{field}}}", str(field_value))

    return result


def call_api(method: str, url: str, body: Dict = None) -> Dict[str, Any]:
    """Make HTTP request."""
    print(f"[Bridge] 📡 {method} {url}")
    if body:
        print(f"[Bridge] 📦 Body: {body}")

    try:
        if method == "GET":
            res = requests.get(url, timeout=10)
        elif method == "POST":
            res = requests.post(url, json=body, timeout=10)
        elif method == "PUT":
            res = requests.put(url, json=body, timeout=10)
        elif method == "PATCH":
            res = requests.patch(url, json=body, timeout=10)
        elif method == "DELETE":
            res = requests.delete(url, timeout=10)
        else:
            return {"error": f"Unsupported method: {method}"}

        res.raise_for_status()
        return {"success": True, "data": res.json()}

    except requests.exceptions.ConnectionError:
        return {"error": "Backend service is not reachable"}
    except requests.exceptions.Timeout:
        return {"error": "Backend service timed out"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"API error: {e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def execute_intent(intent_data: Dict) -> Dict[str, Any]:
    """
    Execute any intent using unified endpoint config.
    Zero application-specific code.
    """
    action = intent_data.get("intent", "")

    # Check aliases
    resolved_action = INTENT_ALIASES.get(action, action)

    print(f"[Bridge] 🎯 Intent: {action} → Endpoint: {resolved_action}")

    # Look up endpoint config
    endpoint_config = ENDPOINTS.get(resolved_action)
    if not endpoint_config:
        return {"reply": f"I understood your request but '{action}' is not configured yet."}

    # Resolve entity
    entity_field = get_entity_field()
    entity_name = intent_data.get(entity_field) or intent_data.get("entity", "")
    path_params_template = endpoint_config.get("path_params", {})

    if entity_name and "$entity" in str(path_params_template.values()):
        entity_id = resolve_entity(entity_name)
        if not entity_id:
            return {
                "error": f"Unknown entity: {entity_name}",
                "reply": f"Sorry, I don't recognize '{entity_name}'."
            }

    # Resolve path params and body
    path_params = resolve_path_params(path_params_template, intent_data)
    body = resolve_body(endpoint_config.get("body"), intent_data)

    # Build URL
    path = endpoint_config["path"].format(**path_params)
    url = f"{BASE_URL}{path}"

    # Call API
    result = call_api(endpoint_config["method"], url, body)

    # Format response
    if result.get("success"):
        api_data = result.get("data", {})

        # Handle list responses
        if isinstance(api_data, list):
            lines = []
            for item in api_data:
                line = format_message(endpoint_config["success_message"], intent_data, item)
                lines.append(line)
            reply = "\n".join(lines) if lines else endpoint_config["success_message"]
            return {"success": True, "reply": reply}

        # Handle object responses — find nested entity data
        nested_data = api_data
        if isinstance(api_data, dict):
            # Try common nested keys dynamically
            for key in api_data:
                if isinstance(api_data[key], dict):
                    nested_data = api_data[key]
                    break

        reply = format_message(endpoint_config["success_message"], intent_data, nested_data)
        return {"success": True, "reply": reply}

    else:
        error_msg = result.get("error", "Unknown error")
        reply = f"{endpoint_config['error_message']}: {error_msg}"
        return {"error": error_msg, "reply": reply}