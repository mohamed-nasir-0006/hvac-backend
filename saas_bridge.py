# saas_bridge.py
"""
SaaS Bridge
Reads saas_config.json and calls hvac-saas-backend APIs
based on parsed chat intents.
"""

import json
import requests
from typing import Optional, Dict, Any

# Load config
with open("saas_config.json", "r") as f:
    CONFIG = json.load(f)["saas_backend"]

BASE_URL = CONFIG["base_url"]
ENDPOINTS = CONFIG["endpoints"]
ZONE_MAP = CONFIG["zone_map"]


def resolve_zone_id(zone_name: str) -> Optional[str]:
    """Map zone name variations to zone_id."""
    normalized = zone_name.lower().strip()
    return ZONE_MAP.get(normalized)


def build_url(endpoint_name: str, **kwargs) -> str:
    """Build full URL from config endpoint."""
    endpoint = ENDPOINTS.get(endpoint_name)
    if not endpoint:
        raise ValueError(f"Unknown endpoint: {endpoint_name}")

    path = endpoint["path"].format(**kwargs)
    return f"{BASE_URL}{path}"


def call_saas_api(endpoint_name: str, path_params: Dict = None, body: Dict = None) -> Dict[str, Any]:
    """Call a SaaS backend API using config."""
    endpoint = ENDPOINTS.get(endpoint_name)
    if not endpoint:
        return {"error": f"Unknown endpoint: {endpoint_name}"}

    method = endpoint["method"]
    url = build_url(endpoint_name, **(path_params or {}))

    try:
        if method == "GET":
            res = requests.get(url, timeout=10)
        elif method == "POST":
            res = requests.post(url, json=body, timeout=10)
        else:
            return {"error": f"Unsupported method: {method}"}

        res.raise_for_status()
        return {"success": True, "data": res.json()}

    except requests.exceptions.ConnectionError:
        return {"error": "SaaS backend is not reachable"}
    except requests.exceptions.Timeout:
        return {"error": "SaaS backend timed out"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"API error: {e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def execute_intent(intent: Dict) -> Dict[str, Any]:
    """
    Execute a parsed chat intent by calling the appropriate SaaS API.
    Returns the result to be sent back to the chat user.
    """
    action = intent.get("intent")
    zone_name = intent.get("zone", "")
    value = intent.get("value")

    print(f"[Bridge] 🎯 Executing intent: {action} | zone: {zone_name} | value: {value}")

    # --- SET TEMPERATURE ---
    if action == "set_temperature":
        zone_id = resolve_zone_id(zone_name)
        if not zone_id:
            return {"error": f"Unknown zone: {zone_name}", "reply": f"Sorry, I don't recognize the zone '{zone_name}'."}

        result = call_saas_api(
            "set_temperature",
            path_params={"zone_id": zone_id},
            body={"temperature": value, "changed_by": "chat"},
        )

        if result.get("success"):
            zone_data = result["data"].get("zone", {})
            return {
                "success": True,
                "reply": f"✅ {zone_data.get('name', zone_name)} temperature set to {value}°C",
                "zone": zone_data,
            }
        return {"error": result.get("error"), "reply": f"❌ Failed to set temperature: {result.get('error')}"}

    # --- SET MODE ---
    elif action == "set_mode":
        zone_id = resolve_zone_id(zone_name)
        if not zone_id:
            return {"error": f"Unknown zone: {zone_name}", "reply": f"Sorry, I don't recognize the zone '{zone_name}'."}

        mode = intent.get("mode") or intent.get("value", "")
        result = call_saas_api(
            "set_mode",
            path_params={"zone_id": zone_id},
            body={"mode": mode.lower(), "changed_by": "chat"},
        )

        if result.get("success"):
            zone_data = result["data"].get("zone", {})
            return {
                "success": True,
                "reply": f"✅ {zone_data.get('name', zone_name)} mode set to {mode}",
                "zone": zone_data,
            }
        return {"error": result.get("error"), "reply": f"❌ Failed to set mode: {result.get('error')}"}

    # --- TURN ON ---
    elif action == "turn_on":
        zone_id = resolve_zone_id(zone_name)
        if not zone_id:
            return {"error": f"Unknown zone: {zone_name}", "reply": f"Sorry, I don't recognize the zone '{zone_name}'."}

        result = call_saas_api(
            "set_status",
            path_params={"zone_id": zone_id},
            body={"status": "on", "changed_by": "chat"},
        )

        if result.get("success"):
            zone_data = result["data"].get("zone", {})
            return {
                "success": True,
                "reply": f"✅ {zone_data.get('name', zone_name)} turned ON",
                "zone": zone_data,
            }
        return {"error": result.get("error"), "reply": f"❌ Failed to turn on: {result.get('error')}"}

    # --- TURN OFF ---
    elif action == "turn_off":
        zone_id = resolve_zone_id(zone_name)
        if not zone_id:
            return {"error": f"Unknown zone: {zone_name}", "reply": f"Sorry, I don't recognize the zone '{zone_name}'."}

        result = call_saas_api(
            "set_status",
            path_params={"zone_id": zone_id},
            body={"status": "off", "changed_by": "chat"},
        )

        if result.get("success"):
            zone_data = result["data"].get("zone", {})
            return {
                "success": True,
                "reply": f"✅ {zone_data.get('name', zone_name)} turned OFF",
                "zone": zone_data,
            }
        return {"error": result.get("error"), "reply": f"❌ Failed to turn off: {result.get('error')}"}

    # --- GET TEMPERATURE / STATUS ---
    elif action in ["get_temperature", "get_status"]:
        zone_id = resolve_zone_id(zone_name)
        if not zone_id:
            return {"error": f"Unknown zone: {zone_name}", "reply": f"Sorry, I don't recognize the zone '{zone_name}'."}

        result = call_saas_api("get_zone", path_params={"zone_id": zone_id})

        if result.get("success"):
            zone = result["data"]
            return {
                "success": True,
                "reply": f"🌡️ {zone['name']}: {zone['temperature']}°C | Mode: {zone['mode']} | Status: {zone['status']} | Humidity: {zone['humidity']}%",
                "zone": zone,
            }
        return {"error": result.get("error"), "reply": f"❌ Failed to get zone info: {result.get('error')}"}

    # --- GET ALL ZONES ---
    elif action == "get_all_zones":
        result = call_saas_api("get_all_zones")

        if result.get("success"):
            zones = result["data"]
            lines = [f"🌡️ {z['name']}: {z['temperature']}°C | {z['mode']} | {z['status']}" for z in zones]
            return {
                "success": True,
                "reply": "📊 All Zones:\n" + "\n".join(lines),
                "zones": zones,
            }
        return {"error": result.get("error"), "reply": f"❌ Failed to get zones: {result.get('error')}"}

    # --- UNKNOWN INTENT ---
    else:
        return {"reply": f"I understood your request but I don't know how to handle '{action}' yet."}