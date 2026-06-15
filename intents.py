# intents.py
# All known HVAC intents the system can handle

KNOWN_INTENTS = [
    {
        "intent": "set_temperature",
        "description": "User wants to set a target temperature for a zone",
        "fields": ["zone", "value", "unit", "schedule"],
    },
    {
        "intent": "get_temperature",
        "description": "User wants to know the current temperature of a zone",
        "fields": ["zone"],
    },
    {
        "intent": "set_mode",
        "description": "User wants to change HVAC mode (cooling, heating, auto, off, setback)",
        "fields": ["zone", "mode"],
    },
    {
        "intent": "get_status",
        "description": "User wants to know the current HVAC system status",
        "fields": ["zone"],
    },
    {
        "intent": "set_schedule",
        "description": "User wants to create or modify a schedule",
        "fields": ["zone", "schedule", "value", "unit", "mode"],
    },
    {
        "intent": "general_question",
        "description": "User is asking a general HVAC knowledge question (use RAG context)",
        "fields": [],
    },
    {
        "intent": "unclear",
        "description": "Intent is not clear, ask for clarification",
        "fields": [],
    },
]

def get_intent_prompt_block() -> str:
    """Format intents into a string for the system prompt."""
    lines = ["KNOWN INTENTS:"]
    for it in KNOWN_INTENTS:
        fields = ", ".join(it["fields"]) if it["fields"] else "(none)"
        lines.append(f'  - {it["intent"]}: {it["description"]} | fields: {fields}')
    return "\n".join(lines)