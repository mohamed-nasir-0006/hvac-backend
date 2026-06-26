# intents.py
# All known intents — driven by application domain

KNOWN_INTENTS = [
    {
        "intent": "set_temperature",
        "description": "User wants to set a target temperature for a zone",
        "fields": ["zone", "value", "unit"],
    },
    {
        "intent": "get_temperature",
        "description": "User wants to know the current temperature of a zone",
        "fields": ["zone"],
    },
    {
        "intent": "set_mode",
        "description": "User wants to change mode (cooling, heating, auto, off)",
        "fields": ["zone", "mode"],
    },
    {
        "intent": "get_status",
        "description": "User wants to know the current system status",
        "fields": ["zone"],
    },
    {
        "intent": "turn_on",
        "description": "User wants to turn on a zone",
        "fields": ["zone"],
    },
    {
        "intent": "turn_off",
        "description": "User wants to turn off a zone",
        "fields": ["zone"],
    },
    {
        "intent": "get_all_zones",
        "description": "User wants to see all zones status",
        "fields": [],
    },
    {
        "intent": "general_question",
        "description": "User is asking a general knowledge question (use RAG context)",
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