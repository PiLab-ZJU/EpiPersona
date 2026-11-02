"""Prompt text used to build history embeddings."""

SYSTEM_MESSAGE = [{"role": "system", "content": "You are a helpful assistant."}]


def stable_preference_prompt(context, chosen, rejected) -> str:
    # f-string, not str.format: dialogue text may contain braces.
    return f"""
Your task: Identify the stable preference revealed by the chosen answer compared to the rejected answer. 
If no stable preference is implied, output an empty string.

**Definition of Stable Preference**:  
A stable preference is a user preference that reflects enduring aspects of the user’s persona, values, or identity. 
It is transferable across different dialogue contexts and unlikely to change due to short-term situations or local context.

Dialogue context: {context}

Chosen answer: {chosen}

Rejected answer: {rejected}

Respond strictly in the following JSON format and nothing else:
```json
{{"stable_preferences": "<preference>"}}
```"""
