"""
Prompts for the vision model.
"""

def build_vision_prompt(previous_summary: str = "", previous_objects: list = None,
                     previous_actions: list = None, frame_count: int = 0) -> str:
    """
    Build a detailed vision prompt with previous context.
    """
    if previous_objects is None:
        previous_objects = []
    if previous_actions is None:
        previous_actions = []

    context_parts = []

    if previous_summary:
        context_parts.append(f"PREVIOUS SCENE: {previous_summary}")
    if previous_objects:
        context_parts.append(f"PREVIOUS OBJECTS: {', '.join(previous_objects)}")
    if previous_actions:
        context_parts.append(f"PREVIOUS ACTIONS: {', '.join(previous_actions)}")

    context_block = ""
    if context_parts:
        context_block = "## PREVIOUS CONTEXT\n" + "\n".join(context_parts) + "\n\n"

    prompt_template = """You are analyzing frame #{} from a live webcam feed.

{}## TASK
Provide a detailed, memory-usable analysis of this frame. Be specific, factual, and descriptive.

## OUTPUT FORMAT (Respond in this exact JSON format):
{{
  "summary": "2-4 sentence detailed description of what's happening in the scene",
  "objects": ["specific item 1", "item 2", "item 3"],
  "actions": ["what people are doing", "any movement or activity"],
  "changes": "what changed from previous observation, or 'none'",
  "scene_status": "active (movement/people), idle (empty/still), or unknown",
  "confidence": 0.0-1.0,
  "important_notes": "any notable details worth remembering"
}}

## REQUIREMENTS
- summary: Be detailed enough this can serve as future context memory. Include positions, colors, approximate locations.
- objects: List specific items, max 10. Include what they're doing if applicable.
- actions: What are people doing? Posture? Interactions? Max 5.
- changes: What is different from previous observation?
- confidence: Rate your confidence 0.0-1.0 based on image quality and clarity.
- Only output valid JSON, no explanations or markdown."""

    return prompt_template.format(frame_count, context_block)

def build_agent_summary_prompt(recent_context: list) -> str:
    """
    Build a prompt for the agent to summarize recent observations.
    """
    context_lines = []
    for i, ctx in enumerate(recent_context[-5:]):
        context_lines.append(
            f"{i+1}. [{ctx.get('timestamp', '')}] {ctx.get('summary', '')}"
        )

    context_text = "\n".join(context_lines)

    return f"""Analyze the following recent observations and create a coherent summary.

## RECENT OBSERVATIONS
{context_text}

## TASK
Create a brief agent summary (2-3 sentences) that represents what's been happening.
Focus on:
- Main activities observed
- Key objects/people
- Overall pattern or trend

## OUTPUT FORMAT
{{
  "agent_summary": "2-3 sentence summary of recent activity",
  "key_observations": ["list of key things to remember"],
  "scene_status": "active/idle/transitioning"
}}"""

RETRY_PROMPT = """The previous response was not valid JSON. Please respond with ONLY valid JSON in this format:

{"summary": "...", "objects": [], "actions": [], "changes": "...", "scene_status": "...", "confidence": 0.0, "important_notes": "..."}

Do not include any other text."""