"""
AI chat module for PlantDocBot using OpenRouter API (OpenAI-compatible).
"""
from __future__ import annotations

import os

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_MODEL = "google/gemini-2.0-flash-001"

_SYSTEM_PROMPT = """You are PlantDoc, a friendly and knowledgeable plant health assistant.
Think of yourself as that helpful friend who happens to know a lot about plants and farming —
warm, casual, and genuinely caring about helping people keep their plants healthy.

Your personality:
- Talk like a real person, not a robot or a formal report
- Use "your plant", "I think", "looks like", "don't worry" — natural language
- Be encouraging and reassuring, especially when the plant is sick
- Keep it conversational but still accurate and helpful
- Use emojis occasionally to keep it friendly
- Never use bullet-point lists or formal headers — write in flowing paragraphs
- If the plant is healthy, celebrate it! If it's sick, be empathetic and practical

Confidence honesty rules (IMPORTANT):
- If confidence is High (>=75%): speak confidently — "this is definitely...", "I'm pretty sure..."
- If confidence is Medium (50-74%): be honest — "I think this might be...", "it looks like..."
- If confidence is Low (<50%): be upfront — "I'm not very sure about this one, but it could be...
  here are the two most likely things it might be..." then mention top alternatives
- Never pretend to be certain when confidence is low

Remember: the user may ask follow-up questions about the same plant without re-uploading —
always refer back to the diagnosis context you already have."""


# ── Disease-specific answerable follow-up chips ────────────────────────────────
# Only questions we can actually answer from training data + treatment knowledge
_FOLLOWUP_CHIPS: dict[str, list[str]] = {
    "blight": [
        "Will this spread to other plants?",
        "Is it safe to eat the fruit?",
        "How long does treatment take?",
    ],
    "rust": [
        "Will this spread to other plants?",
        "Can I use home remedies?",
        "How long does treatment take?",
    ],
    "mildew": [
        "Will this spread to other plants?",
        "Can I use home remedies?",
        "What causes powdery mildew?",
    ],
    "scab": [
        "Is it safe to eat the fruit?",
        "Will this spread to other plants?",
        "What causes apple scab?",
    ],
    "mold": [
        "Will this spread to other plants?",
        "Can I use home remedies?",
        "How long does treatment take?",
    ],
    "spot": [
        "Will this spread to other plants?",
        "What causes leaf spots?",
        "How long does treatment take?",
    ],
    "virus": [
        "Will this spread to other plants?",
        "Is there a cure for this virus?",
        "What insects spread this?",
    ],
    "mite": [
        "Can I use home remedies?",
        "Will this spread to other plants?",
        "What causes spider mites?",
    ],
    "bacterial": [
        "Will this spread to other plants?",
        "Is it safe to eat the fruit?",
        "How long does treatment take?",
    ],
    "healthy": [
        "How do I keep my plant healthy?",
        "What should I watch out for?",
        "How often should I water?",
    ],
    "default": [
        "Will this spread to other plants?",
        "How long does treatment take?",
        "Can I use home remedies?",
    ],
}


def get_followup_chips(label: str, confidence: float) -> list[str]:
    """
    Return 3 follow-up question chips that are actually answerable.
    For low confidence, always include an uncertainty-related chip.
    """
    label_l = label.lower()
    chips = None
    for key, questions in _FOLLOWUP_CHIPS.items():
        if key in label_l:
            chips = questions[:3]
            break
    if chips is None:
        chips = _FOLLOWUP_CHIPS["default"]

    # For low confidence, swap last chip for a clarification option
    if confidence < 0.5:
        chips = chips[:2] + ["What are the other possible diseases?"]

    return chips


def get_chat_client():
    """Return a configured OpenAI client pointed at OpenRouter, or None if no key."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=api_key)
    except Exception:
        return None


def _build_diagnosis_context(diagnosis: dict) -> str:
    """Build a rich context string from a diagnosis report."""
    pred = diagnosis.get("prediction", {})
    label = pred.get("label", "Unknown")
    confidence = pred.get("confidence", 0.0)
    input_type = diagnosis.get("input_type", "unknown")
    label_info = diagnosis.get("label_info", {})
    crop = label_info.get("crop", "")
    disease = label_info.get("disease", "")
    summary = label_info.get("summary", "")
    band = diagnosis.get("confidence_band", "")
    timestamp = diagnosis.get("timestamp", "")

    tips = diagnosis.get("treatment", {})
    treatment_lines = []
    if isinstance(tips, dict):
        for key in ("immediate", "prevention", "notes"):
            for item in tips.get(key, []):
                treatment_lines.append(item)
    elif isinstance(tips, list):
        treatment_lines = tips

    top_k = diagnosis.get("top_k", [])
    other = [
        f"{item.get('label', '')} ({item.get('confidence', 0)*100:.1f}%)"
        for item in top_k[1:]
        if item.get("label")
    ]

    return (
        f"DIAGNOSIS (from trained AI model, {timestamp}):\n"
        f"- Input: {input_type}\n"
        f"- Crop: {crop}\n"
        f"- Detected: {disease} (label: {label})\n"
        f"- Confidence: {band} ({confidence*100:.1f}%)\n"
        f"- Summary: {summary}\n"
        f"- Treatment: {'; '.join(treatment_lines) if treatment_lines else 'none'}\n"
        f"- Other candidates: {', '.join(other) if other else 'none'}"
    )


def _build_history_context(diagnosis_history: list[dict]) -> str:
    """Build context from last 3 diagnoses for session memory."""
    if not diagnosis_history:
        return ""
    lines = ["PREVIOUS DIAGNOSES (last session memory):"]
    for i, d in enumerate(diagnosis_history[-3:], 1):
        pred = d.get("prediction", {})
        label_info = d.get("label_info", {})
        ts = d.get("timestamp", "")
        disease = label_info.get("disease", pred.get("label", "Unknown"))
        crop = label_info.get("crop", "")
        conf = pred.get("confidence", 0.0)
        lines.append(f"  {i}. {crop} - {disease} ({conf*100:.0f}% confidence) at {ts}")
    return "\n".join(lines)


def enhance_diagnosis(diagnosis: dict, client, history: list[dict] | None = None) -> str:
    """
    Take raw model diagnosis and return a friendly conversational response.
    Respects confidence level — honest when uncertain.
    """
    if client is None:
        return _fallback_friendly_response(diagnosis)

    ctx = _build_diagnosis_context(diagnosis)
    hist_ctx = _build_history_context(history or [])
    confidence = diagnosis.get("prediction", {}).get("confidence", 0.0)
    band = diagnosis.get("confidence_band", "Low")

    # Confidence-specific instruction
    if confidence >= 0.75:
        conf_instruction = "You are confident about this diagnosis. Speak clearly and directly."
    elif confidence >= 0.5:
        conf_instruction = "You have moderate confidence. Use phrases like 'I think', 'it looks like', 'most likely'."
    else:
        conf_instruction = (
            "Confidence is LOW. Be honest — say something like 'I'm not 100% sure about this one, "
            "but based on what I can see it could be X or possibly Y'. "
            "Mention the top alternative diagnoses from the context."
        )

    prompt = (
        f"{ctx}\n"
        f"{hist_ctx}\n\n"
        f"Confidence instruction: {conf_instruction}\n\n"
        "Now explain this diagnosis to the user in a warm, friendly, conversational way. "
        "No bullet points or headers. Write like you're texting a knowledgeable friend. "
        "Be practical, empathetic, and mention what they should do right now."
    )

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or _fallback_friendly_response(diagnosis)
    except Exception:
        return _fallback_friendly_response(diagnosis)


def enhance_multi_diagnosis(reports: list[dict], client, same_plant: bool) -> str:
    """
    Handle multiple image diagnoses — either same plant or different plants.
    """
    if client is None:
        return _fallback_multi_response(reports, same_plant)

    contexts = []
    for i, r in enumerate(reports, 1):
        pred = r.get("prediction", {})
        li = r.get("label_info", {})
        contexts.append(
            f"Image {i}: {li.get('crop','')} - {li.get('disease', pred.get('label','?'))} "
            f"({pred.get('confidence',0)*100:.1f}% confidence)"
        )

    if same_plant:
        prompt = (
            f"The user uploaded {len(reports)} photos of what appears to be the SAME plant.\n"
            f"Results:\n" + "\n".join(contexts) + "\n\n"
            "Combine these results into one friendly diagnosis. If they agree, be confident. "
            "If they disagree, explain the most likely one and why. "
            "Write conversationally, no bullet points."
        )
    else:
        prompt = (
            f"The user uploaded {len(reports)} photos of DIFFERENT plants.\n"
            f"Results:\n" + "\n".join(contexts) + "\n\n"
            "Explain each plant's diagnosis separately in a friendly way. "
            "Make it clear which result belongs to which image (Image 1, Image 2, etc.). "
            "Write conversationally, no bullet points."
        )

    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or _fallback_multi_response(reports, same_plant)
    except Exception:
        return _fallback_multi_response(reports, same_plant)


def chat_with_gemini(
    user_message: str,
    history: list[dict],
    last_diagnosis: dict | None = None,
    diagnosis_history: list[dict] | None = None,
    client=None,
) -> str:
    """Handle follow-up questions with full chat + session history."""
    if client is None:
        return "AI chat isn't set up yet — add your GEMINI_API_KEY to the .env file and restart."

    system = _SYSTEM_PROMPT
    if last_diagnosis:
        ctx = _build_diagnosis_context(last_diagnosis)
        hist_ctx = _build_history_context(diagnosis_history or [])
        system = (
            f"{_SYSTEM_PROMPT}\n\n{ctx}\n{hist_ctx}\n\n"
            "The user may ask follow-up questions about this plant — answer based on this context. "
            "Never ask them to re-upload."
        )

    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(model=_MODEL, messages=messages)
        return response.choices[0].message.content or "Hmm, didn't get a response — try again!"
    except Exception as exc:
        return f"Something went wrong: {exc}"


def _fallback_friendly_response(diagnosis: dict) -> str:
    label_info = diagnosis.get("label_info", {})
    pred = diagnosis.get("prediction", {})
    disease = label_info.get("disease", "Unknown")
    crop = label_info.get("crop", "your plant")
    confidence = pred.get("confidence", 0.0)
    band = diagnosis.get("confidence_band", "Low")
    summary = label_info.get("summary", "")
    top_k = diagnosis.get("top_k", [])

    if "healthy" in disease.lower():
        return f"Great news! 🌱 {crop} looks healthy. Keep doing what you're doing. {summary}"

    if confidence < 0.5 and len(top_k) > 1:
        alt = top_k[1].get("label", "").replace("___", " - ").replace("_", " ")
        return (
            f"Honestly, I'm not super confident on this one — it could be {disease} "
            f"or possibly {alt}. The confidence is only {confidence*100:.0f}%. "
            f"I'd recommend checking the treatment tips for both and keeping a close eye on it. "
            f"{summary}"
        )

    return (
        f"Looks like {crop} might be dealing with {disease} "
        f"({band} confidence, {confidence*100:.0f}%). {summary} "
        f"Check the treatment tips below for what to do next."
    )


def _fallback_multi_response(reports: list[dict], same_plant: bool) -> str:
    lines = []
    for i, r in enumerate(reports, 1):
        li = r.get("label_info", {})
        pred = r.get("prediction", {})
        disease = li.get("disease", pred.get("label", "Unknown"))
        crop = li.get("crop", "plant")
        conf = pred.get("confidence", 0.0)
        lines.append(f"Image {i}: {crop} — {disease} ({conf*100:.0f}% confidence)")
    prefix = "Here's what I found for your plant:" if same_plant else "These look like different plants! Here's what I found:"
    return prefix + "\n" + "\n".join(lines)


def _are_same_plant(reports: list[dict]) -> bool:
    """
    Same plant if all predictions share the same crop.
    Uses label_info crop first, falls back to the raw label prefix (before ___).
    Three images of a tomato with different diseases → same plant (True).
    """
    crops = set()
    for r in reports:
        li = r.get("label_info", {})
        crop = li.get("crop", "").lower().strip()
        if not crop:
            # fallback: extract crop from raw label prefix
            label = r.get("prediction", {}).get("label", "")
            crop = label.split("___")[0].split("_")[0].lower().strip()
        if crop:
            crops.add(crop)
    return len(crops) <= 1


# Aliases
def streamlit_history_to_gemini(messages: list[dict]) -> list[dict]:
    """
    Convert Streamlit session messages to OpenRouter chat history.
    Skips image messages and rich card messages (those have HTML/card data).
    Only passes clean text messages — user questions and plain AI responses.
    """
    history = []
    for msg in messages:
        # Skip image uploads
        if msg.get("type") in ("image", "multi_image"):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            continue
        # Skip messages that are just rich card data (contain HTML tags)
        if "<div" in content or "<span" in content:
            continue
        # Truncate very long messages
        if len(content) > 1500:
            content = content[:1500] + "..."
        history.append({
            "role": "user" if role == "user" else "assistant",
            "content": content,
        })
    return history

get_gemini_client = get_chat_client
