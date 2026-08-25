"""
llm_engine.py — Asset generators + Agentic Script Generator (Architect + Developer)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List, Optional, Dict, Any

from openai import OpenAI

NVIDIA_API_KEY =(
    "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"
)

_client: Optional[OpenAI] = None


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------
def prepare_model() -> bool:
    global _client
    if _client is not None:
        return True
    if not NVIDIA_API_KEY:
        print("[LLM] NVIDIA_API_KEY is not set.")
        return False
    _client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )
    print("NVIDIA Nemotron client ready.")
    return True


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _collect_stream(completion) -> tuple[str, str]:
    full_text = ""
    reasoning_text = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        if delta.content is not None:
            full_text += delta.content
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning is not None:
            reasoning_text += reasoning
    return full_text, reasoning_text


def _extract_flat_list(text: str) -> List[str]:
    if not text:
        return []
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(item).strip() for item in data if item]
        except json.JSONDecodeError:
            pass
    items = re.findall(r'"([^"]+)"', text)
    if items:
        return [i.strip() for i in items if i.strip()]
    parts = [p.strip().strip('"\'') for p in text.split(',') if p.strip()]
    return parts


def _call_llm_raw(system_prompt: str) -> str:
    """Non‑streaming LLM call for script generation."""
    if _client is None:
        prepare_model()
        if _client is None:
            return ""
    try:
        completion = _client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.7,
            top_p=0.95,
            max_tokens=4096,      # enough for a single script
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=False,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return ""


def _call_llm_stream(system_prompt: str) -> List[str]:
    """Streaming LLM call for asset terms."""
    if _client is None:
        prepare_model()
        if _client is None:
            return []
    try:
        completion = _client.chat.completions.create(
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stream=True,
        )
        full_text, _ = _collect_stream(completion)
        return _extract_flat_list(full_text)
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# Asset generators (unchanged)
# ---------------------------------------------------------------------------
async def generate_3d(prompt: str, qty: int) -> List[str]:
    prepare_model()
    system = (
        f"Generate exactly {qty} specific 3D model search terms for a game environment based on the prompt: '{prompt}'. "
        "Return ONLY a JSON list of strings, e.g. ['modern_house', 'street_lamp']. "
        "CRITICAL: Do NOT add suffixes like '_3d', '_model', '_asset', '_interior', or '_pack' to the terms. "
        "Use clean, simple search terms (e.g., 'wooden_bridge', 'cash_register')."
    )
    return await asyncio.to_thread(_call_llm_stream, system)


async def generate_audio(prompt: str, qty: int) -> List[str]:
    prepare_model()
    system = (
        f"Generate exactly {qty} specific sound effect search terms for a game based on the prompt: '{prompt}'. "
        "Return ONLY a JSON list of strings, e.g. ['door_open', 'car_engine']. "
        "CRITICAL: Do NOT add suffixes like '_sound', '_audio', '_sfx', or '_effect' to the terms. "
        "Use clean, simple search terms (e.g., 'door_open', 'car_engine')."
    )
    return await asyncio.to_thread(_call_llm_stream, system)


async def generate_textures(prompt: str, qty: int) -> List[str]:
    prepare_model()
    system = (
        f"Generate exactly {qty} specific texture/material search terms for a game based on the prompt: '{prompt}'. "
        "Return ONLY a JSON list of strings, e.g. ['concrete_road', 'grass_tile']. "
        "CRITICAL: Do NOT add suffixes like '_texture', '_material', '_map', or '_tile' to the terms. "
        "Use clean, simple search terms (e.g., 'concrete_road', 'grass_tile')."
    )
    return await asyncio.to_thread(_call_llm_stream, system)


async def generate_ui(prompt: str, qty: int) -> List[str]:
    prepare_model()
    system = (
        f"You are a game UI designer. Generate exactly {qty} specific UI component terms for a game based on the prompt: '{prompt}'. "
        "Use the game-icons.net naming convention. Use single words or hyphenated terms (e.g., 'fishing-pole', 'health-potion', 'wooden-shield', 'coin', 'sword', 'fish'). "
        "Do NOT use underscores. "
        "Do NOT generate abstract concepts like 'night_city' or 'modern_architecture'. "
        "Only output concrete game UI elements. "
        "Return ONLY a JSON list of strings."
    )
    return await asyncio.to_thread(_call_llm_stream, system)


# ---------------------------------------------------------------------------
# Agentic Script Generator – Architect Step
# ---------------------------------------------------------------------------
async def generate_script_architecture(prompt: str, engine: str) -> List[str]:
    """
    Step 1: Architect plans the file structure for a complete MVP.
    Returns a list of file paths (e.g., ["Core/GameManager.cs", "Player/PlayerController.cs"]).
    """
    prepare_model()
    system = (
        f"You are an expert game development architect. Analyze this game concept: '{prompt}' for {engine}. "
        "Decide how many scripts are needed for a complete MVP (15 to 30 scripts). "
        "Output STRICT JSON containing a list of file paths. "
        "Format: {\"files\": [\"Core/GameManager.cs\", \"Player/PlayerController.cs\", \"UI/UIManager.cs\"]}. "
        "Do not output code, ONLY output the JSON list of file paths."
    )
    raw = await asyncio.to_thread(_call_llm_raw, system)
    if not raw:
        print("[Architect] Empty response.")
        return []
    # Extract JSON
    match = re.search(r'\{.*"files".*\}', raw, re.DOTALL)
    if not match:
        print(f"[Architect] No JSON found: {raw[:200]}...")
        return []
    try:
        data = json.loads(match.group(0))
        files = data.get("files", [])
        if isinstance(files, list) and all(isinstance(f, str) for f in files):
            return files
        else:
            print("[Architect] 'files' is not a list of strings.")
            return []
    except json.JSONDecodeError as e:
        print(f"[Architect] JSON parse error: {e}")
        return []


# ---------------------------------------------------------------------------
# Agentic Script Generator – Developer Step
# ---------------------------------------------------------------------------
async def generate_single_script(prompt: str, engine: str, file_path: str) -> str:
    """
    Step 2: Developer writes the complete code for a single file.
    Returns raw code as a string.
    """
    prepare_model()
    system = (
        f"You are an expert game developer. We are building a game about: '{prompt}' using {engine}. "
        f"Write the complete, clean, commented code for the file: '{file_path}'. "
        "Output ONLY raw code. Do not use markdown formatting (no ```csharp). "
        "Do not add any explanations before or after the code."
    )
    raw = await asyncio.to_thread(_call_llm_raw, system)
    if not raw:
        print(f"[Developer] Empty response for {file_path}.")
        return ""
    # Remove any markdown code fences that might appear
    raw = re.sub(r'^```.*$', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'^```$', '', raw, flags=re.MULTILINE)
    return raw.strip()
