"""Shared response profile controls for CLI and web workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileSpec:
    key: str
    label: str
    instruction: str


PROFILE_SPECS: dict[str, ProfileSpec] = {
    "default": ProfileSpec(
        key="default",
        label="Default",
        instruction="Produce clear, high-quality responses aligned to the user request.",
    ),
    "twitter_short": ProfileSpec(
        key="twitter_short",
        label="Twitter Short",
        instruction=(
            "Write as a single social post optimized for brevity and impact. "
            "Favor strong hooks and concise phrasing."
        ),
    ),
    "twitter_longform": ProfileSpec(
        key="twitter_longform",
        label="Twitter Longform",
        instruction=(
            "Write as a coherent thread-style longform post. "
            "Organize into clear sections and maintain narrative flow."
        ),
    ),
    "creative_story": ProfileSpec(
        key="creative_story",
        label="Creative Story",
        instruction=(
            "Prioritize voice, imagery, pacing, and emotional resonance. "
            "Use concrete details and avoid generic phrasing."
        ),
    ),
}


LENGTH_GUIDANCE: dict[str, str] = {
    "short": "Target a short response. Keep it tight and avoid unnecessary elaboration.",
    "medium": "Target a medium response with balanced detail and pacing.",
    "long": "Target a long response with developed structure and richer detail.",
}


BINARY_MODE_GUIDANCE: dict[str, str] = {
    "use_emojis": "Use emojis sparingly where they improve readability or tone.",
    "include_hashtags": "Include relevant hashtags at the end if appropriate.",
    "include_cta": "End with a clear call-to-action.",
    "thread_numbering": "Use explicit thread numbering (e.g., 1/N, 2/N) when multi-part.",
    "strict_280_chars": "Ensure each output segment stays within 280 characters.",
}


def parse_binary_modes(value: str | None) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        out.append(key)
    return out


def build_control_instruction(
    *,
    profile: str | None,
    desired_length: str | None,
    style: str | None,
    binary_modes: list[str] | None,
) -> str:
    lines: list[str] = []

    profile_key = (profile or "default").strip().lower()
    spec = PROFILE_SPECS.get(profile_key, PROFILE_SPECS["default"])
    lines.append(f"Response profile: {spec.label}.")
    lines.append(spec.instruction)

    length_key = (desired_length or "medium").strip().lower()
    length_line = LENGTH_GUIDANCE.get(length_key)
    if length_line:
        lines.append(length_line)

    if style:
        lines.append(f"Style focus: {style.strip()}.")

    for mode in (binary_modes or []):
        tip = BINARY_MODE_GUIDANCE.get(mode.strip())
        if tip:
            lines.append(tip)

    return "\n".join(lines).strip()


def merge_system_prompt(
    *,
    base_system_prompt: str | None,
    profile: str | None,
    desired_length: str | None,
    style: str | None,
    binary_modes: list[str] | None,
) -> str:
    control = build_control_instruction(
        profile=profile,
        desired_length=desired_length,
        style=style,
        binary_modes=binary_modes,
    )
    base = (base_system_prompt or "").strip()
    if base and control:
        return f"{base}\n\n{control}".strip()
    return (base or control).strip()


def list_profiles() -> list[dict[str, str]]:
    return [
        {"key": spec.key, "label": spec.label, "instruction": spec.instruction}
        for spec in PROFILE_SPECS.values()
    ]
