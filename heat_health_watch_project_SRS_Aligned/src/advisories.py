"""General health advisories per project-defined category (educational only)."""
from __future__ import annotations

from . import config as C

_ADVISORIES = {
    "Very Low": {
        "summary": "Conditions are comfortable. Maintain normal routines and stay hydrated.",
        "actions": ["Drink water regularly.", "Normal outdoor activity is generally fine."],
    },
    "Low": {
        "summary": "Mild heat or air-quality stress. Stay hydrated and take sensible precautions.",
        "actions": ["Drink water regularly.", "Take breaks in shade during long outdoor activity.",
                    "Check the local air-quality report if you have respiratory conditions."],
    },
    "Moderate": {
        "summary": "Noticeable heat and/or pollution load. Drink plenty of water and avoid prolonged exposure to direct sunlight.",
        "actions": ["Increase water intake.", "Prefer shade and cooler parts of the day for exertion.",
                    "Wear light, loose clothing.", "Watch official weather and air-quality alerts."],
    },
    "High": {
        "summary": "Elevated heat-health stress. Limit strenuous outdoor activity and stay in cool, well-ventilated places.",
        "actions": ["Reduce strenuous outdoor work or exercise, especially in the afternoon.",
                    "Rest in cool or well-ventilated places.",
                    "Check on older adults, children and people with chronic conditions.",
                    "Follow official IMD / CPCB warnings."],
    },
    "Severe": {
        "summary": "Very high heat-health stress. Avoid unnecessary outdoor exposure and seek professional help if heat-related symptoms appear.",
        "actions": ["Stay in a cool indoor space; avoid unnecessary outdoor exposure.",
                    "Avoid strenuous activity outdoors.",
                    "Seek medical help if you feel dizzy, faint or unwell.",
                    "Follow official local warnings and instructions."],
    },
}


def advisory_for_category(category: str) -> dict:
    """Return {'category','summary','actions','disclaimer'}; KeyError for unknown category."""
    if category not in _ADVISORIES:
        raise KeyError(f"Unknown risk category {category!r}. Valid: {C.LEVEL_ORDER}")
    a = _ADVISORIES[category]
    return {"category": category, "summary": a["summary"], "actions": list(a["actions"]),
            "disclaimer": C.DISCLAIMER}


def advisory_text(category: str) -> str:
    return advisory_for_category(category)["summary"]
