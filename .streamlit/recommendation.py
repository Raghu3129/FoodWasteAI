"""
recommendation.py
------------------
Rule-based recommendation engine. Generates actionable, human-readable
suggestions based on the predicted waste category, waste percentage and
extracted features. Recommendations vary across calls (not a single
static string) by drawing from category-specific pools and adding
data-driven notes.
"""

import random

LOW_WASTE_TIPS = [
    "Portion sizes for this item are well matched to demand — maintain current serving guidelines.",
    "Consider this dish a benchmark; apply its recipe/portioning approach to higher-waste items.",
    "Minor plate residue only — no immediate action needed, continue monitoring weekly.",
]

MODERATE_WASTE_TIPS = [
    "Slightly reduce default serving size for this item by 10-15% and monitor feedback.",
    "Review menu planning for this item; consider offering smaller/regular portion options.",
    "Increase visibility of 'take only what you eat' signage near this counter.",
    "Cross-check preparation quantity against average footfall to reduce overproduction.",
]

HIGH_WASTE_TIPS = [
    "Significantly reduce default serving size and offer a smaller portion option at the counter.",
    "Investigate root cause: taste, temperature, or overproduction — collect quick student feedback.",
    "Launch an awareness campaign (posters, announcements) highlighting food waste impact.",
    "Optimize kitchen preparation quantities using recent consumption trends to avoid overcooking.",
    "Consider donating excess prepared food to reduce this category of waste going forward.",
]

GENERAL_TIPS = [
    "Track this item's waste trend over the next 7 days to confirm if this is a one-off or a pattern.",
    "Compare waste levels across meal timings (breakfast/lunch/dinner) to identify peak-waste slots.",
]


def generate_recommendations(category: str, waste_percentage: float, features: dict, max_tips=4):
    """
    Build a recommendation list tailored to the prediction outcome.

    The pool sampled from depends on the category, and one general
    monitoring tip is always appended for continuity across analyses.
    """
    if category == "Low Waste":
        pool = LOW_WASTE_TIPS
    elif category == "Moderate Waste":
        pool = MODERATE_WASTE_TIPS
    else:
        pool = HIGH_WASTE_TIPS

    n_specific = min(max_tips - 1, len(pool))
    tips = random.sample(pool, n_specific)
    tips.append(random.choice(GENERAL_TIPS))

    # Add one data-driven note referencing the actual waste percentage
    data_note = f"Detected leftover coverage of {waste_percentage:.1f}% of the plate area in this analysis."
    tips.insert(0, data_note)

    return tips
