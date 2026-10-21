"""Turn a RAGTruth Data2txt business record (Yelp JSON) into plain sentences for NLI.

NLI models are trained on sentences, not JSON. Rules (from notes/ragtruth_inspection.md):
- null / missing fields are omitted: null means *unknown*, and writing it as "no" would turn an
  unsupported claim into a contradicted one.
- 24h times become 12h ("20:30" -> "8:30 PM"), so "10:30 pm" vs 8:30 PM is a plain mismatch.
- "0:0-0:0" hours are ambiguous (38 annotator notes read them as closed, 4 as open 24/7, 105
  say nothing). Decided: follow the annotator majority ("closed"); "literal" kept for comparison.
- Music is sometimes a string with every "u" stripped ("{dj: None, live: Tre, jkebox: ...}");
  it is repaired and parsed.
"""

from __future__ import annotations

import re

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ZERO_HOURS_RULES = ("closed", "literal")

# attribute -> (sentence if True, sentence if False); {n} is the business name
BOOL_ATTRS = {
    "RestaurantsReservations": ("{n} takes reservations.", "{n} does not take reservations."),
    "OutdoorSeating": ("{n} has outdoor seating.", "{n} does not have outdoor seating."),
    "RestaurantsTakeOut": ("{n} offers takeout.", "{n} does not offer takeout."),
    "RestaurantsGoodForGroups": ("{n} is good for groups.", "{n} is not good for groups."),
}
WIFI = {"free": "{n} offers free WiFi.", "paid": "{n} offers paid WiFi.", "no": "{n} does not offer WiFi."}
PARKING = {
    "garage": "garage parking",
    "street": "street parking",
    "lot": "a parking lot",
    "valet": "valet parking",
    "validated": "validated parking",
}
MUSIC = {
    "dj": "a DJ",
    "live": "live music",
    "jukebox": "a jukebox",
    "video": "music videos",
    "background_music": "background music",
    "karaoke": "karaoke",
}
# the "u"-stripped spellings found in the data
_MUSIC_REPAIR = {"jkebox": "jukebox", "backgrond_msic": "background_music", "no_msic": "no_music"}
_BOOL_REPAIR = {"Tre": True, "True": True, "False": False, "None": None}


def format_time(hm: str) -> str:
    """'20:30' or '20:0' -> '8:30 PM'; '0:0' -> '12:00 AM'."""
    h, m = (int(x) for x in hm.split(":"))
    suffix = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12}:{m:02d} {suffix}"


def hours_sentence(name: str, day: str, span: str, zero_rule: str = "closed") -> str:
    if zero_rule not in ZERO_HOURS_RULES:
        raise ValueError(f"zero_rule must be one of {ZERO_HOURS_RULES}")
    start, end = span.split("-")
    if span == "0:0-0:0":
        if zero_rule == "closed":
            return f"{name} is closed on {day}."
        return f"On {day}, the listed hours for {name} are 0:00 to 0:00."
    return f"On {day}, {name} is open from {format_time(start)} to {format_time(end)}."


def parse_music(value) -> dict[str, bool | None] | bool | None:
    if not isinstance(value, str):
        return value
    out = {}
    for key, raw in re.findall(r"(\w+):\s*(\w+)", value):
        out[_MUSIC_REPAIR.get(key, key)] = _BOOL_REPAIR.get(raw)
    return out


def business_facts(info: dict, zero_rule: str = "closed") -> list[str]:
    """One sentence per known fact about the business (reviews excluded)."""
    n = info["name"]
    facts = [f"{n} is located at {info['address']}, {info['city']}, {info['state']}."]
    if info.get("categories"):
        facts.append(f"{n} is categorized as: {info['categories']}.")
    if info.get("business_stars") is not None:
        facts.append(f"{n} has an overall rating of {info['business_stars']} stars.")

    for day in DAYS:  # days not listed are unknown, so omitted
        span = (info.get("hours") or {}).get(day)
        if span:
            facts.append(hours_sentence(n, day, span, zero_rule))

    attrs = info.get("attributes") or {}
    for key, (yes, no) in BOOL_ATTRS.items():
        if attrs.get(key) is not None:
            facts.append((yes if attrs[key] else no).format(n=n))
    if attrs.get("WiFi") in WIFI:
        facts.append(WIFI[attrs["WiFi"]].format(n=n))
    for key, phrase in PARKING.items():
        val = (attrs.get("BusinessParking") or {}).get(key)
        if val is not None:
            facts.append(f"{n} {'has' if val else 'does not have'} {phrase}.")

    music = parse_music(attrs.get("Music"))
    if isinstance(music, bool):
        facts.append(f"{n} {'has' if music else 'does not have'} music.")
    elif isinstance(music, dict):
        for key, phrase in MUSIC.items():
            if music.get(key) is not None:
                facts.append(f"{n} {'has' if music[key] else 'does not have'} {phrase}.")

    for key, val in (attrs.get("Ambience") or {}).items():
        if val is not None:
            facts.append(f"The ambience at {n} is {'' if val else 'not '}{key}.")
    return facts


def review_passages(info: dict) -> list[str]:
    return [
        f"A {r['review_stars']:g}-star customer review from {r['review_date'][:10]}: "
        f"{r['review_text'].strip()}"
        for r in info.get("review_info") or []
    ]
