"""Interpret explicit visible room rows, never the MCP's guessed total flag."""

from __future__ import annotations

import math
import re
from urllib.parse import parse_qs, urlsplit


def compact(text):
    return " ".join(str(text).replace("\u00a0", " ").split())


def visible_hotel_quote(evidence, nights, adults, rooms, currency, checkin, checkout):
    query = parse_qs(urlsplit(evidence.get("url", "")).query)
    if query.get("checkin") != [checkin] or query.get("checkout") != [checkout]:
        return None
    contexts = evidence.get("stay_context", [])
    header = compact(contexts[0]) if contexts else ""
    cards = []
    for card in evidence.get("cards", []):
        text = compact(card.get("text", ""))
        if not re.search(rf"\b{nights} nights?\b", text, re.I) or not re.search(
            rf"\b{adults} adults?\b", text, re.I
        ):
            continue
        if rooms > 1 and not re.search(rf"\b{rooms} rooms?\b", text, re.I):
            continue
        # The upstream price selector can read the "Per night" badge instead
        # of the discounted stay total shown elsewhere in the same card.
        amounts = re.findall(rf"Current price {currency}\s*([\d,]+(?:\.\d+)?)", text, re.I)
        if not amounts:
            amounts = re.findall(
                rf"(?<!Original )Price {currency}\s*([\d,]+(?:\.\d+)?)", text, re.I
            )
        if not amounts and not re.search("per night", text, re.I):
            amounts = re.findall(
                rf"{currency}\s*([\d,]+(?:\.\d+)?)", compact(card.get("price_text", ""))
            )
        if amounts:
            amount = float(amounts[-1].replace(",", ""))
            cards.append(
                {
                    "total_price": amount,
                    "currency": currency,
                    "price_is_total": True,
                    "evidence_excerpt": card["text"],
                    "room_name": "搜索卡片对应房型，详情需复核",
                    "price_note": f"可见搜索卡片明确{nights}晚、{adults}人；{rooms}间入住总额{amount:g} {currency}。",
                    "tax_note": "含税费"
                    if re.search("includes taxes|taxes and fees included", text, re.I)
                    else "卡片税费范围需复核",
                }
            )
    if cards:
        return min(cards, key=lambda c: c["total_price"])
    if not re.search(rf"Price for {nights} nights?\b", header, re.I) and not any(
        re.search(rf"\b{nights} nights?\b", compact(row.get("text", "")), re.I) for row in evidence.get("rows", [])
    ):
        return None
    if currency not in compact(evidence.get("currency_text", "")):
        return None
    if not any(
        re.search(rf"\b{adults} adults?\b", compact(c))
        and re.search(rf"\b{rooms} rooms?\b", compact(c))
        for c in contexts
    ):
        return None
    candidates = []
    for row in evidence.get("rows", []):
        text = compact(row.get("text", ""))
        occupancy = re.search(r"Sleeps:\s*(\d+) adults|Max\.\s*people:\s*(\d+)", text, re.I)
        capacity = next((int(g) for g in occupancy.groups() if g is not None), 0) if occupancy else 0
        if capacity < math.ceil(adults / rooms):
            continue
        current = re.findall(rf"Current price {currency}\s*([\d,]+(?:\.\d+)?)", text, re.I)
        values = current or re.findall(rf"(?:Price )?{currency}\s*([\d,]+(?:\.\d+)?)", text)
        if not values:
            continue
        amount = float(values[0].replace(",", "")) * rooms
        tax = re.search(
            r"(?:Excluded|Included):.{0,80}?(?=Exceptional breakfast|Breakfast|Free cancellation|Pay |Select Rooms|$)",
            text,
            re.I,
        )
        candidates.append(
            {
                "total_price": amount,
                "currency": currency,
                "price_is_total": True,
                "evidence_excerpt": row["text"],
                "room_name": re.split(r"Sleeps:|Max\.\s*people:", text, maxsplit=1, flags=re.I)[0].strip(),
                "price_note": f"可见房型表明确为{nights}晚，{rooms}间、{adults}人；入住总额{amount:g} {currency}。",
                "tax_note": tax[0] if tax else "税费范围需复核",
            }
        )
    return min(candidates, key=lambda c: c["total_price"]) if candidates else None
