"""Budget arithmetic: provider quotes, sourced estimates, and explicit reserves."""

from __future__ import annotations

import re

from travel_agent.reports.planning import number


def grounded_amounts(price, regional_currency):
    """Read money tokens from the actual excerpt; reject invented scales/units."""
    text = price["excerpt"]
    mixed = {
        "food": r"房费|住宿费|每晚|机票|车费|打车|门票",
        "taxi": r"餐费|饭钱|人均一餐|房费|住宿费|门票",
        "tickets": r"餐费|饭钱|房费|住宿费|打车费",
    }
    if re.search(mixed.get(price.get("category"), r"(?!)"), text):
        return None
    pattern = r"(?P<prefix>CNY|RMB|人民币|VND|越南盾|USD|美元|¥|￥|₫)?\s*(?P<number>\d+(?:[.,]\d+)?)\s*(?P<scale>万|千|w|k)?\s*(?P<suffix>CNY|RMB|人民币|VND|越南盾|USD|美元|元|块|盾|₫|r)?"
    values = []
    for match in re.finditer(pattern, text, re.I):
        prefix, suffix, scale = (match.group(k) or "" for k in ("prefix", "suffix", "scale"))
        token = (prefix + suffix).upper()
        if not token and not scale:
            continue
        if re.search(r"CNY|RMB|人民币|元|块|¥|￥|^R$", token):
            currency = "CNY"
        elif re.search(r"VND|越南盾|盾|₫", token):
            currency = "VND"
        elif re.search(r"USD|美元", token):
            currency = "USD"
        elif scale and regional_currency == "VND":
            currency = "VND"
        else:
            continue
        multiplier = {"万": 10000, "W": 10000, "千": 1000, "K": 1000}.get(scale.upper(), 1)
        amount = float(match["number"].replace(",", "")) * multiplier
        values.append((amount, currency))
    currency = price.get("currency")
    matching = [n for n, c in values if c == currency]
    if not matching:
        return None
    lo, hi = number(price.get("low")), number(price.get("high"))
    if (
        lo is None
        or hi is None
        or lo > hi
        or not all(any(abs(v - n) < 0.001 for n in matching) for v in (lo, hi))
    ):
        return None
    group = re.search(r"([一二三四五六七八九十\d]+)(?:个)?人.{0,5}(?:一共|总共|共花|合计)", text)
    if price["category"] == "food" and group:
        words = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        count = words.get(group[1], int(group[1]) if group[1].isdigit() else 1)
        lo, hi = lo / count, hi / count
    elif (
        price["category"] == "food"
        and re.search("海鲜餐|大餐|家常小炒", price.get("label", ""))
        and not re.search("人均|每人|一个人", text)
    ):
        return None
    if price["category"] == "taxi" and "往返" in text:
        lo, hi = lo / 2, hi / 2
    return {**price, "low": lo, "high": hi, "amount_validation": "原文金额、倍率及人数校验"}


def build_budget(planner, report):
    trip = planner.trip
    people, days = trip.traveler_count or 1, trip.duration_days or 1
    currency = trip.budget.currency
    previous = getattr(planner, "previous", {})
    prior_budget = previous.get("budget", {})
    prior_fx = prior_budget.get("fx", {})
    reuse_evidence = previous.get("notes") == report.get("notes")
    rate_data = (
        {"rates": prior_fx["rates"], "updated_at": prior_fx.get("updated_at")}
        if reuse_evidence and prior_fx.get("base") == currency and prior_fx.get("rates")
        else planner.call("research__exchange_rates", {"base": currency})
    )
    spending_basis = trip.budget.amount or trip.budget.min_amount or 1000 * people * days
    rates = rate_data.get("rates", {currency: 1})
    items = []

    def add(category, label, low, high, unit_currency, basis, status, url=""):
        factor = rates.get(unit_currency) if unit_currency else None
        if factor and low is not None and high is not None:
            low, high = round(low / factor), round(high / factor)
        else:
            low = high = None
        items.append(
            {
                "category": category,
                "label": label,
                "low": low,
                "high": high,
                "amount": low if low == high else None,
                "currency": currency,
                "basis": basis,
                "status": status,
                "url": url,
            }
        )

    for group in report["hotel_regions"]:
        baseline = next((h for h in group["hotels"] if h.get("baseline")), None)
        if baseline and baseline.get("price_is_total") and group.get("compatible", True):
            price = baseline["total_price"]
            tax_note = baseline.get("tax_note", "")
            taxes_unclear = not re.search(
                r"全部.*含税|含全部税|^含税费$|all taxes included|taxes.*included", tax_note, re.I
            )
            excluded = re.search(r"Excluded:\s*(.*)", tax_note, re.I)
            explicit_tax_rate = sum(float(p) for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", excluded[1])) / 100 if excluded else 0
            tax_reserve = max(0.15, explicit_tax_rate) if taxes_unclear else 0
            add(
                "酒店住宿",
                baseline["name"],
                price,
                price * (1 + tax_reserve),
                baseline["currency"],
                f"{group['nights']}晚 × {baseline['rooms']}间；{people}人。{baseline['price_note']}；{tax_note}"
                + (f"；上限另留{tax_reserve:.1%}税费差异" if taxes_unclear else ""),
                "检索报价",
                baseline["url"],
            )
        else:
            nights = group.get("nights", max(days - 1, 1))
            rooms = (people + 1) // 2
            # A reserve is a spending assumption, not an invented hotel quote.
            allowance = spending_basis * 0.25 / max(days - 1, 1) * nights
            add(
                "酒店住宿",
                (baseline or {}).get("name", group["city"] + "住宿预留"),
                allowance * 0.65,
                allowance,
                currency,
                f"{nights}晚、{rooms}间；按目标预算25%的住宿额度分配，未核实房价时使用，非该酒店报价。",
                "规划预留",
                (baseline or {}).get("url", ""),
            )
    for segment in report["transport"]:
        if not segment.get("compatible", True):
            continue
        option = next(
            (
                o
                for o in segment["options"]
                if number(o.get("price")) is not None or number(o.get("price_low")) is not None
            ),
            None,
        )
        if option:
            option = {
                **option,
                "price": option.get("price")
                if option.get("price") is not None
                else option.get("price_low"),
            }
            if option.get("price_basis") in {"per_group", "per_vehicle"}:
                option["price_unit"] = "group"
        if not option:
            reserve = spending_basis * (0.22 if segment.get("scope") == "external" else 0.05)
            add(
                "往返交通" if segment.get("scope") == "external" else "城际交通",
                f"{segment['origin']} → {segment['destination']}交通预留",
                reserve * 0.6,
                reserve,
                currency,
                "尚无可核实票价，按规划支出基准分配预留；非具体班次报价。",
                "规划预留",
                segment.get("url", ""),
            )
            continue
        if option.get("directions") and segment.get("scope") != "intercity":
            price = option["price"] * people
            add(
                "往返机票",
                f"{segment['origin']} ↔ {segment['destination']}",
                price,
                price * 1.1,
                option["currency"],
                f"每人往返平均参考价 {option['price']} × {people}人；"
                + (f"来源为{option.get('quote_passengers')}人合计报价{option.get('source_price')}；" if option.get('source_price') is not None else "")
                + "上限预留10%票价变化，行李等额外费用另核。",
                "检索报价推算",
                segment["url"],
            )
        else:
            price = option["price"] * (1 if option.get("price_unit") == "group" else people)
            add(
                "城际交通",
                f"{segment['origin']} → {segment['destination']}",
                price,
                (option.get("price_high") or option["price"])
                * (1 if option.get("price_unit") == "group" else people),
                option["currency"],
                f"{option.get('price_basis', '每人参考价')}；按{people}人推算，选择一个候选计入预算。",
                "检索报价" if segment.get("status") == "quote" else "来源参考估算",
                segment["url"],
            )
    snippets = []
    for note in report["notes"]:
        text = note["content"]
        matches = list(
            re.finditer(
                r"\d+(?:[.,]\d+)?\s*(?:RMB|CNY|VND|USD|元|万|千|w\b|k\b|r\b|美元|盾)", text, re.I
            )
        )
        excerpts = list(
            dict.fromkeys(text[max(0, m.start() - 55) : m.end() + 75] for m in matches)
        )[:8]
        if excerpts:
            snippets.append({"id": note["id"], "content": "\n".join(excerpts)[:1000]})
    snippets = snippets[:16]
    try:
        extracted = (
            {"prices": prior_budget.get("evidence", [])}
            if reuse_evidence
            else {"prices": []}
            if not snippets
            else planner.ask(
                "从正文提取用于费用估算的原始价格，不估造数值。只取food（每人一餐）、taxi（每车一段）、"
                "tickets（单个景点成人票）；单位只能是per_person_meal/per_vehicle_trip/per_person_ticket。"
                "最多12条，每类至多4条，excerpt必须为单个片段里连续原文。必须提供excerpt、note_id、币种ISO代码、原始low/high。没有价格就不返回。"
                '禁止把摩托租赁当出租车、两人船价当每人价格。返回 {"prices":[{"category":string,"label":string,'
                '"low":number,"high":number,"currency":string,"unit":string,"note_id":string,"excerpt":string}]}。',
                {"notes": snippets},
                "cost_evidence",
            )
        )
    except RuntimeError:
        planner.issues.append("费用经验价提取未完成，本版按明确标注的规划预留估算相关分项。")
        extracted = {"prices": []}
    byid = {n["id"]: n for n in report["notes"]}
    valid = []
    for p in extracted.get("prices", []):
        note = byid.get(p.get("note_id"))
        excerpt = str(p.get("excerpt") or "")
        low, high = number(p.get("low")), number(p.get("high"))
        if (
            note
            and excerpt
            and excerpt in note["content"]
            and re.search(r"\d", excerpt)
            and low is not None
            and high is not None
            and low <= high
            and p.get("currency") in rates
        ):
            grounded = grounded_amounts(p, report["region"].get("currency"))
            if grounded:
                valid.append(grounded)
    for category, label, share, unit in [
        ("food", "餐饮、小吃与咖啡", 0.12, "per_person_meal"),
        ("taxi", "市内打车与机场接驳", 0.06, "per_vehicle_trip"),
        ("tickets", "门票与体验", 0.08, "per_person_ticket"),
    ]:
        evidence = [p for p in valid if p.get("category") == category and p.get("unit") == unit]
        if evidence:
            # Experience prices are historical references with a deliberately wider range.
            lo = min(p["low"] / rates[p["currency"]] for p in evidence)
            hi = max(p["high"] / rates[p["currency"]] for p in evidence)
            count = (
                people * days * 2.5
                if category == "food"
                else sum(len(d["legs"]) for d in report["days"]) + 2
                if category == "taxi"
                else people * max(1, round(report["point_count"] * 0.4))
            )
            p = evidence[0]
            add(
                "餐饮" if category == "food" else "市内交通" if category == "taxi" else "门票体验",
                label,
                lo * count * 0.8,
                hi * count * 1.25,
                currency,
                f"笔记经验价范围 × {count:g}份/次，并留浮动；非实时报价。依据：{p['excerpt'][:120]}",
                "正文参考估算",
                byid[p["note_id"]]["url"],
            )
        else:
            allowance = spending_basis * share
            add(
                "餐饮" if category == "food" else "市内交通" if category == "taxi" else "门票体验",
                label,
                allowance * 0.6,
                allowance,
                currency,
                f"未取得单位明确的参考价；按规划支出基准{share:.0%}预留，{people}人{days}天，非商家报价。",
                "规划预留",
            )
    if not any(i["category"] in {"往返机票", "城际交通"} and i["low"] is not None for i in items):
        reserve = spending_basis * 0.3
        add(
            "往返交通",
            "尚待补查的交通预留",
            reserve * 0.65,
            reserve,
            currency,
            "未取得可用票价，按规划支出基准30%预留；非航班/车次报价。",
            "规划预留",
        )
    # Unconvertible facts remain visible; complete the estimate with a labelled reserve.
    for item in items:
        if item["low"] is None:
            reserve = spending_basis * 0.15
            item.update(
                low=round(reserve * 0.6),
                high=round(reserve),
                status="汇率缺失，规划预留",
                basis=item["basis"] + "；原币种未完成换算，以预算额度暂留。",
            )
    subtotal_low = sum(i["low"] for i in items)
    subtotal_high = sum(i["high"] for i in items)
    add(
        "机动费用",
        "应急及价格变化",
        subtotal_low * 0.08,
        subtotal_high * 0.12,
        currency,
        "按以上分项下限8%、上限12%预留，不重复计算备选酒店。",
        "规划预留",
    )
    low, high = sum(i["low"] for i in items), sum(i["high"] for i in items)
    target = trip.budget.max_amount if trip.budget.max_amount is not None else trip.budget.amount
    return {
        "items": items,
        "finalized": False,
        "total": None,
        "low": low,
        "high": high,
        "currency": currency,
        "per_person_low": round(low / people),
        "per_person_high": round(high / people),
        "target": target,
        "remaining_low": round(target - high) if target is not None else None,
        "remaining_high": round(target - low) if target is not None else None,
        "fx": {
            "updated_at": rate_data.get("updated_at"),
            "source_url": "https://www.exchangerate-api.com",
            "base": currency,
            "rates": {
                k: v
                for k, v in rates.items()
                if k in {currency, report["region"].get("currency"), "USD", "HKD"}
            },
        },
        "evidence": valid,
        "reason": "规划估算范围；包含实查报价、正文经验参考与明确标注的预留，未代表已锁价或已预订。",
    }
