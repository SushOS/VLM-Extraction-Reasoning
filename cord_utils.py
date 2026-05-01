from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_ground_truth(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value


def normalize_menu(menu: Any) -> list[dict[str, Any]]:
    if isinstance(menu, dict):
        menu = [menu]
    if not isinstance(menu, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in menu:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "nm": item.get("nm"),
                "cnt": item.get("cnt"),
                "unitprice": item.get("unitprice"),
                "price": item.get("price"),
            }
        )
    return rows


def normalize_optional_block(block: Any, keys: list[str]) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {key: None for key in keys}
    return {key: block.get(key) for key in keys}


def cord_ground_truth_to_target(value: str | dict[str, Any]) -> dict[str, Any]:
    payload = load_ground_truth(value)
    gt_parse = payload.get("gt_parse", {})
    return {
        "menu": normalize_menu(gt_parse.get("menu")),
        "sub_total": normalize_optional_block(
            gt_parse.get("sub_total"),
            ["subtotal_price", "discount_price", "service_price", "tax_price"],
        ),
        "total": normalize_optional_block(
            gt_parse.get("total"),
            ["total_price", "cashprice", "changeprice", "creditcardprice", "menuqty_cnt"],
        ),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
