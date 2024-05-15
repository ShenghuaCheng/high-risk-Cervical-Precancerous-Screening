# -*- coding: utf-8 -*-
import json


def read_json(json_path: str) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        j = json.load(f)

    return j


def save_json(dict_dataset: dict, name: str):
    with open(name, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict_dataset, ensure_ascii=False, indent=4))