"""Read local UTF-8 JSONL and validate the fields used by preparation."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path


class Jsonl:
    def __init__(self, path: Path):
        self.path = path
        self.sha256 = hashlib.sha256()
        self.rows = 0

    def __iter__(self) -> Iterator[tuple[dict, str]]:
        with self.path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                self.sha256.update(raw)
                if not raw.strip():
                    continue
                where = f"{self.path}:{line_number}"
                try:
                    item = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{where}: expected a UTF-8 JSON object") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"{where}: expected a JSON object")
                self.rows += 1
                yield item, where

    def provenance(self) -> dict:
        return {"path": str(self.path), "sha256": self.sha256.hexdigest(), "rows": self.rows}


def text_field(item: dict, key: str, where: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key!r} must be a nonempty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{where}: {key!r} contains invalid Unicode") from exc
    return value


def teacher_fields(item: dict, where: str) -> dict:
    fields = {key: text_field(item, key, where) for key in ("problem", "solution", "source")}
    identifier = item.get("id")
    if identifier is not None:
        if isinstance(identifier, str):
            fields["id"] = text_field(item, "id", where)
        elif type(identifier) is int:
            fields["id"] = identifier
        else:
            raise ValueError(f"{where}: 'id' must be a nonempty string or an integer")
    return fields
