"""Explicit Arrow schemas for the exported record types, derived from the pydantic models.

JSON type inference breaks on nullable columns: a field that is null in the first file is typed
``null`` and a later file that holds a string or a number cannot be cast into it (this is what
makes ``load_dataset("json", ...)`` fail on ``temporal_qa.jsonl`` / ``video_descriptions.jsonl``).
Writing Parquet with the schema below removes the guesswork: every field has the type the model
declares, nested models become structs, and free-form ``dict`` / ``list[dict]`` fields are stored
as JSON strings (their keys are not fixed, so they have no stable Arrow type).
"""

from __future__ import annotations

import enum
import json
import types
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

Transform = Callable[[Any], Any] | None


def _json(value: Any) -> Any:
    return None if value is None else json.dumps(value, ensure_ascii=False, default=str)


def _plan(annotation: Any) -> tuple[pa.DataType, Transform]:
    """(arrow type, value transform) for one annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _plan(non_none[0])
        return pa.string(), _json  # mixed unions have no single Arrow type
    if annotation is typing.Any:
        return pa.string(), _json
    if annotation is bool:
        return pa.bool_(), None
    if annotation is int:
        return pa.int64(), None
    if annotation is float:
        return pa.float64(), None
    if annotation is str or (isinstance(annotation, type) and issubclass(annotation, enum.Enum)):
        return pa.string(), None
    if origin in (list, typing.List, tuple, set):  # noqa: UP006
        inner_t, inner_tf = _plan(args[0]) if args else (pa.string(), _json)
        if inner_tf is _json:
            return pa.string(), _json  # list of free-form values -> one JSON string
        if inner_tf is None:
            return pa.list_(inner_t), None
        return pa.list_(inner_t), (lambda xs, f=inner_tf: None if xs is None else [f(x) for x in xs])
    if origin in (dict, typing.Dict):  # noqa: UP006
        return pa.string(), _json
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        fields, tf = _struct_plan(annotation)
        return pa.struct(fields), tf
    return pa.string(), _json  # anything exotic is kept, as JSON


def _struct_plan(model: type[BaseModel]) -> tuple[list[pa.Field], Transform]:
    fields: list[pa.Field] = []
    transforms: dict[str, Callable[[Any], Any]] = {}
    for name, info in model.model_fields.items():
        t, tf = _plan(info.annotation)
        fields.append(pa.field(name, t, nullable=True))
        if tf is not None:
            transforms[name] = tf
    if not transforms:
        return fields, None

    def convert(d: Any) -> Any:
        if d is None:
            return None
        if isinstance(d, BaseModel):
            d = d.model_dump(mode="json")
        return {k: (transforms[k](v) if k in transforms else v) for k, v in d.items()}

    return fields, convert


def arrow_schema(model: type[BaseModel]) -> tuple[pa.Schema, Transform]:
    """Schema plus the row transform that JSON-encodes free-form fields (None when none exist)."""
    fields, tf = _struct_plan(model)
    return pa.schema(fields), tf


def to_table(model: type[BaseModel], rows: list[Any]) -> pa.Table:
    schema, tf = arrow_schema(model)
    dicts = [r.model_dump(mode="json") if isinstance(r, BaseModel) else dict(r) for r in rows]
    known = set(schema.names)
    dicts = [{k: v for k, v in d.items() if k in known} for d in dicts]
    if tf is not None:
        dicts = [tf(d) for d in dicts]
    return pa.Table.from_pylist(dicts, schema=schema)


def write_typed_parquet(model: type[BaseModel], rows: list[Any], path: Path) -> int:
    table = to_table(model, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return table.num_rows


def json_schemas(models: dict[str, type[BaseModel]]) -> dict[str, Any]:
    """JSON Schema per record type (for the dataset card / downstream validation)."""
    return {name: model.model_json_schema() for name, model in models.items()}
