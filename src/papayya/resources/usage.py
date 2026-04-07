from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from papayya.api import APIClient


class Usage:
    def __init__(self, api: APIClient) -> None:
        self._api = api

    def summary(self, *, from_date: str | None = None, to_date: str | None = None) -> dict[str, Any]:
        params: dict[str, str] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._api._request("GET", "/v1/usage", params=params)

    def breakdown(self, *, from_date: str | None = None, to_date: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, str] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return self._api._request("GET", "/v1/usage/breakdown", params=params)
