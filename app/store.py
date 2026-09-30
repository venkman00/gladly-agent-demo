"""Ticket persistence: DynamoDB in AWS, a dict locally. Same interface.

`create` is conditional, which gives us idempotency: Gladly (and SQS) deliver
*at least once*, so the same event can arrive twice and must only be processed once.
"""

import asyncio
import json
from decimal import Decimal
from typing import Protocol

import boto3
from boto3.dynamodb.conditions import Attr

from app.models import Ticket


class Store(Protocol):
    async def create(self, ticket: Ticket) -> bool: ...  # False if it already exists
    async def get(self, ticket_id: str) -> Ticket | None: ...
    async def save(self, ticket: Ticket) -> None: ...
    async def recent(self, limit: int = 50) -> list[Ticket]: ...


class MemoryStore:
    def __init__(self):
        self.items: dict[str, Ticket] = {}

    async def create(self, ticket: Ticket) -> bool:
        return self.items.setdefault(ticket.id, ticket) is ticket

    async def get(self, ticket_id: str) -> Ticket | None:
        return self.items.get(ticket_id)

    async def save(self, ticket: Ticket) -> None:
        self.items[ticket.id] = ticket

    async def recent(self, limit: int = 50) -> list[Ticket]:
        return sorted(self.items.values(), key=lambda t: t.created_at, reverse=True)[:limit]


class DynamoStore:
    def __init__(self, table_name: str, region: str):
        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)

    @staticmethod
    def _item(ticket: Ticket) -> dict:
        # DynamoDB wants Decimal, not float. A JSON round-trip handles nesting.
        return json.loads(ticket.model_dump_json(), parse_float=Decimal)

    async def create(self, ticket: Ticket) -> bool:
        try:
            await asyncio.to_thread(
                self.table.put_item, Item=self._item(ticket), ConditionExpression=Attr("id").not_exists()
            )
            return True
        except self.table.meta.client.exceptions.ConditionalCheckFailedException:
            return False

    async def get(self, ticket_id: str) -> Ticket | None:
        resp = await asyncio.to_thread(self.table.get_item, Key={"id": ticket_id})
        return Ticket.model_validate(resp["Item"]) if "Item" in resp else None

    async def save(self, ticket: Ticket) -> None:
        await asyncio.to_thread(self.table.put_item, Item=self._item(ticket))

    async def recent(self, limit: int = 50) -> list[Ticket]:
        # Fine for a demo-sized table. At scale, add a GSI on (status, created_at) and Query it instead.
        resp = await asyncio.to_thread(self.table.scan, Limit=500)
        tickets = [Ticket.model_validate(i) for i in resp["Items"]]
        return sorted(tickets, key=lambda t: t.created_at, reverse=True)[:limit]
