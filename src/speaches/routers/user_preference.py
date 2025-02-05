from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field, conlist
from typing import List
from enum import IntEnum
import json
from pathlib import Path

router = APIRouter(tags=["user_preference"], prefix="/v1/preferences")


class CEFRLevel(IntEnum):
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6


class UserPreferences(BaseModel):
    user_level: CEFRLevel = Field(description="CEFR level: 1=A1 through 6=C2")
    user_interests: conlist(str, min_items=1, max_items=10) = Field(
        description="List of topics/themes the user is interested in learning"
    )


class JSONStore:
    def __init__(self, file_path: str = "preferences_store.json"):
        self.file_path = Path(file_path)
        self.initialize_store()

    def initialize_store(self):
        if not self.file_path.exists():
            self.file_path.write_text("{}")

    def read(self) -> dict:
        return json.loads(self.file_path.read_text())

    def write(self, data: dict):
        self.file_path.write_text(json.dumps(data, indent=2))

    def get_user_preferences(self, user_id: str) -> dict:
        store = self.read()
        return store.get(str(user_id))

    def save_user_preferences(self, user_id: str, preferences: dict):
        store = self.read()
        store[str(user_id)] = preferences
        self.write(store)

    def user_exists(self, user_id: str) -> bool:
        store = self.read()
        return str(user_id) in store


store = JSONStore()


@router.post("/{user_id}")
async def create_user_preferences(
    user_id: int, preferences: UserPreferences
) -> UserPreferences:
    try:
        if store.user_exists(str(user_id)):
            raise HTTPException(
                status_code=409,
                detail="User preferences already exist. Use PUT to update.",
            )

        store.save_user_preferences(str(user_id), preferences.model_dump())
        return preferences
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{user_id}")
async def update_user_preferences(
    user_id: int, preferences: UserPreferences
) -> UserPreferences:
    try:
        if not store.user_exists(str(user_id)):
            raise HTTPException(
                status_code=404,
                detail="User preferences not found. Use POST to create.",
            )

        store.save_user_preferences(str(user_id), preferences.model_dump())
        return preferences
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{user_id}")
async def get_user_preferences(user_id: int) -> UserPreferences:
    try:
        prefs = store.get_user_preferences(str(user_id))
        if not prefs:
            raise HTTPException(status_code=404, detail="User preferences not found")
        return UserPreferences(**prefs)
    except Exception as e:
        raise HTTPException(status_code=404, detail="User preferences not found")
