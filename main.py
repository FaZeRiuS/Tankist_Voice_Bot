import asyncio
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, Router
from aiogram.dispatcher.middlewares.base import BaseMiddleware
from aiogram.enums import ChatType
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineQuery,
    InlineQueryResultCachedVoice,
    Message,
)

from db import add_voice_sample, get_random_voice_sample, init_db, search_voice_samples


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    bot_token: str
    owner_id: int
    db_path: str = "voice_samples.db"


def _load_settings() -> Settings:
    token = (os.getenv("BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("BOT_TOKEN is required")

    owner_raw = (os.getenv("OWNER_ID") or "").strip()
    if not owner_raw:
        raise RuntimeError("OWNER_ID is required")
    try:
        owner_id = int(owner_raw)
    except ValueError as e:
        raise RuntimeError("OWNER_ID must be an integer") from e

    db_path = (os.getenv("DB_PATH") or "voice_samples.db").strip()
    return Settings(bot_token=token, owner_id=owner_id, db_path=db_path)


class IngestVoice(StatesGroup):
    waiting_for_title = State()


router = Router()


class SettingsMiddleware(BaseMiddleware):
    def __init__(self, settings: Settings):
        self._settings = settings

    async def __call__(self, handler, event, data):
        data["settings"] = self._settings
        return await handler(event, data)


def _is_owner_private(message: Message, owner_id: int) -> bool:
    return (
        message.chat.type == ChatType.PRIVATE
        and message.from_user is not None
        and message.from_user.id == owner_id
    )


@router.message(Command("start"))
async def start(message: Message) -> None:
    await message.answer(
        "Send me a voice message in this private chat (owner only) to save it.\n"
        "Then use me inline: type @<bot_username> <query> in any chat."
    )


@router.message(Command("cancel"))
async def cancel(message: Message, state: FSMContext) -> None:
    if await state.get_state() is None:
        return
    await state.clear()
    await message.answer("Cancelled.")


@router.message(F.new_chat_members)
async def welcome_random_voice(message: Message, settings: Settings, bot: Bot) -> None:
    if message.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        return
    members = message.new_chat_members or []
    if not members:
        return
    if not any(not u.is_bot for u in members):
        return
    try:
        row = await get_random_voice_sample(settings.db_path)
    except Exception:
        logger.exception("Failed to fetch random voice sample for welcome")
        return
    if row is None:
        logger.info("No voice samples in DB; skipping welcome voice.")
        return
    file_id = str(row.get("file_id") or "").strip()
    if not file_id:
        return
    try:
        await bot.send_voice(
            chat_id=message.chat.id,
            voice=file_id,
            reply_to_message_id=message.message_id,
        )
    except Exception:
        logger.exception("Failed to send welcome voice to chat %s", message.chat.id)


@router.message(F.voice)
async def ingest_voice(message: Message, state: FSMContext, settings: Settings) -> None:
    # Only owner in private chat may ingest
    if message.chat.type != ChatType.PRIVATE:
        return
    if message.from_user is None or message.from_user.id != settings.owner_id:
        await message.answer("Not authorized.")
        return

    file_id = message.voice.file_id
    caption = (message.caption or "").strip()

    if caption:
        try:
            row_id = await add_voice_sample(settings.db_path, caption, file_id)
        except Exception:
            logger.exception("Failed to add voice sample (caption path)")
            await message.answer("Failed to save voice sample. Check logs.")
            return

        await message.answer(f"Saved voice sample #{row_id}: {caption}")
        return

    await state.set_state(IngestVoice.waiting_for_title)
    await state.update_data(file_id=file_id)
    await message.answer("Send a title for this voice sample (or /cancel).")


@router.message(IngestVoice.waiting_for_title, F.text)
async def ingest_title(message: Message, state: FSMContext, settings: Settings) -> None:
    if not _is_owner_private(message, settings.owner_id):
        # Ignore silently (state is per-user; but be defensive)
        return

    title = (message.text or "").strip()
    if not title:
        await message.answer("Title cannot be empty. Send a title (or /cancel).")
        return

    data = await state.get_data()
    file_id = str(data.get("file_id") or "").strip()
    if not file_id:
        await state.clear()
        await message.answer("Missing pending voice. Please send the voice message again.")
        return

    try:
        row_id = await add_voice_sample(settings.db_path, title, file_id)
    except Exception:
        logger.exception("Failed to add voice sample (FSM title path)")
        await message.answer("Failed to save voice sample. Check logs.")
        return
    finally:
        await state.clear()

    await message.answer(f"Saved voice sample #{row_id}: {title}")


@router.inline_query()
async def inline_query_handler(inline_query: InlineQuery, settings: Settings) -> None:
    query = (inline_query.query or "").strip()
    offset_raw = (inline_query.offset or "").strip()
    try:
        offset = int(offset_raw) if offset_raw else 0
    except ValueError:
        offset = 0
    offset = max(0, offset)
    page_size = 20

    try:
        rows = await search_voice_samples(
            settings.db_path,
            query=query,
            limit=page_size,
            offset=offset,
        )
    except Exception:
        logger.exception("Inline search failed")
        await inline_query.answer(
            results=[],
            is_personal=True,
            cache_time=1,
            switch_pm_text="Search error. Try again later.",
            switch_pm_parameter="start",
        )
        return

    results = [
        InlineQueryResultCachedVoice(
            id=str(r["id"]),
            voice_file_id=str(r["file_id"]),
            title=str(r["title"]),
        )
        for r in rows
    ]

    if not results:
        # No results for this page. If it's the first page, show the friendly empty-state.
        if offset == 0:
            await inline_query.answer(
                results=[],
                is_personal=True,
                cache_time=1,
                switch_pm_text="No matches. Ask admin to add samples.",
                switch_pm_parameter="start",
            )
            return
        await inline_query.answer(
            results=[],
            is_personal=True,
            cache_time=1,
            next_offset="",
        )
        return

    await inline_query.answer(
        results=results,
        is_personal=True,
        cache_time=3,
        next_offset=str(offset + len(results)),
    )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load variables from .env into process environment (if file exists).
    load_dotenv()

    settings = _load_settings()
    await init_db(settings.db_path)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    dp.update.outer_middleware(SettingsMiddleware(settings))
    dp.include_router(router)

    logger.info("Bot started. Inline mode must be enabled in BotFather.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass

"""
BotFather setup reminder:
- In @BotFather -> your bot -> Bot Settings -> Inline Mode -> ENABLE
- Set an inline placeholder (e.g., 'Search voice samples…')
"""

