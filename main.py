import asyncio
import logging
import os
import random
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

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

from db import (
    add_voice_sample,
    get_message_count,
    get_random_voice_sample,
    get_user_history,
    get_user_profile,
    get_user_profile,
    init_db,
    is_user_tracked,
    add_tracked_user,
    remove_tracked_user,
    log_user_message,
    search_voice_samples,
    update_user_profile,
)
from llm import generate_personalized_reply, setup_gemini, summarize_user_history


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    bot_token: str
    owner_id: int
    gemini_api_key: str | None = None
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
    gemini_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    return Settings(
        bot_token=token,
        owner_id=owner_id,
        db_path=db_path,
        gemini_api_key=gemini_key or None,
    )


class IngestVoice(StatesGroup):
    waiting_for_title = State()


router = Router()


class SettingsMiddleware(BaseMiddleware):
    def __init__(self, settings: Settings):
        self._settings = settings

    async def __call__(self, handler, event, data):
        data["settings"] = self._settings
        return await handler(event, data)


# Per chat: after a random 50–100 user messages, reply with a random voice (groups only).
_random_voice_chat_state: dict[int, dict[str, int]] = {}


def _random_voice_next_threshold() -> int:
    return random.randint(25, 60)


async def _analyze_user_if_needed(user_id: int, settings: Settings) -> None:
    """Every 10 messages, trigger background analysis to update the user profile."""
    count = await get_message_count(settings.db_path, user_id)
    if count > 0 and count % 10 == 0:
        logger.info("Triggering analysis for user %s", user_id)
        history = await get_user_history(settings.db_path, user_id, limit=30)
        profile = await summarize_user_history(history)
        if profile:
            await update_user_profile(settings.db_path, user_id, profile)
            logger.info("Updated profile for user %s: %s", user_id, profile)


class RandomVoiceReplyMiddleware(BaseMiddleware):
    """Logs messages and occasionally replies with a voice sample or a personalized LLM response."""

    def __init__(self, settings: Settings):
        self._settings = settings

    async def __call__(self, handler, event, data):
        msg = event if isinstance(event, Message) else None
        
        # Log text messages ONLY if the user is tracked
        if msg and msg.text and not msg.from_user.is_bot:
            if await is_user_tracked(self._settings.db_path, msg.from_user.id):
                await log_user_message(
                    self._settings.db_path, 
                    msg.from_user.id, 
                    msg.chat.id, 
                    msg.text
                )
                # Trigger analysis in background
                asyncio.create_task(_analyze_user_if_needed(msg.from_user.id, self._settings))

        result = await handler(event, data)
        
        if msg is None:
            return result
        if msg.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
            return result
        if msg.from_user is None or msg.from_user.is_bot:
            return result

        bot = data.get("bot")
        if bot is None:
            return result

        st = _random_voice_chat_state.get(msg.chat.id)
        if st is None:
            st = {"count": 0, "threshold": _random_voice_next_threshold()}
            _random_voice_chat_state[msg.chat.id] = st
        st["count"] += 1
        if st["count"] < st["threshold"]:
            return result
        
        st["count"] = 0
        st["threshold"] = _random_voice_next_threshold()

        # Decide between Voice and Text
        # If Gemini is not set up, always fallback to voice
        use_text = random.random() < 0.4 and self._settings.gemini_api_key is not None
        
        if use_text:
            profile = await get_user_profile(self._settings.db_path, msg.from_user.id)
            if profile and msg.text:
                reply_text = await generate_personalized_reply(profile, msg.text)
                if reply_text:
                    try:
                        await bot.send_message(
                            chat_id=msg.chat.id,
                            text=reply_text,
                            reply_to_message_id=msg.message_id,
                        )
                        return result
                    except Exception:
                        logger.exception("Failed to send personalized reply")
        
        # Fallback to random voice
        try:
            row = await get_random_voice_sample(self._settings.db_path)
        except Exception:
            logger.exception("Random voice reply: failed to fetch sample")
            return result
        
        if row is None:
            return result
        
        file_id = str(row.get("file_id") or "").strip()
        if not file_id:
            return result
            
        try:
            await bot.send_voice(
                chat_id=msg.chat.id,
                voice=file_id,
                reply_to_message_id=msg.message_id,
            )
        except Exception:
            logger.exception("Random voice reply: send_voice failed for chat %s", msg.chat.id)
        
        return result


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
        "Then use me inline: type @<bot_username> <query> in any chat.\n\n"
        "I also analyze chat behavior to generate personalized responses!"
    )


@router.message(Command("profile"))
async def show_profile(message: Message, settings: Settings) -> None:
    """Show the collected information about the user."""
    profile = await get_user_profile(settings.db_path, message.from_user.id)
    if profile:
        await message.reply(f"🔍 **Ось що я про тебе знаю:**\n\n{profile}")
    else:
        count = await get_message_count(settings.db_path, message.from_user.id)
        await message.reply(
            f"Я ще не зібрав достатньо інформації про тебе. "
            f"Мені потрібно ще хоча б {10 - count} твоїх повідомлень."
        )


@router.message(Command("whois"))
async def whois_user(message: Message, settings: Settings) -> None:
    """Show the collected information about the replied-to user."""
    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("Ця команда працює лише як відповідь (reply) на повідомлення іншого користувача.")
        return

    target_user = message.reply_to_message.from_user
    if target_user.is_bot:
        await message.reply("Я не аналізую інших ботів.")
        return

    profile = await get_user_profile(settings.db_path, target_user.id)
    if profile:
        await message.reply(f"🔍 **Ось що я знаю про {target_user.full_name}:**\n\n{profile}")
    else:
        count = await get_message_count(settings.db_path, target_user.id)
        await message.reply(
            f"У мене ще немає профілю для {target_user.full_name}. "
            f"В базі всього {count} повідомлень від цього користувача."
        )


@router.message(Command("analyze"))
async def force_analyze_user(message: Message, settings: Settings) -> None:
    """Force an LLM analysis for a user via reply."""
    if not message.reply_to_message or not message.reply_to_message.from_user:
        target_user = message.from_user
    else:
        target_user = message.reply_to_message.from_user

    if target_user.is_bot:
        await message.reply("Я не можу аналізувати ботів.")
        return

    # Add to tracked list if not already there, to ensure we can analyze
    await add_tracked_user(settings.db_path, target_user.id)

    if not settings.gemini_api_key:
        await message.reply("AI-функції вимкнені (немає API ключа).")
        return

    await message.answer(f"⏳ Запускаю аналіз історії для {target_user.full_name}...")
    
    history = await get_user_history(settings.db_path, target_user.id, limit=50)
    if not history:
        await message.reply("Історія повідомлень порожня. Бот почне збирати її з цього моменту.")
        return

    profile = await summarize_user_history(history)
    if profile:
        await update_user_profile(settings.db_path, target_user.id, profile)
        await message.reply(f"✅ Аналіз завершено для {target_user.full_name}:\n\n{profile}")
    else:
        await message.reply("Не вдалося виконати аналіз. Спробуйте пізніше або перевірте API ключ.")


@router.message(Command("track"))
async def track_user_cmd(message: Message, settings: Settings) -> None:
    """Add a user to the whitelist for tracking (Owner only)."""
    if message.from_user.id != settings.owner_id:
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("Дай відповідь (reply) на повідомлення того, за ким треба стежити.")
        return

    target_user = message.reply_to_message.from_user
    await add_tracked_user(settings.db_path, target_user.id)
    await message.reply(f"🎯 Починаю стежити за {target_user.full_name}. Його повідомлення тепер будуть записуватись.")


@router.message(Command("untrack"))
async def untrack_user_cmd(message: Message, settings: Settings) -> None:
    """Remove a user from the whitelist (Owner only)."""
    if message.from_user.id != settings.owner_id:
        return

    if not message.reply_to_message or not message.reply_to_message.from_user:
        await message.reply("Дай відповідь (reply) на повідомлення того, за ким треба припинити стежити.")
        return

    target_user = message.reply_to_message.from_user
    await remove_tracked_user(settings.db_path, target_user.id)
    await message.reply(f"🚫 Більше не стежу за {target_user.full_name}.")


@router.message(F.forward_from | F.forward_sender_name)
async def handle_forwarded_message(message: Message, settings: Settings) -> None:
    """Ingest forwarded messages to build history for other users manually."""
    # If it's a forward from a real user
    if message.forward_from:
        target_id = message.forward_from.id
        text = message.text or message.caption
        if text:
            await log_user_message(settings.db_path, target_id, message.chat.id, f"[Forwarded] {text}")
            # We don't notify to keep it silent, or just a small emoji
            await message.react([{"type": "emoji", "emoji": "📥"}])
    elif message.forward_sender_name:
        # If user has privacy settings on, we can't get their ID. 
        # We could use a hash of the name, but it's not reliable for profiles.
        pass


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


def _health_listen_port() -> int:
    raw = (os.getenv("PORT") or "8080").strip()
    try:
        return int(raw)
    except ValueError:
        return 8080


class _HealthHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = (self.path or "").split("?", 1)[0]
        if path in ("/", "/health"):
            body = b"OK"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)


def _serve_health_http_forever(port: int) -> None:
    server = HTTPServer(("0.0.0.0", port), _HealthHTTPHandler)
    logger.info("Health HTTP server listening on 0.0.0.0:%s", port)
    server.serve_forever()


def start_health_server_background() -> None:
    port = _health_listen_port()
    thread = threading.Thread(
        target=_serve_health_http_forever,
        args=(port,),
        daemon=True,
        name="health-http",
    )
    thread.start()


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load variables from .env into process environment (if file exists).
    load_dotenv()

    settings = _load_settings()
    await init_db(settings.db_path)

    if settings.gemini_api_key:
        setup_gemini(settings.gemini_api_key)
        logger.info("Gemini LLM initialized.")
    else:
        logger.warning("Gemini API key not found. Personalized text replies will be disabled.")

    start_health_server_background()

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher(storage=MemoryStorage())

    dp.update.outer_middleware(SettingsMiddleware(settings))
    router.message.outer_middleware(RandomVoiceReplyMiddleware(settings))
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

