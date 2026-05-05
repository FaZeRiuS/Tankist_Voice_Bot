import aiosqlite
import difflib
import re
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS voice_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  file_id TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_voice_samples_title ON voice_samples(title);
""".strip()


_WORD_RE = re.compile(r"[\\w]+", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower())).strip()


async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA_SQL)
        await db.commit()


async def add_voice_sample(db_path: str, title: str, file_id: str) -> int:
    title = title.strip()
    file_id = file_id.strip()
    if not title:
        raise ValueError("title must be non-empty")
    if not file_id:
        raise ValueError("file_id must be non-empty")

    async with aiosqlite.connect(db_path) as db:
        cur = await db.execute(
            "INSERT INTO voice_samples (title, file_id) VALUES (?, ?)",
            (title, file_id),
        )
        await db.commit()
        return int(cur.lastrowid)


async def get_random_voice_sample(db_path: str) -> dict[str, Any] | None:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT id, title, file_id FROM voice_samples ORDER BY RANDOM() LIMIT 1"
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)


async def search_voice_samples(
    db_path: str,
    query: str,
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Two-stage fuzzy search:
    1) SQLite LIKE prefilter for speed
    2) Python ranking via SequenceMatcher ratio
    """
    limit = max(1, min(int(limit), 50))
    offset = max(0, int(offset))
    raw_query = (query or "").strip()
    norm_query = _normalize(raw_query)

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row

        if not norm_query:
            cur = await db.execute(
                "SELECT id, title, file_id FROM voice_samples ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

        tokens = [t for t in norm_query.split() if t]
        like_patterns = [f"%{t}%" for t in tokens[:5]]  # cap tokens to keep query simple
        where = " AND ".join(["LOWER(title) LIKE ?"] * len(like_patterns)) or "1=1"

        # Overfetch a bit; ranking will trim. Also account for page offset.
        prefetch = max((offset + limit) * 6, 120)
        prefetch = min(prefetch, 1000)
        sql = f"""
            SELECT id, title, file_id
            FROM voice_samples
            WHERE {where}
            ORDER BY id DESC
            LIMIT ?
        """.strip()

        cur = await db.execute(sql, (*like_patterns, prefetch))
        candidates = [dict(r) for r in await cur.fetchall()]

    def score(item: dict[str, Any]) -> float:
        title = str(item.get("title") or "")
        norm_title = _normalize(title)
        # Blend title similarity with substring bonus.
        ratio = difflib.SequenceMatcher(None, norm_query, norm_title).ratio()
        bonus = 0.15 if norm_query and norm_query in norm_title else 0.0
        return ratio + bonus

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[offset : offset + limit]
