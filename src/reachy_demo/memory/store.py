"""SQLite memory store: schema, migrations and every query. No LLM, no network.

Design: docs/superpowers/specs/2026-11-17-memory-design.md §6. Times are UTC epoch seconds.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

log = logging.getLogger(__name__)

OWNER_ID = 1
Clock = Callable[[], float]
FOREVER = 1e18


@dataclass(frozen=True)
class Speaker:
    person_id: int
    role: Literal["owner", "guest"]


OWNER = Speaker(OWNER_ID, "owner")


@dataclass(frozen=True)
class Person:
    """`people` is the single source of truth for a person's name. Each thing known about them
    is a fact; the owner also gets `description`, a short profile rewritten from those facts."""

    id: int
    role: str
    name: str | None
    aliases: list[str]  # earlier names, newest last
    description: str = ""


@dataclass(frozen=True)
class Fact:
    id: int
    subject_person_id: int | None
    subject_text: str
    text: str
    text_original: str | None
    category: str
    source: str
    importance: float
    sensitive: bool
    valid_from: float | None
    valid_to: float | None
    created: float
    expired: float | None
    last_used: float | None
    use_count: int


@dataclass(frozen=True)
class Episode:
    id: int
    session_id: str | None
    kind: str
    start: float
    end: float
    summary: str
    topics: list[str]
    importance: float


@dataclass(frozen=True)
class FollowUp:
    id: int
    person_id: int
    text: str
    due: float | None
    created: float


SCHEMA_V1 = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE people(
  id INTEGER PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('owner','guest')),
  name TEXT, aliases_json TEXT NOT NULL DEFAULT '[]', language TEXT,
  consent_recognize INTEGER NOT NULL DEFAULT 0, created REAL, last_seen REAL,
  retention_until REAL);
CREATE TABLE facts(
  id INTEGER PRIMARY KEY, subject_person_id INTEGER REFERENCES people(id) ON DELETE CASCADE,
  subject_text TEXT NOT NULL, text TEXT NOT NULL, text_original TEXT NOT NULL DEFAULT '',
  category TEXT NOT NULL DEFAULT 'other'
    CHECK(category IN ('bio','preference','relation','plan','setting','other')),
  source TEXT NOT NULL CHECK(source IN ('said','seen','inferred')),
  confidence REAL NOT NULL DEFAULT 1.0, importance REAL NOT NULL DEFAULT 0.5,
  sensitive INTEGER NOT NULL DEFAULT 0, shared INTEGER NOT NULL DEFAULT 0,
  valid_from REAL, valid_to REAL, created REAL NOT NULL, expired REAL,
  replaced_by INTEGER REFERENCES facts(id) ON DELETE SET NULL,
  last_used REAL, use_count INTEGER NOT NULL DEFAULT 0,
  evidence_json TEXT NOT NULL DEFAULT '[]');
CREATE INDEX facts_person ON facts(subject_person_id, expired);
CREATE VIRTUAL TABLE facts_fts USING fts5(text, subject_text, text_original,
  content='facts', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, text, subject_text, text_original)
  VALUES (new.id, new.text, new.subject_text, new.text_original);
END;
CREATE TRIGGER facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, text, subject_text, text_original)
  VALUES ('delete', old.id, old.text, old.subject_text, old.text_original);
END;
CREATE TRIGGER facts_au AFTER UPDATE OF text, subject_text, text_original ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, text, subject_text, text_original)
  VALUES ('delete', old.id, old.text, old.subject_text, old.text_original);
  INSERT INTO facts_fts(rowid, text, subject_text, text_original)
  VALUES (new.id, new.text, new.subject_text, new.text_original);
END;
CREATE TABLE episodes(
  id INTEGER PRIMARY KEY, session_id TEXT,
  kind TEXT NOT NULL CHECK(kind IN ('conversation','task','visual','moment')),
  start REAL NOT NULL, end REAL NOT NULL, person_ids_json TEXT NOT NULL DEFAULT '[]',
  summary TEXT NOT NULL, topics_json TEXT NOT NULL DEFAULT '[]',
  importance REAL NOT NULL DEFAULT 0.5,
  outcome TEXT CHECK(outcome IN ('success','failure','partial')), details_json TEXT,
  compressed_level INTEGER NOT NULL DEFAULT 0, source_event_max INTEGER);
CREATE INDEX episodes_start ON episodes(start);
CREATE UNIQUE INDEX episodes_session ON episodes(session_id) WHERE session_id IS NOT NULL;
CREATE VIRTUAL TABLE episodes_fts USING fts5(summary, topics_json,
  content='episodes', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER episodes_ai AFTER INSERT ON episodes BEGIN
  INSERT INTO episodes_fts(rowid, summary, topics_json)
  VALUES (new.id, new.summary, new.topics_json);
END;
CREATE TRIGGER episodes_ad AFTER DELETE ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, summary, topics_json)
  VALUES ('delete', old.id, old.summary, old.topics_json);
END;
CREATE TABLE follow_ups(
  id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  text TEXT NOT NULL, due REAL, created REAL NOT NULL,
  episode_id INTEGER REFERENCES episodes(id) ON DELETE SET NULL, done REAL);
CREATE INDEX follow_ups_open ON follow_ups(person_id, done);
CREATE TABLE event_log(
  id INTEGER PRIMARY KEY, t REAL NOT NULL, kind TEXT NOT NULL, person_id INTEGER,
  session_id TEXT, payload_json TEXT NOT NULL, consolidated INTEGER NOT NULL DEFAULT 0);
CREATE INDEX event_log_t ON event_log(t);
CREATE INDEX event_log_session ON event_log(session_id);
"""
OWN_NAME = (
    r"(?:'s|’s)? (?:name is|is called|is named|goes by|wants to be called)\s+(?P<name>[^.,;!?]+)"
)


def own_name_stated(text: str, *names: str | None) -> str | None:
    """The name in a fact about the speaker's own name ('The owner's name is Ha' → 'Ha')."""
    who = "|".join(re.escape(n) for n in ("the owner", "owner", *names) if n)
    m = re.match(rf"\s*(?:{who}){OWN_NAME}", text, re.IGNORECASE)
    return m["name"].strip().strip("\"'") if m else None


def own_name_fact(text: str, *names: str | None) -> bool:
    return own_name_stated(text, *names) is not None


def _v2_identity_in_people(db: sqlite3.Connection, now: float) -> None:
    """One source of truth: names only in `people`, language only as a fact.

    Moves each person's language into a 'setting' fact and drops the column, then turns
    'the owner's name is X' facts into people.name / aliases and closes them.
    """
    langs = {"vi": "Vietnamese", "en": "English"}
    for row in db.execute(
        "SELECT id, name, language FROM people WHERE language IS NOT NULL"
    ).fetchall():
        if lang := langs.get(row["language"]):
            who = row["name"] or "The owner"
            db.execute(
                "INSERT INTO facts(subject_person_id, subject_text, text, category, source, "
                "importance, created) VALUES (?, ?, ?, 'setting', 'said', 0.8, ?)",
                (row["id"], who, f"{who} wants answers in {lang}.", now),
            )
    db.execute("ALTER TABLE people DROP COLUMN language")
    for p in db.execute("SELECT id, name, aliases_json FROM people").fetchall():
        aliases: list[str] = json.loads(p["aliases_json"])
        stated: list[str] = []
        facts = db.execute(
            "SELECT id, text FROM facts WHERE subject_person_id=? AND expired IS NULL "
            "ORDER BY created",
            (p["id"],),
        ).fetchall()
        for f in facts:
            if n := own_name_stated(f["text"], p["name"], *aliases):
                stated.append(n)
                db.execute("UPDATE facts SET expired=? WHERE id=?", (now, f["id"]))
        name = p["name"] or (stated[-1] if stated else None)
        for n in stated:
            if name and n.lower() != name.lower() and n.lower() not in map(str.lower, aliases):
                aliases.append(n)
        db.execute(
            "UPDATE people SET name=?, aliases_json=? WHERE id=?",
            (name, json.dumps(aliases, ensure_ascii=False), p["id"]),
        )


Migration = str | Callable[[sqlite3.Connection, float], None]
V3_DESCRIPTION = "ALTER TABLE people ADD COLUMN description TEXT NOT NULL DEFAULT '';"
V4_VECTORS = """
CREATE TABLE fact_vecs(
  fact_id INTEGER NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
  model TEXT NOT NULL, vec BLOB NOT NULL, PRIMARY KEY(fact_id, model));
CREATE TABLE episode_vecs(
  episode_id INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
  model TEXT NOT NULL, vec BLOB NOT NULL, PRIMARY KEY(episode_id, model));
"""
MIGRATIONS: list[Migration] = [SCHEMA_V1, _v2_identity_in_people, V3_DESCRIPTION, V4_VECTORS]
VecKind = Literal["fact", "episode"]
_VEC_TABLE = {"fact": ("fact_vecs", "fact_id"), "episode": ("episode_vecs", "episode_id")}

# Small EN + VN list: words that would make every fact match. Not exhaustive on purpose.
_STOPWORDS_TEXT = """
    a an the and or but of to in on at for from with by about as is are was were be been
    am do does did have has had i me my mine you your yours he she it its we our they their
    them this that these those what which who whom whose when where why how there here
    not no yes can could would should will just also very really so than then too
    tell told know remember recall say said ask asked
    tôi tao mình bạn anh chị em nó của là và hay hoặc thì mà có không đã đang sẽ được bị
    với cho này kia đó ấy gì nào đâu sao như cái những các một rồi nhé nha ạ à ơi
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def match_query(text: str) -> str | None:
    """FTS5 MATCH string: content words OR-ed, each quoted, so user text can't inject syntax."""
    seen: list[str] = []
    for word in re.findall(r"\w+", text.lower()):
        if len(word) > 1 and word not in STOPWORDS and word not in seen:
            seen.append(word)
    return " OR ".join(f'"{w}"' for w in seen) if seen else None


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 1 and w not in STOPWORDS}


class MemoryStore:
    def __init__(self, path: str | Path, clock: Clock = time.time) -> None:
        self.path = Path(path) if str(path) != ":memory:" else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._db = sqlite3.connect(str(path), isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        if self.path:
            self._db.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        self._db.execute(
            "INSERT OR IGNORE INTO people(id, role, created) VALUES (?, 'owner', ?)",
            (OWNER_ID, self._clock()),
        )

    # -- lifecycle ---------------------------------------------------------------------------

    def _migrate(self) -> None:
        has_meta = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        version = 0
        if has_meta:
            row = self._db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            version = int(row[0]) if row else 0
        for n in range(version, len(MIGRATIONS)):
            if self.path and version > 0:
                shutil.copyfile(self.path, self.path.with_name(f"{self.path.name}.bak-{n}"))
            step = MIGRATIONS[n]
            done = f"INSERT OR REPLACE INTO meta VALUES ('schema_version', '{n + 1}')"  # noqa: S608
            try:
                if isinstance(step, str):  # a script: executescript runs its own transaction
                    self._db.executescript(f"BEGIN;\n{step}\n{done};\nCOMMIT;")
                else:
                    self._db.execute("BEGIN")
                    step(self._db, self._clock())
                    self._db.execute(done)
                    self._db.execute("COMMIT")
            except sqlite3.Error:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise
            log.info("memory schema migrated", extra={"schema_version": n + 1})

    @property
    def schema_version(self) -> int:
        row = self._db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row[0])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._db.in_transaction:  # nested: the outer one commits
            yield
            return
        self._db.execute("BEGIN")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def backup(self, dest: str | Path) -> None:
        target = sqlite3.connect(str(dest))
        try:
            self._db.backup(target)
        finally:
            target.close()

    def close(self) -> None:
        self._db.close()

    # -- people ------------------------------------------------------------------------------

    def person(self, person_id: int) -> Person | None:
        row = self._db.execute(
            "SELECT id, role, name, aliases_json, description FROM people WHERE id=?", (person_id,)
        ).fetchone()
        return _person(row) if row else None

    def find_person(self, name: str) -> Person | None:
        """By current or earlier name, ignoring case."""
        key = name.strip().lower()
        for row in self._db.execute("SELECT id, role, name, aliases_json, description FROM people"):
            p = _person(row)
            if any(n.lower() == key for n in [p.name or "", *p.aliases] if n):
                return p
        return None

    def set_name(self, person_id: int, name: str) -> None:
        """The newest name wins; the one it replaces is kept as an alias (earlier name)."""
        p = self.person(person_id)
        if p is None:
            return
        aliases = [a for a in p.aliases if a.lower() != name.lower()]
        if p.name and p.name.lower() != name.lower() and p.name not in aliases:
            aliases.append(p.name)
        self._db.execute(
            "UPDATE people SET name=?, aliases_json=? WHERE id=?",
            (name, json.dumps(aliases, ensure_ascii=False), person_id),
        )

    def set_description(self, person_id: int, description: str) -> None:
        self._db.execute(
            "UPDATE people SET description=? WHERE id=?", (description.strip(), person_id)
        )

    # -- event log ---------------------------------------------------------------------------

    def log_event(
        self, kind: str, payload: dict[str, Any], *, person_id: int | None, session_id: str | None
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO event_log(t, kind, person_id, session_id, payload_json) "
            "VALUES (?,?,?,?,?)",
            (self._clock(), kind, person_id, session_id, json.dumps(payload, ensure_ascii=False)),
        )
        return int(cur.lastrowid or 0)

    def mark_consolidated(self, event_id: int) -> None:
        self._db.execute("UPDATE event_log SET consolidated=1 WHERE id=?", (event_id,))

    def unconsolidated_turns(
        self, since: float, limit: int
    ) -> list[tuple[int, int | None, dict[str, Any]]]:
        rows = self._db.execute(
            "SELECT id, person_id, payload_json FROM event_log "
            "WHERE kind='turn' AND consolidated=0 AND t>=? ORDER BY id DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        return [(r["id"], r["person_id"], json.loads(r["payload_json"])) for r in reversed(rows)]

    def session_turns(self, session_id: str) -> list[tuple[float, dict[str, Any]]]:
        rows = self._db.execute(
            "SELECT t, payload_json FROM event_log WHERE kind='turn' AND session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [(r["t"], json.loads(r["payload_json"])) for r in rows]

    def last_turns(self, n: int) -> list[tuple[float, dict[str, Any]]]:
        rows = self._db.execute(
            "SELECT t, payload_json FROM event_log WHERE kind='turn' ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [(r["t"], json.loads(r["payload_json"])) for r in reversed(rows)]

    def last_turn_time(self, person_id: int | None = None) -> float | None:
        sql = "SELECT max(t) FROM event_log WHERE kind='turn'"
        args: tuple[Any, ...] = ()
        if person_id is not None:
            sql += " AND person_id=?"
            args = (person_id,)
        row = self._db.execute(sql, args).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def sessions_without_episode(self, exclude: str | None) -> list[tuple[str, int | None]]:
        rows = self._db.execute(
            "SELECT session_id, max(person_id) AS person_id FROM event_log "
            "WHERE kind='turn' AND session_id IS NOT NULL AND session_id IS NOT ? "
            "AND session_id NOT IN (SELECT session_id FROM episodes WHERE session_id IS NOT NULL) "
            "GROUP BY session_id ORDER BY min(id)",
            (exclude,),
        ).fetchall()
        return [(r["session_id"], r["person_id"]) for r in rows]

    # -- facts -------------------------------------------------------------------------------

    def add_fact(
        self,
        *,
        subject_person_id: int | None,
        subject_text: str,
        text: str,
        text_original: str = "",
        category: str = "other",
        source: str = "said",
        importance: float = 0.5,
        sensitive: bool = False,
        valid_from: float | None = None,
        valid_to: float | None = None,
        evidence: list[int] | None = None,
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO facts(subject_person_id, subject_text, text, text_original, category, "
            "source, importance, sensitive, valid_from, valid_to, created, evidence_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                subject_person_id,
                subject_text,
                text,
                text_original,
                category,
                source,
                min(max(importance, 0.0), 1.0),
                int(sensitive),
                valid_from,
                valid_to,
                self._clock(),
                json.dumps(evidence or []),
            ),
        )
        return int(cur.lastrowid or 0)

    def fact(self, fact_id: int) -> Fact | None:
        row = self._db.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        return _fact(row) if row else None

    def update_fact(self, fact_id: int, text: str, text_original: str | None = None) -> None:
        self._db.execute("DELETE FROM fact_vecs WHERE fact_id=?", (fact_id,))  # old wording
        if text_original is None:
            self._db.execute("UPDATE facts SET text=? WHERE id=?", (text, fact_id))
        else:
            self._db.execute(
                "UPDATE facts SET text=?, text_original=? WHERE id=?",
                (text, text_original, fact_id),
            )

    def invalidate_fact(
        self, fact_id: int, *, valid_to: float | None, replaced_by: int | None
    ) -> None:
        now = self._clock()
        self._db.execute(
            "UPDATE facts SET expired=?, valid_to=coalesce(?, valid_to, ?), replaced_by=? "
            "WHERE id=?",
            (now, valid_to, now, replaced_by, fact_id),
        )

    def delete_facts(self, ids: list[int]) -> int:
        """Hard delete, with the logged turns each fact came from (privacy wins over history)."""
        n = 0
        with self.transaction():
            for i in ids:
                row = self._db.execute(
                    "SELECT evidence_json FROM facts WHERE id=?", (i,)
                ).fetchone()
                if row is None:
                    continue
                for eid in json.loads(row[0]):
                    self._db.execute("DELETE FROM event_log WHERE id=?", (eid,))
                self._db.execute("DELETE FROM facts WHERE id=?", (i,))
                n += 1
        return n

    def touch_facts(self, ids: list[int]) -> None:
        now = self._clock()
        self._db.executemany(
            "UPDATE facts SET last_used=?, use_count=use_count+1 WHERE id=?",
            [(now, i) for i in ids],
        )

    def search_facts(
        self,
        query: str,
        speaker: Speaker,
        *,
        person_id: int | None = None,
        subject_text: str | None = None,
        start: float | None = None,
        end: float | None = None,
        limit: int = 5,
    ) -> list[tuple[Fact, float]]:
        """Best FTS matches, as (fact, bm25 rank; lower is better). Scoped to the speaker.

        Without a time window only current facts match. With one, facts whose validity
        overlaps it match too, expired or not ("where did I live last year?").
        """
        match = match_query(query)
        if match is None:
            return []
        sql = (
            "SELECT f.*, bm25(facts_fts) AS rank FROM facts_fts "
            "JOIN facts f ON f.id=facts_fts.rowid WHERE facts_fts MATCH ?"
        )
        args: list[Any] = [match]
        sql, args = _fact_filters(sql, args, speaker, person_id, subject_text, start, end)
        sql += " ORDER BY rank LIMIT ?"
        args.append(limit)
        return [(_fact(r), float(r["rank"])) for r in self._db.execute(sql, args)]

    def current_facts(self, speaker: Speaker, person_id: int) -> list[Fact]:
        """All current, non-sensitive facts of one person (the briefing ranks them in Python)."""
        sql = (
            "SELECT f.* FROM facts f "
            "WHERE f.expired IS NULL AND f.sensitive=0 AND f.subject_person_id=?"
        )
        sql, args = _scope(sql, [person_id], speaker)
        return [_fact(r) for r in self._db.execute(sql, args)]

    # -- episodes ----------------------------------------------------------------------------

    def add_episode(
        self,
        *,
        session_id: str | None,
        start: float,
        end: float,
        person_ids: list[int],
        summary: str,
        topics: list[str],
        importance: float = 0.5,
        kind: str = "conversation",
        source_event_max: int | None = None,
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO episodes(session_id, kind, start, end, person_ids_json, summary, "
            "topics_json, importance, source_event_max) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                kind,
                start,
                end,
                json.dumps(person_ids),
                summary,
                json.dumps(topics, ensure_ascii=False),
                min(max(importance, 0.0), 1.0),
                source_event_max,
            ),
        )
        return int(cur.lastrowid or 0)

    def recent_episodes(self, person_id: int, n: int, kind: str = "conversation") -> list[Episode]:
        rows = self._db.execute(
            "SELECT * FROM episodes WHERE kind=? AND importance > 0 AND "  # noqa: S608 - constant
            + _HAS_PERSON
            + " ORDER BY start DESC LIMIT ?",  # importance 0 = a greeting only
            (kind, person_id, n),
        ).fetchall()
        return [_episode(r) for r in rows]

    def search_episodes(
        self,
        query: str,
        speaker: Speaker,
        *,
        start: float | None = None,
        end: float | None = None,
        limit: int = 5,
    ) -> list[tuple[Episode, float]]:
        """FTS matches; with a time window and no matching words, the episodes in that window."""
        match = match_query(query)
        args: list[Any] = []
        if match is not None:
            sql = (
                "SELECT e.*, bm25(episodes_fts) AS rank FROM episodes_fts "
                "JOIN episodes e ON e.id=episodes_fts.rowid WHERE episodes_fts MATCH ?"
            )
            args.append(match)
        else:
            sql = "SELECT e.*, 0.0 AS rank FROM episodes e WHERE 1"
        if match is None and (start is None or end is None):
            return []
        sql, args = _episode_filters(sql, args, speaker, start, end)
        sql += " ORDER BY rank, e.start DESC LIMIT ?"
        args.append(limit)
        rows = self._db.execute(sql, args).fetchall()
        if not rows and match is not None and start is not None and end is not None:
            return self.search_episodes("", speaker, start=start, end=end, limit=limit)
        return [(_episode(r), float(r["rank"])) for r in rows]

    # -- vectors -----------------------------------------------------------------------------

    def set_vec(self, kind: VecKind, item_id: int, model: str, vec: np.ndarray) -> None:
        table, col = _VEC_TABLE[kind]
        self._db.execute(
            f"INSERT OR REPLACE INTO {table}({col}, model, vec) VALUES (?,?,?)",  # noqa: S608
            (item_id, model, np.asarray(vec, dtype="<f4").tobytes()),
        )

    def load_vecs(self, kind: VecKind, model: str) -> list[tuple[int, np.ndarray]]:
        table, col = _VEC_TABLE[kind]
        rows = self._db.execute(
            f"SELECT {col}, vec FROM {table} WHERE model=? ORDER BY {col}",  # noqa: S608
            (model,),
        ).fetchall()
        return [(int(r[0]), np.frombuffer(r[1], dtype="<f4").copy()) for r in rows]

    def missing_vecs(self, kind: VecKind, model: str, limit: int) -> list[tuple[int, str]]:
        """Rows without a vector for `model`: current facts, or episode summaries."""
        if kind == "fact":
            sql = (
                "SELECT id, text FROM facts WHERE expired IS NULL AND id NOT IN "
                "(SELECT fact_id FROM fact_vecs WHERE model=?) ORDER BY id LIMIT ?"
            )
        else:
            sql = (
                "SELECT id, summary FROM episodes WHERE id NOT IN "
                "(SELECT episode_id FROM episode_vecs WHERE model=?) ORDER BY id LIMIT ?"
            )
        return [(int(r[0]), str(r[1])) for r in self._db.execute(sql, (model, limit))]

    def allowed_ids(
        self,
        kind: VecKind,
        speaker: Speaker,
        *,
        person_id: int | None = None,
        subject_text: str | None = None,
        start: float | None = None,
        end: float | None = None,
        include_sensitive: bool = True,
    ) -> set[int]:
        """Ids the speaker may see under the same filters as keyword search (no text match).

        Facts: current only unless a time window is given, as in `search_facts`.
        """
        args: list[Any] = []
        if kind == "fact":
            sql = "SELECT f.id FROM facts f WHERE 1"
            if not include_sensitive:
                sql += " AND f.sensitive=0"
            sql, args = _fact_filters(sql, args, speaker, person_id, subject_text, start, end)
        else:
            sql = "SELECT e.id FROM episodes e WHERE e.importance > 0"
            sql, args = _episode_filters(sql, args, speaker, start, end)
            if person_id is not None:
                sql += " AND " + _HAS_PERSON.replace("person_ids_json", "e.person_ids_json")
                args.append(person_id)
        return {int(r[0]) for r in self._db.execute(sql, args)}

    def facts_by_ids(self, ids: list[int]) -> list[Fact]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(f"SELECT * FROM facts WHERE id IN ({marks})", ids)  # noqa: S608
        found = {r["id"]: _fact(r) for r in rows}
        return [found[i] for i in ids if i in found]

    def episodes_by_ids(self, ids: list[int]) -> list[Episode]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(f"SELECT * FROM episodes WHERE id IN ({marks})", ids)  # noqa: S608
        found = {r["id"]: _episode(r) for r in rows}
        return [found[i] for i in ids if i in found]

    # -- follow-ups --------------------------------------------------------------------------

    def add_follow_up(
        self, person_id: int, text: str, due: float | None, episode_id: int | None = None
    ) -> int:
        """Add one, unless an open follow-up says nearly the same (then return that one's id).

        The turn extractor and the session summary often propose the same follow-up.
        """
        words = content_words(text)
        for fu in self.all_open_follow_ups(person_id):
            other = content_words(fu.text)
            if words and other and len(words & other) / len(words | other) >= 0.6:
                return fu.id
        cur = self._db.execute(
            "INSERT INTO follow_ups(person_id, text, due, created, episode_id) VALUES (?,?,?,?,?)",
            (person_id, text, due, self._clock(), episode_id),
        )
        return int(cur.lastrowid or 0)

    def open_follow_ups(self, person_id: int, now: float, limit: int = 3) -> list[FollowUp]:
        """Not done, and due within ±2 days (or undated and made in the last 7 days)."""
        day = 86400.0
        rows = self._db.execute(
            "SELECT id, person_id, text, due, created FROM follow_ups "
            "WHERE person_id=? AND done IS NULL "
            "AND ((due IS NOT NULL AND due BETWEEN ? AND ?) OR (due IS NULL AND created >= ?)) "
            "ORDER BY coalesce(due, created) LIMIT ?",
            (person_id, now - 2 * day, now + 2 * day, now - 7 * day, limit),
        ).fetchall()
        return [FollowUp(r["id"], r["person_id"], r["text"], r["due"], r["created"]) for r in rows]

    def all_open_follow_ups(self, person_id: int) -> list[FollowUp]:
        rows = self._db.execute(
            "SELECT id, person_id, text, due, created FROM follow_ups "
            "WHERE person_id=? AND done IS NULL",
            (person_id,),
        ).fetchall()
        return [FollowUp(r["id"], r["person_id"], r["text"], r["due"], r["created"]) for r in rows]

    def close_follow_up(self, follow_up_id: int) -> None:
        self._db.execute("UPDATE follow_ups SET done=? WHERE id=?", (self._clock(), follow_up_id))

    # -- forget ------------------------------------------------------------------------------

    def forget_person(self, person_id: int) -> None:
        """Hard delete of everything about one person; keeps the row (name cleared)."""
        with self.transaction():
            self._db.execute("DELETE FROM facts WHERE subject_person_id=?", (person_id,))
            self._db.execute("DELETE FROM follow_ups WHERE person_id=?", (person_id,))
            self._db.execute("DELETE FROM episodes WHERE " + _HAS_PERSON, (person_id,))  # noqa: S608
            self._db.execute("DELETE FROM event_log WHERE person_id=?", (person_id,))
            self._db.execute(
                "UPDATE people SET name=NULL, aliases_json='[]' WHERE id=?",
                (person_id,),
            )


_HAS_PERSON = "EXISTS (SELECT 1 FROM json_each(person_ids_json) WHERE value = ?)"


def _fact_filters(
    sql: str,
    args: list[Any],
    speaker: Speaker,
    person_id: int | None,
    subject_text: str | None,
    start: float | None,
    end: float | None,
) -> tuple[str, list[Any]]:
    """Scope, person, subject and validity clauses on `facts f`, shared by search and vectors."""
    sql, args = _scope(sql, args, speaker)
    if person_id is not None:
        sql += " AND f.subject_person_id=?"
        args = [*args, person_id]
    if subject_text:
        sql += " AND f.subject_text LIKE ?"
        args = [*args, f"%{subject_text}%"]
    if start is None or end is None:
        sql += " AND f.expired IS NULL"
    else:
        sql += (
            " AND coalesce(f.valid_from, f.created) < ? AND coalesce(f.valid_to, f.expired, ?) >= ?"
        )
        args = [*args, end, FOREVER, start]
    return sql, args


def _episode_filters(
    sql: str, args: list[Any], speaker: Speaker, start: float | None, end: float | None
) -> tuple[str, list[Any]]:
    """Guest scope and time window on `episodes e`, shared by search and vectors."""
    if speaker.role != "owner":
        sql += " AND " + _HAS_PERSON.replace("person_ids_json", "e.person_ids_json")
        args = [*args, speaker.person_id]
    if start is not None and end is not None:
        sql += " AND e.start < ? AND e.end >= ?"
        args = [*args, end, start]
    return sql, args


def _scope(sql: str, args: list[Any], speaker: Speaker) -> tuple[str, list[Any]]:
    """Owner reads everything; a guest reads only their own facts and shared ones."""
    if speaker.role == "owner":
        return sql, args
    return sql + " AND (f.subject_person_id=? OR f.shared=1)", [*args, speaker.person_id]


def _person(r: sqlite3.Row) -> Person:
    return Person(
        r["id"], r["role"], r["name"], json.loads(r["aliases_json"]), r["description"] or ""
    )


def _fact(r: sqlite3.Row) -> Fact:
    return Fact(
        id=r["id"],
        subject_person_id=r["subject_person_id"],
        subject_text=r["subject_text"],
        text=r["text"],
        text_original=r["text_original"] or None,
        category=r["category"],
        source=r["source"],
        importance=r["importance"],
        sensitive=bool(r["sensitive"]),
        valid_from=r["valid_from"],
        valid_to=r["valid_to"],
        created=r["created"],
        expired=r["expired"],
        last_used=r["last_used"],
        use_count=r["use_count"],
    )


def _episode(r: sqlite3.Row) -> Episode:
    return Episode(
        id=r["id"],
        session_id=r["session_id"],
        kind=r["kind"],
        start=r["start"],
        end=r["end"],
        summary=r["summary"],
        topics=json.loads(r["topics_json"]),
        importance=r["importance"],
    )
