# Memory Phase 2: objects and places

Date: 2026-11-19. Status: draft, waiting for the spike (§11).
Replaces §8 of [2026-11-17-memory-design.md](2026-11-17-memory-design.md) and its P2 tables in §6.

## Intent

Reachy remembers where things are in the room. You say "learn the room" and it looks around once. It names the landmarks it sees ("desk", "couch", "shelf") and the objects on them ("phone on the desk"). After that, every search keeps the memory up to date. "Where did I leave my phone?" is answered from memory without moving: "on the desk, 5 minutes ago". "Find my phone" looks at the remembered spot first. Objects are open-vocabulary: keys and earbuds count too, not only the 80 COCO classes.

## Goals

- **Room scan** (`learn_room`, owner only): each stop is described by the vision model (entities, relations, region), and YOLOE confirms what it says. One text LLM call merges the stops. Landmarks and sightings are then written to SQLite.
- **find_object reads memory first**: any valid sighting, whatever its age → look exactly there and check one frame; not there → mark it invalid, then full sweep. No sighting → full sweep. Age only changes what Reachy says.
- **find_object writes memory**: a found object is a sighting, a confirmed one becomes certain, a rejected one is deleted, and a remembered spot found empty is marked invalid.
- **Open-vocabulary find**: the sweep uses YOLOE with the user's noun and the vision model checks the hits. YOLOX stays for object tracking only.
- **Task episodes**: every search, found or not, writes `episodes(kind='task')` listing the spots it checked. "Did you find my keys earlier?" is answered from that. The next search checks those spots last.
- **recall** gains `kind: "sighting" | "task"`.
- A test for every branch; the policy runs on a fake clock. Usage docs are updated.

## Non-goals

- Realtime or idle watching of the room. Memory changes only through a scan or a search. (Phase 3, CPU permitting.)
- Storing images or thumbnails. Memory holds text and angles only.
- Object-to-object relations ("mug next to laptop") as queryable edges. The vision model's text keeps them in `where_text`, but there is no `relations` table.
- "Usual place" ("you usually leave it on the desk"). The data supports it later; it is not built here.
- Embedding match for synonyms ("smartphone" vs "phone"). See §6.4.
- Depth or 3D positions. Positions are bearings from a robot that doesn't move.
- Replacing YOLOX in `track.py`.

## Decisions taken in discussion

| Decision | Why |
|---|---|
| Hybrid source: one scan for landmarks, tasks for objects, no realtime | Landmarks rarely change. Small objects change hourly. Realtime costs CPU the laptop doesn't have. |
| The vision model proposes entities, YOLOE confirms them | The vision model is open-vocabulary and gives relations, but its boxes are unreliable and it makes up small objects. YOLOE gives exact boxes. |
| Landmarks YOLOE misses are kept at medium confidence; small objects YOLOE misses are dropped | Open-vocabulary detectors miss large, cut-off or background things (desk at the frame edge, a shelf, a door). The vision model rarely makes up furniture, but often makes up small objects. |
| The merge LLM returns ID groups, never angles | LLMs get numbers wrong. Code computes bearings. |
| The memory lookup is code (`memory/spatial.py`), not an LLM call | Fast (< 50 ms), deterministic, testable with a fake clock. |
| Invalidate (`valid_to`), don't delete, except a sighting the user rejects | Keeps history for "did you find it earlier?" and a later "usual place". A rejected one was never true. |
| One memory rule, no age tiers: any valid sighting → look there first | Age tiers (10 min / 1 day / older) all ended up looking at the old spot first anyway, since the nearest sweep stop always has it in view. One branch instead of three; age only changes the wording. |
| Spots that failed searches checked are searched last, not skipped | The object may have been put there since. Same speed in the normal case, never a miss. |
| YOLOE via `ultralytics` (AGPL-3.0) | Chosen by the owner; license accepted. Recorded in NOTICE. |
| Memory recall of places and tasks is owner-only | Same as `find_object`. Guests can't ask where the owner's things are. |

## 1. Components

```
perception/open_vocab.py   OpenVocabDetector: YOLOE, text prompts, the Detects protocol
attention/search.py        Search: YOLOE sweep, memory hooks, plan from SpatialMemory
attention/scan.py          RoomScan: stops -> vision model -> YOLOE -> merge -> write
memory/spatial.py          SpatialMemory (store + wall clock) and pure policy functions
memory/store.py            V5_SPATIAL migration, sighting/landmark/task-episode queries
memory/prompts.py          SCAN_STOP and SCAN_MERGE prompts
memory/recall.py           kind="sighting" | "task"
agent/tools/find.py        find_object (object noun), learn_room, name_place
```

Dependencies go into constructors, as today. `Search` and `RoomScan` take `spatial: SpatialMemory | None`; `None` (MEMORY=off, DB failed) means today's behaviour with no memory. `Search` never sees SQL.

## 2. Schema: `V5_SPATIAL`

Appended to `MIGRATIONS` in `store.py` as one script. The existing `_migrate` backs up the DB first.

```sql
CREATE TABLE landmarks(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,                 -- what Reachy says: "desk", "my desk" after name_place
  noun TEXT NOT NULL,                 -- short noun for YOLOE: "desk"
  yaw REAL NOT NULL, pitch REAL NOT NULL,   -- world degrees (yaw + = left, pitch + = down)
  radius REAL NOT NULL DEFAULT 20,    -- degrees; a sighting within it is "at" this landmark
  confidence REAL NOT NULL,           -- 1.0 user, 0.9 boxed, 0.6 region only
  source TEXT NOT NULL CHECK(source IN ('scan','user')),
  created REAL NOT NULL, updated REAL NOT NULL);
CREATE UNIQUE INDEX landmarks_name ON landmarks(name);

CREATE TABLE sightings(
  id INTEGER PRIMARY KEY,
  noun TEXT NOT NULL,                 -- "phone", "keys": the join key with find_object
  target TEXT,                        -- the user's words: "my iPhone"
  description TEXT,                   -- "black phone in a blue case"
  yaw REAL NOT NULL, pitch REAL NOT NULL,
  landmark_id INTEGER REFERENCES landmarks(id) ON DELETE SET NULL,
  relation TEXT CHECK(relation IN ('on','in','under','next_to')),
  where_text TEXT NOT NULL,           -- "on the desk" | "about 40° to my left"
  confidence REAL NOT NULL,
  seen REAL NOT NULL,                 -- wall clock (store clock), never monotonic
  valid_to REAL,                      -- set when found not there / superseded
  source TEXT NOT NULL CHECK(source IN ('scan','find_object','user')));
CREATE INDEX sightings_noun ON sightings(noun, valid_to, seen);
CREATE VIRTUAL TABLE sightings_fts USING fts5(noun, target, description,
  content='sightings', content_rowid='id', tokenize='unicode61 remove_diacritics 2');
-- + insert/delete/update triggers, same pattern as facts_fts
```

Task episodes use the existing `episodes` table (`kind='task'`, `outcome`, `details_json` already exist). `add_episode` gains `outcome` and `details` keyword arguments.

Store methods (all take `now` from the store clock):
`add_landmark`, `rename_landmark`, `landmarks()`, `clear_places()`,
`add_sighting`, `set_confidence(id, c)`, `delete_sighting(id)`, `invalidate(id, now)`,
`invalidate_scan_sightings(before)`, `latest_sightings(query, limit)` (FTS match on noun/target/description, `valid_to IS NULL` first, newest first),
`task_episodes(noun, since)`.

## 3. Open-vocabulary detector

`perception/open_vocab.py`:

```python
class OpenVocabDetector:            # implements the existing Detects protocol
    def detect(self, bgr: Frame, label: str | None = None) -> list[Detection]: ...
    def detect_many(self, bgr: Frame, nouns: list[str]) -> list[Detection]: ...
```

- YOLOE (`ultralytics`, CPU torch, one thread like YOLOX). The model comes from `OPEN_VOCAB_MODEL`, default `yoloe-11s-seg.pt` [verify name and the smallest model that works].
- `set_classes(nouns, text_pe)` is called only when the noun list changes, with the last list cached. The text encoder cost is paid once per search, not per tile.
- `detect_tiled` is typed on the `Detects` protocol instead of `ObjectDetector`, so both detectors use it.
- Leased like YOLOX: built by a factory on the first search or scan, and released when it ends (rule 6).
- New extra `openvocab = ["ultralytics==<pin after spike>"]` plus a CPU torch index note in 06-operations.
- **Fail small**: if the extra isn't installed or the model doesn't load, `find_object` falls back to YOLOX for nouns that map to COCO (small alias table: phone→cell phone, sofa→couch, table→dining table, tv, laptop, cup, bottle, book, remote…). Other nouns return `unsupported_object` as today, and `learn_room` returns `scan_unavailable`.

YOLOE scores are not comparable to YOLOX's, so `SURE` doesn't apply to YOLOE hits: every hit goes to `choose()` unless there is no vision model. The thresholds are `OPEN_MIN_SCORE` for find and `SCAN_MIN_SCORE` (lower) for scan confirmation, both set from the spike [verify: start 0.25 / 0.15].

## 4. Room scan (`learn_room`)

Owner only. The head is owned through `motion` like the search ("search stop" moves). Takes about 60–90 s [verify]. The agent says one sentence first, as with find.

```
for stop in STOPS (the same 9 as the search):
    look, settle, grab a still frame                      (same code as Search)
    ask(frame, SCAN_STOP + known_so_far) -> JSON          (§4.1)
    validate JSON; drop bad entries; cap 15 entities
    YOLOE detect_many(frame, nouns), tiled, SCAN_MIN_SCORE
    match each entity -> box (§4.2)
    keep: boxed -> exact bearing; landmark unboxed -> region bearing; object unboxed -> drop
merge (§4.3) -> landmarks + sightings
write in ONE transaction (§4.4)
head back home
```

A spoken stop (barge-in) cancels the scan, and **nothing is written**. A half room would invalidate the other half.

### 4.1 Per-stop prompt (`SCAN_STOP`)

It asks for JSON only:

```json
{"entities": [
  {"id": 0, "noun": "desk", "description": "white wooden desk", "kind": "landmark",
   "region": "bottom-left"},
  {"id": 1, "noun": "phone", "description": "black phone", "kind": "object",
   "region": "bottom-left", "relation": "on", "on": 0}
]}
```

- `noun`: the plainest common English noun, one or two words (the same rule as `find_object`'s `object` argument, so both sides produce the same word).
- `kind`: `landmark` = furniture or a fixed thing (desk, shelf, couch, door, window, TV); `object` = something a person moves.
- `region`: one of the 9 cells of a 3×3 grid (`top-left` … `bottom-right`).
- `on`: the `id` of the landmark it rests on or in, or absent.
- `known_so_far`: the landmarks from earlier stops with their bearing relative to this stop ("desk, about 40° to the right of centre"), with the rule "use the same noun if it is the same thing".

The JSON is a trust boundary: parse it strictly, drop entries with a missing or bad noun, kind or region, and treat an unparseable answer as an empty stop. Never raise.

### 4.2 Matching entities to YOLOE boxes

For each entity, the candidates are YOLOE boxes with the same noun (case-insensitive) whose centre lies in the entity's region cell or a neighbouring one. Pick the highest score. Two entities can't take the same box. A box makes the bearing exact (`_aim`, moved to a shared helper). With no box, the region cell's centre goes through `_aim` as a pseudo-box (accurate to about ±15°).

### 4.3 Merge

Neighbouring stops overlap (40° steps, about 88° of view), so one desk shows up 2–3 times.

1. **LLM grouping**: one call to the memory chat model (non-reasoning, already wired in `memory_from_env`). Input: every kept entity as `{ref: "s3e1", noun, description, kind, bearing}`, with bearings rounded to 5° as context. Output: `{"groups": [["s2e0", "s3e0"], ...]}`. Every ref appears exactly once, and refs it leaves out become groups of one.
2. **Code check**: a group whose members are more than 30° apart is split into the members nearest each other. A group mixing `landmark` and `object` is split by kind.
3. **Bearing**: the mean of boxed members if any, else the mean of region members. Confidence is 0.9 boxed or 0.6 region-only for landmarks, 0.7 for objects. The name is the most common noun in the group.
4. **Fallback**: if the call fails or the JSON is bad, code merges same-noun members within 25°. It is never a hard failure.
5. An object's landmark is its `on` from the vision model, mapped through the groups. Without one, the nearest landmark within its `radius` gives `relation=None`, and `where_text` = "near the desk".

### 4.4 Write

In one transaction:
- **Landmarks**: a scan landmark within 20° of an existing one with the same noun updates it (bearing, `updated`), and `source='user'` names are kept. New ones are inserted, and names that collide get a suffix ("chair 2").
- **Sightings**: `invalidate_scan_sightings(before=now)`, then insert the new ones with `source='scan'`. `find_object` sightings are left alone: a newer sighting of the same noun takes over in `plan`, and an empty look-first invalidates the old one. A scan miss (YOLOE missed the phone) must not erase a real find.
- `learn_room(fresh=true)` runs `clear_places()` first. That is the reset for "I moved the robot".

`shortcut: bearings assume the robot base never moves, rescan with fresh=true after moving it; upgrade: detect a move from landmark bearings drifting.`

## 5. Find with memory

### 5.1 Tool change

`find_object(target, object, resume?, confirmed?)`: `object` replaces `coco_class` and is the plainest English noun, free text. `unsupported_object` remains only in the YOLOX fallback (§3). The PROMPT also changes:

- "where is / where did I leave X" → `recall(kind="sighting", query=X)`. Answer from memory without moving, and offer to look.
- "find / look for / show me X" → say one sentence, then `find_object`.
- "did you find X earlier?" → `recall(kind="task", query=X)`.

### 5.2 Policy (`memory/spatial.py`, pure functions, fake clock)

```python
@dataclass(frozen=True)
class Plan:
    look_first: Spot | None            # (yaw, pitch, sighting_id, seen) to check before any sweep
    stop_order: tuple[int, ...]        # indices into STOPS: the sweep order

def plan(sightings, failures, now) -> Plan
```

One rule:

| Latest valid sighting of the noun (`valid_to IS NULL`) | What the search does |
|---|---|
| exists, any age, any confidence | Look **exactly** at its yaw/pitch, check one frame (YOLOE + `choose`). Found → report with `from_memory: true`. Not seen → `invalidate` it, then full sweep. |
| none | Full sweep. |

Age only changes the wording. The result carries `seen_ago` ("5 minutes ago", via `ago()`), and the agent says "it should be on the desk, I saw it 5 minutes ago" or "I last saw it there 3 days ago, let me check".

The sweep order is the default `STOPS` order, with one change: every stop listed in a **failed task episode for this noun in the last 24 h** moves to the end, unless a sighting newer than that failure lies at that stop. The stop nearest the look-first spot moves to the end of the first pass too, since it was just checked.

The only constant in `spatial.py` is `FAILURE_WINDOW_S = 86_400`.

### 5.3 Search hooks (writes)

`Search` calls `SpatialMemory`. Each call is wrapped so that a memory error only logs; the search goes on.

| Moment | Write |
|---|---|
| Found (sweep or look-first) | `add_sighting(source='find_object', confidence=0.6)`, nearest landmark → `where_text`; keep `sighting_id` on `_Run` |
| `confirmed=true` | `set_confidence(id, 1.0)` |
| `resume=true` ("not that one") | `delete_sighting(id)` |
| No answer within `RESUME_S` | nothing: it stays at 0.6 |
| Look-first spot empty | `invalidate(old_id)` |
| Search ends (found, not found, stopped) | task episode (§5.4) |

The result dict gains `where_text` ("on the desk") next to the existing egocentric `where`/`height`, plus `from_memory: true` when it was found at the look-first spot.

Timestamps come from `SpatialMemory`'s clock (the store's `time.time`). `Search._clock` stays monotonic and is used only for durations.

### 5.4 Task episodes

Template, no LLM. Written when the search ends:

```
summary: "Searched for my keys: not found. Checked 9 spots (front, left, right), 31 s."
         "Searched for my phone: found on the desk, 6 s, from memory."
kind='task', outcome='success'|'failure'|'partial' (stopped by the user),
person_ids=[owner], importance=0.3, topics=[noun],
details_json={"goal": target, "noun": noun, "stops": [yaw, ...], "found": {yaw, pitch,
              where_text} | null, "seconds": n, "from_memory": bool}
```

A resumed search updates its own episode (same `_Run`), so one request makes one episode.

## 6. Recall

### 6.1 `recall(kind="sighting", query)`

Owner only: a guest gets `not_owner`. It runs `latest_sightings(query)` and returns at most 3 lines, newest first:

```
"phone: on the desk, 5 minutes ago (I found it, you confirmed)"
"phone: was on the desk 2 hours ago, not there when I checked at 15:10"
```

It uses the existing `ago()`. Confidence < 0.7 adds "(not sure)". Nothing found → `{"found": false}`, and the prompt offers to look.

### 6.2 `recall(kind="task", query)`

Owner only. Uses the `kind='task'` episodes for `query` (FTS over summary/topics), with the `when` window as today, rendered as "task 14:02: Searched for my keys: not found…".

### 6.3 `kind="any"`

Unchanged: facts and conversation episodes only. Places and tasks are asked for explicitly, so they don't crowd personal recall.

### 6.4 Matching nouns

The find tool and the scan prompt both follow the "plainest common English noun" rule, and FTS over noun/target/description catches "cell phone" vs "phone".
`shortcut: no synonym match (smartphone≠phone), add sighting_vecs with the existing embedder when a miss shows up in testing.`

## 7. Places

`name_place(name, landmark?)`, owner only:
- With `landmark`: rename that landmark ("the dining table is my desk"). It is now `source='user'`, confidence 1.0, kept by later scans.
- Without: name the landmark nearest the current head direction within its radius, or create one at the head bearing ("this is my desk").

`where_text` always uses `name`. A YOLOE search for a landmark uses `noun`.

## 8. Configuration

| Env | Default | Meaning |
|---|---|---|
| `OPEN_VOCAB_MODEL` | `yoloe-11s-seg.pt` [verify] | YOLOE weights; empty → YOLOX fallback |
| `OPEN_MIN_SCORE` | from spike | find threshold |
| `SCAN_MIN_SCORE` | from spike | scan confirmation threshold |

`MEMORY=off` turns off all of Phase 2's reads and writes. Find and scan still work, but `learn_room` says it can't remember without memory.

## 9. Failure modes

| Failure | Behaviour |
|---|---|
| YOLOE not installed / fails to load | YOLOX fallback for COCO nouns, `learn_room` → `scan_unavailable` |
| Vision model down during scan | That stop is empty. With no stops at all → `scan_failed`, nothing written. |
| Merge LLM fails | Code merge (§4.3.4) |
| Memory write raises | Log, carry on. The search result is unchanged. |
| Barge-in during scan | Nothing written, head home |
| Barge-in during find | Episode `outcome='partial'`, as written |
| Robot moved | Wrong bearings until `learn_room(fresh=true)` (shortcut, §4.4) |

## 10. Tests

Unit tests, no robot, with fakes for motion, grab, ask, detector and clock:

- **`spatial.plan`**: a valid sighting (5 min old and 3 days old give the same plan) → `look_first` set; no sighting → none; invalidated sighting ignored, the newest valid one used when there are several; failed stops moved last; a failed stop with a newer sighting not moved; a failure older than 24 h ignored; the look-first stop moved to the end of the first pass.
- **`where_text`** and nearest landmark: inside radius, outside (falls back to "about 40° to my left"), a tie.
- **Scan**: entity JSON validation (bad JSON, bad region, too many entities); matching (region, neighbour cell, two entities and one box); landmark kept unboxed, object dropped unboxed; merge split over 30°, kind split, LLM failure → code merge; write keeps user names, invalidates old scan sightings only, `fresh=true` clears; barge-in writes nothing.
- **Search**: look-first found → `from_memory`, look-first empty → invalidate + sweep in plan order; found → sighting 0.6; confirm → 1.0; resume → deleted; not found → failure episode with stops; memory error doesn't break the search; YOLOE hit always goes through `choose`.
- **Store**: V5 migration from V4 (backup made), FTS triggers, `latest_sightings` order.
- **Recall**: sighting/task as owner, guest → `not_owner`, invalidated line text, `kind="any"` unchanged.
- **find tool**: `object` argument, the YOLOX fallback alias table, `unsupported_object` only in fallback.
- **Integration** (`@pytest.mark.integration`): YOLOE loads and finds a phone in a saved frame.

## 11. Spike (before any feature code)

`scripts/spike_scan.py` on 3–5 saved frames from adjacent stops (plus live timing on the robot laptop):

1. **Vision model** (both `VISION_PROVIDER`s): valid JSON rate, whether `region` is right, and the same noun for the same desk across overlapping frames with `known_so_far`.
2. **YOLOE on CPU**: model load time, `set_classes` time, one 640 pass, full frame + 9 tiles. Recall on phone, cup, keys and remote at the room's distance. Which landmarks it misses.
3. **Merge LLM**: correct groups for about 9 stops × 10 entities.

The results decide: tiles in find or scan only, `OPEN_MIN_SCORE`/`SCAN_MIN_SCORE`, the YOLOE model size, the `ultralytics` pin, and whether the 40 s find promise still holds. If a YOLOE sweep is over about 40 s with tiles, find uses the whole frame only and relies on memory first.

## 12. Build order

Each step ships green (ruff, mypy, pytest) with its usage-doc change.

1. Spike (§11).
2. `V5_SPATIAL` + store methods + `spatial.py` policy and `where_text` (fake-clock tests).
3. `find_object` writes sightings and task episodes; `recall(kind="sighting"|"task")`. Uses YOLOX, with egocentric `where_text` until landmarks exist.
4. Memory-first find (`plan` → look-first → ordered sweep).
5. `OpenVocabDetector` + `object` noun argument + YOLOX fallback.
6. `RoomScan` + `learn_room` + `name_place`.

Steps 2–4 already help with no scan and no YOLOE.

## 13. Acceptance

- "Where is my phone?" after it was found less than 10 minutes ago: answered from memory in ≤ 3 s, with no head movement.
- "Find my phone" with a valid sighting and the phone still there: found from memory in ≤ 8 s.
- After `learn_room`, `where_text` names the right landmark for ≥ 80% of objects in a labelled test of 10 placements.
- A failed search, then "did you find my keys earlier?": answered from the task episode.
- No stop is checked twice in one search's first pass; a stop a failed search checked comes last in the next.
- A full scan finishes in ≤ 90 s [verify after spike]; a barge-in writes nothing.
- With YOLOE uninstalled, everything except `learn_room` works as before Phase 2.

## 14. Docs to update with the code

- `docs/usage/memory.md`: new section "Places and objects" (`learn_room`, `name_place`, the questions that work, `fresh=true` after moving the robot, `MEMORY=off`).
- `docs/usage/find-object.md`: the `object` noun, memory first, open-vocabulary, YOLOE install and fallback.
- `docs/FEATURES.md`: the memory and find rows, branch table, "Last updated".
- `docs/architecture/05-agent-tools-memory.md` §5: point to this spec.
- `NOTICE`: YOLOE / ultralytics (AGPL-3.0).
- `docs/architecture/06-operations.md`: `openvocab` extra and CPU torch install.
