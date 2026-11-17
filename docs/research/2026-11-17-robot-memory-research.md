# Robot memory for Reachy Mini: research report and feature plan

Date: 2026-11-17
Scope: what the state of the art in robot and agent memory says, which ideas fit a stationary desktop robot (Reachy Mini Wireless), which use cases memory unlocks, and what to build, in what order, in this repo (`reachy_demo`).

---

## 0. TL;DR

1. **Reachy Mini does not move around.** It has a head (6-DoF Stewart platform), body yaw, antennas, one wide camera, a 4-mic array and a speaker. Most robot-memory research (RAVEN, STaR, ConceptGraphs, 3D-Mem) builds 3D metric maps for mobile robots. For Reachy, the "space" is the **view sphere around the robot** (head yaw/pitch plus image position) and a small, fixed set of **places on the desk/room** ("left of the monitor"). Don't port 3D mapping. Port the *ideas*: object-centric memory, time-stamped captions, task-conditioned retrieval.
2. **Reachy is a social robot first.** The highest-value memory is about **people** (who they are, what they said, what they like, how they felt), then **conversations** (episodes), then **things** (where the keys were), then **skills/routines** (what worked). Social-robot papers (HARMONI, SUMMER, EgoMem, Ella) matter more here than navigation papers.
3. **Recommended architecture: five memory types, one store, one retrieval tool.** Copy RoboMemory's split (working / episodic / semantic / spatial) and add a **person memory**. Keep everything in the existing SQLite file plus a vector index (`sqlite-vec`). Write through an LLM-driven **ADD / UPDATE / DELETE / NOOP** consolidator (RoboMemory, Mem0). Read through an **agentic `recall` tool** that searches by text, time, person and place (STaR's three retrievers, adapted).
4. **Build it in four phases.** (P1) people + facts + episodes with hybrid search; (P2) object sightings + a "where is X" pipeline; (P3) background consolidation, reflection and forgetting; (P4) proactive memory (greetings, reminders, routines) and procedural memory. Each phase ships alone and has an acceptance test.
5. **Privacy is a feature, not an afterthought.** The robot has a camera and a microphone in someone's home. Owner-only writes, per-person isolation, "what do you know about me?", "forget that", retention limits, and no raw audio/video by default.

---

## 1. Reachy Mini constraints that shape the design

| Constraint | Consequence for memory |
|---|---|
| Stationary, desk-sized | Spatial memory = head pose (yaw, pitch) + image box + named landmarks. No SLAM, no point clouds. |
| One wide RGB camera, no depth | Object memory is 2D + bearing. "Where" is answered as a direction and a landmark ("on the desk, to your left, next to the mug"), not XYZ. |
| 4-mic array with DoA | Who-spoke-when can be tied to direction; voice embeddings give speaker identity (sherpa-onnx is already an optional dep). |
| App runs on a CPU-only laptop (WSL2) over Wi-Fi | Embeddings must be small (≤ 100M params, ONNX) or remote. Heavy vision runs in the cloud (`ask_image`). Budget memory writes off the hot path. |
| Pipeline voice agent (S2T → DeepSeek → TTS), ~1-3 s round trip | Memory retrieval in the turn must cost ≤ 150 ms locally, or run as a tool only when needed. Consolidation runs after the turn. |
| Multiple people in the room, owner/guest gate exists | Memory must be **per person**, with owner-only writes and guest data that expires. |
| Home/office environment | Strong privacy needs. Data stays local by default. |

Already in the repo (`docs/architecture/05-agent-tools-memory.md`): SQLite `facts`, `sightings` (label, where_text, yaw, pitch, seen), `people` (owner/guest); `remember / recall / forget` tools; `recall` is keyword + recency; prompt injects ≤ 15 facts and ≤ 10 object locations. The reference app (`reference/conversation_app_tools/remember.py`, `forget.py`) has the same tools. This report extends that design; it does not replace it.

---

## 2. The three reference repos

### 2.1 RoboMemory (CoRL 2026) — `reachy_code/opensource/memory/RoboMemory-Code`

**What it is.** A brain-inspired, multi-memory agent for EB-ALFRED household tasks. Paper: arXiv 2508.01415.

**Architecture.**
- **Information preprocessor**: for every observation, 4 parallel LLM calls produce (a) a first-person step summary and (b) separate search queries for spatial, semantic and episodic memory. Key idea: *each memory gets a query written for it*, not the raw user text.
- **Temporal memory**: a bounded buffer of step summaries. When it reaches N entries (default 4) an LLM compresses them into one summary ("recursive summarization"). Per episode.
- **Spatial memory**: a vector index of entity names + a directed NetworkX graph of `(source, relation, target)` triples ("apple — in — fridge"). Update: an LLM extracts triplets; if old relations exist for those entities, an **integration LLM** sees old + new and returns the revised set, so stale relations are removed. Retrieval: vector search for entities, then a bounded-depth graph expansion.
- **Semantic memory**: reusable knowledge, two kinds — *action knowledge* ("to slice, you must hold a knife") and *task knowledge* (strategy for a whole task). Update: extract a candidate, retrieve top-k similar memories, then an LLM classifier chooses **ADD / UPDATE / DELETE / NOOP**, constrained to the retrieved candidates. Persistent (Chroma).
- **Episodic memory**: one autobiographical summary per task attempt, written at the end, success or failure. Persistent.
- **Actor–Critic**: Actor proposes a subgoal and an action plan; the first action runs; later actions are checked by a Critic against the new observation; rejection triggers a replan.

**Take for Reachy.**
- The four-memory split and the short-term / long-term grouping.
- The **ADD/UPDATE/DELETE/NOOP consolidator constrained to retrieved candidates**. This is the single most useful pattern: it stops duplicate facts and handles "I moved to Hanoi" overwriting "lives in Saigon".
- Memory-specific query generation (cheap version: one LLM call that returns `{people, objects, topics, time}` instead of four calls).
- Episode summaries written at session end, including failures.
- The integration step for spatial relations (old + new → revised), applied to object sightings.

**Leave.** The per-step Actor–Critic loop (Reachy's tasks are short and conversational), pickle-serialized graphs (untrusted-input risk; their own docs warn about it), Chroma + LangChain (heavy deps for a lean repo), the ALFRED action vocabulary.

### 2.2 STaR (RA-L 2026) — `reachy_code/opensource/memory/STaR`

**What it is.** Scalable task-conditioned retrieval over long-horizon multimodal robot memory, tested on a Husky mobile robot. arXiv 2602.09255.

**Architecture.**
- **OmniMem** (task-agnostic memory, built while the robot runs): 3D object primitives from RGB + LiDAR (scene graph), **time-aligned video captions** (a VLM captions short clips; each caption stores time, robot pose and the object IDs visible), and **keyframes**. Stored in Milvus.
- **Task-conditioned retrieval via the Information Bottleneck** (code in `clio_batch/ib_cluster.py`, from Clio, arXiv 2404.13696): given a query, cluster memory items so clusters keep the information relevant to the query and drop the rest. Result: a compact, non-redundant set instead of top-k near-duplicates.
- **Hybrid reranker** (`memory/hybrid_reranker.py`): retrieve top-k captions, then greedily drop candidates whose object-ID sets overlap a candidate already chosen. Simple diversity filter.
- **Agentic RAG** (LangGraph): the LLM has three tools — `retrieve_from_text(query)`, `retrieve_from_position((x,y,z))`, `retrieve_from_time("HH:MM:SS")` — and loops until it can answer. The final answer can attach the keyframes at the cited timestamps, so a VLM re-checks the evidence.
- Answers spatial ("where"), temporal ("when"), and descriptive ("what color") questions and outputs a navigation goal.

**Take for Reachy.**
- **Time-aligned captions of what the camera saw** ("14:02 — owner put a blue mug on the desk, left side"). This is the right unit of visual episodic memory for Reachy.
- **Retrieval by text, by time and by place as separate tools**, chosen by the LLM. For Reachy: `by_text`, `by_time`, `by_person`, `by_place`.
- **Diversity filtering** of retrieved items (by object ID / person ID / time window) before they enter the prompt.
- **Keyframe verification**: store a small JPEG thumbnail with a sighting, and re-ask the VLM when the answer matters ("are you sure it was my wallet?").

**Leave.** LiDAR/3D primitives, Milvus, the full IB clustering (useful at 100k+ items; Reachy will have thousands — MMR gives most of the benefit with 20 lines), Docker/ROS stack.

### 2.3 RAVEN (ICRA 2026) — `reachy_code/opensource/memory/RAVEN`

**What it is.** Aerial (drone) navigation with an open-set semantic memory and behaviour adaptation. arXiv 2509.23563. Submodules (RayFronts, AirStack, LVLM) are not checked out in this copy, so only the README was read.

**Architecture.** RayFronts builds an open-vocabulary semantic map: voxels for things in range, plus **semantic ray frontiers** — directions with a language embedding for things *seen but out of range*. A behaviour tree switches between "go to known match", "follow a promising ray", and "ask the LVLM for a hint". Memory keeps the robot resilient when the target is far or occluded.

**Take for Reachy.**
- **Rays, not points.** Reachy cannot measure depth, so a sighting *is* a ray: head pose + pixel → bearing. RAVEN shows bearing-only semantic memory is useful on its own. Store sightings as `(yaw, pitch, label embedding, time)` and look back along that bearing.
- **Behaviour switching on memory confidence**: confident fresh sighting → look straight there; old or weak sighting → look there first, then sweep; no sighting → ask the VLM / full scan. That's exactly the F5 `find_object` search strategy, made memory-aware.

**Leave.** Voxel maps, global planning, Isaac Sim.

---

## 3. Other research worth knowing

Grouped by what they teach. "Fit" is fit to Reachy Mini (H = high, M = medium, L = low).

### 3.1 Agent memory (LLM, text) — the core for a conversational robot

| Work | Key idea | Fit |
|---|---|---|
| **Generative Agents** (Park et al., 2023, arXiv 2304.03442) | Memory stream of observations; retrieval score = **recency × importance × relevance**; periodic **reflection** writes higher-level insights ("Tuan cares about his mother's health"). | H — the retrieval score and reflection are the backbone of a social robot memory. |
| **MemGPT / Letta** (arXiv 2310.08560) | OS-style memory tiers: core memory (always in prompt, editable by the agent), recall (conversation log), archival (vector store). Agent edits its own memory with tools. | H — "core memory block" = the person profile injected every turn. |
| **Mem0** (arXiv 2504.19413) | Extract facts per turn → retrieve similar → LLM picks ADD/UPDATE/DELETE/NOOP. Graph variant adds entity relations. Small token footprint. | H — same pattern as RoboMemory semantic memory; proven on LoCoMo/LongMemEval. |
| **Zep / Graphiti** (arXiv 2501.13956) | **Bi-temporal knowledge graph**: every fact has `valid_from / valid_to` (world time) and `created / expired` (system time). Old facts are invalidated, not deleted. | H — "Where did I live last year?" needs validity intervals. Copy the two time columns, not the graph DB. |
| **A-MEM** (arXiv 2502.12110) | Zettelkasten-style notes: each memory has keywords, tags, context and links; new notes can update old notes' context. | M — the auto-linking is nice; tags are enough at first. |
| **Hindsight** (vectorize.io; integrated in `natively-cluely-ai-assistant/electron/intelligence/memory/`) | Retain / recall / **reflect** API; separate world facts, experiences, opinions; strict tag-based isolation (bank per user, `all_strict` tag filter). | M — the **isolation-by-tags at retrieval** pattern is directly useful for owner vs. guests. |
| **LoCoMo** (arXiv 2402.17753), **LongMemEval** (arXiv 2410.10813) | Benchmarks for long-term conversational memory: single-hop, multi-hop, temporal reasoning, knowledge update, abstention. | H — use their question *categories* to write Reachy's own memory test set. |
| Long-context vs. memory study (arXiv 2603.04814) | A long-context model can beat a fact-memory system on recall, at much higher cost per turn. | H — a reminder: for the first months, the raw history of one person may fit in context. Measure before building. |

### 3.2 Embodied / robot memory

| Work | Key idea | Fit |
|---|---|---|
| **ReMEmbR** (NVIDIA, arXiv 2409.13682) | Robot builds a memory of VLM captions of video chunks with time and position in a vector DB; an LLM agent queries it with text/time/position tools. STaR builds on this. | H — the simplest proven recipe for "what did you see?" |
| **KARMA** (arXiv 2409.14908) | Long-term memory = 3D scene graph of static objects; short-term memory = recent object states with an LRU/hit-rate policy. | M — the short-term "recently changed objects" cache with eviction is the useful part. |
| **Embodied-RAG** (arXiv 2409.18313) | Hierarchical semantic forest: memories grouped into places → regions, retrieved top-down. | M — Reachy's version: landmark → zone ("desk" → "left of keyboard"). |
| **ConceptGraphs** (arXiv 2309.16650), **HOV-SG**, **3D-Mem** (arXiv 2411.17735) | Open-vocabulary 3D scene graphs; 3D-Mem stores "memory snapshots" (images of what was seen) for VLM reasoning. | L for the 3D parts; M for "store snapshots, not just labels". |
| **Clio** (arXiv 2404.13696) | Task-driven scene graphs via Information Bottleneck: granularity depends on the task. | L now; the idea "store fine, retrieve at the granularity the question needs" is worth keeping. |
| **MemoryVLA** (arXiv 2508.19236), **MEM: Multi-Scale Embodied Memory** (arXiv 2603.03596) | Memory inside vision-language-action policies: short video memory + long language memory. | L — Reachy has no learned manipulation policy. |
| **SafeMem** (arXiv 2609.08444, CoRL 2026) | Long-term semantic graph memory feeds an LLM **risk predictor** that checks candidate actions. | M — use memory for safety rules ("don't look at the user when they said they're changing clothes", "owner asked for no camera after 22:00"). |
| **MemCompiler** (arXiv 2605.07594) | "Compile, don't inject": turn memory into state-conditioned instructions instead of dumping retrieved text into the prompt. | M — supports building a compact *briefing* per turn rather than pasting raw memories. |
| **EmbodiedMemory-Bench** (arXiv 2609.28236), **Worldlines** (arXiv 2606.18847) | 2026 benchmarks for long-horizon embodied memory. | M — borrow question types. Not verified in detail. |

### 3.3 Social / companion robot memory — closest to Reachy

| Work | Key idea | Fit |
|---|---|---|
| **HARMONI** (arXiv 2601.19839) | LLM framework for multi-user socially assistive robots: perception → **world/user modeling** → long-term memory across sessions → personalized responses. Beats baselines on user-model accuracy and satisfaction. | H — multi-user is exactly Reachy's owner/guest setting. |
| **SUMMER** (arXiv 2604.12081) | **Selective**, multimodal memory for social robots: decides what is worth storing from multimodal input. | H — "importance gate" before writing. |
| **EgoMem** (arXiv 2509.11914) | Lifelong memory for full-duplex assistants: identifies users by **face and voice**, updates per-user memory from audio-visual streams, retrieves before responding. | H — the face+voice identity-to-memory pipeline. |
| **Ella** (arXiv 2506.24019) | Embodied social agent with name-centric semantic memory + spatiotemporal episodic memory; plans days, forms relationships. | M — name-centric indexing of people. |
| ACT-R + LLM on the Navel robot (CEUR-WS Vol-4058, 2025) | Cognitive-architecture memory (activation decay) so the robot recalls and **refers back** to past chats next session. | M — activation-based forgetting is a clean, explainable decay rule. |
| Embodied agents meet personalization (arXiv 2505.16348) | Benchmarks memory use for personalized assistance; agents fail at **object semantics** ("my mug") and **user patterns** (routines). | H — tells you which two capabilities to test hardest. |
| Child companion memory paradigm (arXiv 2601.08128) | On-device small model (Qwen2.5-7B) + memory beats a bigger raw LLM for a companion toy. | M — evidence that structured memory matters more than model size for companions. |
| DELTAMEM (arXiv 2606.03083) | Incremental *experience* memory as residual trees: store what changed relative to similar past experience. | L-M — idea for procedural memory later. |

---

## 4. Memory taxonomy for Reachy Mini

Five memory types. Each answers one question.

| Memory | Question it answers | Unit | Lifetime | Written by | Read by |
|---|---|---|---|---|---|
| **Working** | What is happening right now? | Last N turns + live `WorldState` + turn summary | Session | Agent loop | Every turn (prompt) |
| **Person** | Who is this, and what do I know about them? | Profile: name, face/voice embeddings, relations, preferences, traits, last seen, mood history | Owner: permanent. Guest: session or 30 days, opt-in | Identity gate + consolidator | Every turn for the current speaker (core block) |
| **Semantic (facts)** | What is true about the world/people? | Atomic fact with subject, predicate-ish text, confidence, `valid_from/valid_to`, source | Permanent until invalidated | Consolidator (ADD/UPDATE/DELETE/NOOP) | `recall`, prompt top facts |
| **Episodic** | What happened, when? | (a) conversation episode summary; (b) visual event caption; (c) notable moment | Permanent, compressed with age | End-of-session summarizer, perception captioner | `recall` by time/person/topic |
| **Spatial (object/place)** | Where is X / what is usually where? | Sighting: label, embedding, yaw/pitch, box, landmark, thumbnail, time, confidence | Latest per object kept; history compressed | `find_object`, idle glances, captioner | `find_object`, `recall` |
| **Procedural (skills/routines)** | How do I do X for this person / when does Y usually happen? | Routine (time pattern + action), learned preference for a behaviour, successful tool sequence | Permanent, revised on feedback | Reflection job | Scheduler, planner prompt |

Plus two cross-cutting stores:
- **Event log** (append-only, raw-ish): every turn transcript, tool call, sighting. Source of truth for re-deriving memories and for debugging. Retention-limited (e.g., 30 days).
- **Reflections**: higher-level insights derived from many memories ("owner is stressed on Monday mornings"). Marked as inferred, lower confidence, never stated as fact to the user unless asked.

---

## 5. Use cases for Reachy Mini memory

Ranked roughly by value ÷ effort. ★ = needs only P1.

### 5.1 People and relationships
1. ★ **Greets people by name** when it sees/hears them ("Hi Tuan, welcome back"). Face + voice ID → person memory.
2. ★ **Remembers preferences**: language (Vietnamese vs. English), reply length, voice speed, topics they like, how they like to be addressed.
3. ★ **Continues past conversations**: "Last time you said your exam was on Friday. How did it go?" (episodic + follow-up hooks).
4. **Knows the household graph**: "Mai is your sister", "Bin is the dog". Answers "who was here yesterday?"
5. **Guest handling**: recognizes a returning guest only if they opted in; otherwise forgets them at session end.
6. **Emotional continuity**: notices "you seemed tired the last few days" from expression history (FER+ is already planned in perception) — reflected, gentle, opt-in.
7. **Birthdays/anniversaries** stated in conversation → reminder on the day.

### 5.2 Objects and the desk
8. **"Where did I leave my keys/phone/glasses?"** — last sighting with time, bearing and landmark; looks there and confirms with a fresh frame.
9. **"Has anyone touched my notebook?"** / "when did the mug appear?" — visual event captions over time.
10. **Object ownership**: "my mug" vs. "Mai's mug" (personalization benchmark says this is where agents fail).
11. **Inventory of a shelf/desk**: "what's on my desk right now vs. this morning?"
12. **Reading memory**: "remember this page" / "what was the Wi-Fi password on that card?" (OCR via `look_at_pointed read_text`, stored as a fact; sensitive → owner-only, encrypted or refused).

### 5.3 Time, routines and proactivity
13. **Reminders** set by voice: "remind me to call mom at 6" → spoken at 6 if the person is present, else when they next appear.
14. **Routine learning**: "you usually take a break around 15:00" → offers a stretch/dance. Detected from presence + episode times.
15. **Proactive context**: "you asked me to wake you for the meeting" / "you've been at the desk for 2 hours".
16. **Daily recap**: "What did we talk about today?" / weekly summary.
17. **Habit tracking**: water, medication (as a stated user reminder, not medical advice), exercise counts confirmed by the user.

### 5.4 Skills and behaviour
18. **Learns which moves/dances a person enjoys** (laughter/smile after a dance → positive signal).
19. **Learns corrections**: "don't nod so much", "speak slower" → persistent behaviour parameters.
20. **Learns custom commands**: "when I say 'focus mode', turn away and stay quiet for 30 minutes" → stored macro (procedural memory).
21. **Tool-use experience**: which search pose found the phone last time, which VLM phrasing worked ("RoboMemory action knowledge").

### 5.5 Teaching, demo and work
22. **Teach-me-a-word / language tutor**: tracks words learned, spaced-repetition quiz (natural for VN/EN users).
23. **Meeting buddy**: "remember the action items" → fact list per meeting, recap later.
24. **Demo/booth mode**: remembers visitors for the day, what it already showed them, avoids repeating the same joke.
25. **Show-and-tell**: owner shows an object and names it ("this is my lucky coin") → custom object with a reference image for later recognition.

### 5.6 Safety, trust, privacy
26. **"What do you know about me?"** — reads back the profile; "forget everything about X".
27. **Privacy rules as memory**: "don't record when I'm on a call", "no camera after 22:00" → enforced in code by policy, stored as user settings.
28. **Consent memory**: who agreed to be recognized.

---

## 6. Proposed architecture

### 6.1 Data flow

```
                 ┌───────────── hot path (per turn, ≤150 ms local) ─────────────┐
 S2T text ──►    │ identity (face+voice+DoA) ─► speaker_id                       │
 WorldState ─►   │ briefing = core(person) + top facts + open follow-ups         │──► LLM (DeepSeek)
                 │ (no vector search unless cheap; agent calls `recall` if needed)│      │ tools: recall,
                 └────────────────────────────────────────────────────────────────┘      │ remember, forget,
                                                                                         │ find_object, ...
                 ┌────────────── warm path (after each turn, async) ──────────────┐      ▼
 turn transcript─►│ event_log append ─► importance gate ─► fact extractor (LLM)   │   answer ─► TTS
                 │ ─► retrieve similar (hybrid) ─► ADD/UPDATE/DELETE/NOOP (LLM)  │
                 └────────────────────────────────────────────────────────────────┘
                 ┌────────────── cold path (session end / idle / nightly) ────────┐
                 │ episode summary · reflection · routine mining · decay/forget   │
                 │ · dedupe/merge · compress old episodes · re-embed on model chg │
                 └────────────────────────────────────────────────────────────────┘
```

Rules:
- **Hot path never calls an extra LLM.** Briefing is built from SQLite with indexes. If the question needs memory, the agent calls `recall` (a normal tool round, which the agent already supports with `MAX_ROUNDS = 4`).
- **Warm path is fire-and-forget** with a queue (like `HindsightRetainQueue`); failure loses a memory write, never a turn.
- **Cold path is idempotent**: it can re-run from the event log.

### 6.2 Storage: one SQLite file

Keep `data/memory.db` (stdlib `sqlite3`, WAL). Add `sqlite-vec` (single extension, no server) for vectors and FTS5 (built in) for keyword search. No Chroma, no Milvus, no graph DB: a `relations` table covers the graph needs at this scale.

```sql
-- who
people(id PK, role CHECK(role IN ('owner','guest')), name, aliases_json,
       consent_recognize INT, created, last_seen, retention_until NULL)
person_embeddings(id PK, person_id FK, kind CHECK(kind IN ('face','voice')),
                  vec BLOB, quality REAL, created)      -- several per person; keep best K

-- what is true (bi-temporal, Zep-style)
facts(id PK, subject_person_id NULL, subject_text, text, category,   -- preference|bio|relation|plan|setting|other
      confidence REAL, importance REAL, source CHECK(source IN ('said','seen','inferred')),
      valid_from NULL, valid_to NULL,            -- world time
      created, expired NULL,                     -- system time; expired = replaced/invalidated
      last_used, use_count, sensitive INT DEFAULT 0, evidence_event_ids_json)
facts_fts USING fts5(text, subject_text, content='facts')
facts_vec USING vec0(embedding float[384])

relations(id PK, src_kind, src_id, rel, dst_kind, dst_id, valid_from, valid_to, created, expired)
  -- person–person ("sister_of"), person–object ("owns"), object–place ("usually_on")

-- what happened
episodes(id PK, kind CHECK(kind IN ('conversation','visual','moment')),
         start, end, person_ids_json, summary, topics_json, importance, emotion_json,
         follow_ups_json,                         -- ["ask how the exam went (due 2026-11-21)"]
         compressed_level INT DEFAULT 0)
episodes_fts, episodes_vec

-- where things are (bearing, not XYZ)
objects(id PK, label, owner_person_id NULL, description, ref_thumb BLOB NULL, created)
sightings(id PK, object_id FK NULL, label, yaw_deg, pitch_deg, box_norm_json, landmark,
          where_text, thumb BLOB NULL, confidence, seen, source CHECK(source IN ('detector','vlm','user')))
landmarks(id PK, name, yaw_deg, pitch_deg, radius_deg)   -- "desk-left", "monitor", "door"

-- how/when
routines(id PK, person_id, pattern_json, action, confidence, last_fired, enabled)
reminders(id PK, person_id, text, due, deliver_when CHECK(deliver_when IN ('at_time','when_present')), done)
settings(person_id, key, value, updated)                 -- behaviour prefs + privacy rules
reflections(id PK, person_id NULL, text, evidence_ids_json, confidence, created, expired)

-- raw
event_log(id PK, t, kind, person_id, payload_json)       -- turns, tool calls, sightings; retention 30 d
```

Why this shape:
- **`expired` instead of DELETE** for facts gives history ("you used to live in Saigon") and safe undo. A real `forget` request does a hard delete (privacy wins over history).
- **`source` = said / seen / inferred** lets the robot phrase confidence correctly ("you told me…", "I saw…", "I think…").
- **Embeddings with model name + dim in a `meta` table**; a model change triggers a cold-path re-embed (RoboMemory warns stored vectors are tied to the embedding model).

### 6.3 Embeddings (CPU, local)

- **Text**: `multilingual-e5-small` or `bge-m3`-small class model in ONNX via onnxruntime (already a dependency). Must handle Vietnamese + English. 384-dim, ~10-20 ms per sentence on CPU **[verify on this laptop]**.
- **Face**: a small ArcFace/MobileFaceNet ONNX on the MediaPipe face crop. Cosine threshold tuned on the owner's own data.
- **Voice**: sherpa-onnx speaker embedding (already the `speaker` extra and planned for the owner gate).
- **Images (optional, P2+)**: SigLIP/CLIP-small ONNX for "find things that look like my lucky coin". Only if detector labels + VLM are not enough.

### 6.4 Write path (consolidation)

1. **Importance gate** (SUMMER, Generative Agents): cheap rules first — skip greetings, filler, ASR garbage (< 3 words, low ASR confidence); keep anything with a name, number, date, "I like / I hate / my …", "remember". Then one LLM score 1-10 only for borderline turns. Store `importance`.
2. **Extract** (one LLM call, JSON schema): list of candidate facts `{subject, text, category, valid_from?, valid_to?, sensitive}`, plus `follow_ups` and `reminders`. Third person, atomic, ≤ 1 sentence (same rule as the reference `remember` tool).
3. **Retrieve similar**: hybrid (FTS5 BM25 + vector) top-5 for the same subject.
4. **Decide** (one LLM call, constrained to those 5 ids): `ADD`, `UPDATE(id)`, `INVALIDATE(id, valid_to)` (Zep-style instead of raw DELETE), `NOOP`. Reject any id not in the candidate set (RoboMemory's constraint).
5. **Apply** in one SQLite transaction; record `evidence_event_ids`.
6. **Policy in code**: guest turns never write owner facts; `sensitive` facts (health, addresses, passwords, finances) are refused or stored only on explicit owner request, flagged, and never put into the prompt automatically.

Steps 2 and 4 can be merged into one call when the candidate list is small — fewer LLM calls, same quality in practice **[verify]**.

### 6.5 Read path (retrieval)

**Briefing (every turn, no LLM, no vectors):**
- Core block for the current speaker: name, relation to owner, top preferences/settings, language, ≤ 10 highest-scoring facts.
- Open follow-ups and due reminders for this person.
- Last episode summary with this person (1-2 sentences).
- Size cap: ~400 tokens. Score = `w_r·recency + w_i·importance + w_u·log(1+use_count)` with exponential recency decay (half-life ~14 days for facts, ~3 days for episodes).

**`recall` tool (agentic, STaR/ReMEmbR-style):** one tool with typed filters rather than three tools — fewer tokens in every request, same expressiveness:
```json
{"name": "recall",
 "parameters": {"query": "string",
                "person": "string|null",   // "me", a name, or null
                "kind": "fact|episode|sighting|any",
                "time_from": "ISO|relative|null", "time_to": "...",
                "place": "landmark name|null",
                "limit": 5}}
```
Implementation: filter by person/time/place in SQL → hybrid score (BM25 + cosine, reciprocal rank fusion) → **MMR diversity** (STaR reranker idea, by object/person/time window) → return compact lines with `[said|seen|inferred, 3 days ago]` tags so the LLM phrases confidence correctly.

**Time parsing**: "yesterday", "last Monday", "this morning" resolved in code (VN + EN), not by the LLM, so temporal questions (LongMemEval's hardest category) don't depend on model arithmetic.

**Abstention**: if the best score is below a threshold, `recall` returns `{"found": false}` and the prompt rule "never invent memories" makes the robot say it doesn't remember. LongMemEval tests this explicitly; it's the biggest trust killer when wrong.

### 6.6 Spatial memory for a stationary robot

- **Sighting = ray**: head yaw/pitch at capture + box centre → absolute bearing `(az, el)` in the robot base frame. Body yaw included.
- **Landmarks**: owner names zones once ("this is my desk", "that's the door") while Reachy looks there; stored as bearing + radius. Sightings get `landmark` = nearest zone, so answers are human ("on the desk, near the monitor").
- **Freshness-aware search** (RAVEN behaviour switching): `age < 10 min & conf ≥ 0.7` → look there and confirm one frame; `age < 1 day` → look there first then partial sweep; else → full F5 scan. Always write the new sighting.
- **Integration step** (RoboMemory spatial): new sighting of the same object replaces "current location"; old ones move to history; "not seen at its usual place" is a fact the robot can say ("it's not where you usually leave it").
- **Idle glance captioning** (P3, optional, ReMEmbR/STaR): every few minutes while idle and the owner allows it, caption one frame with the VLM → visual episode `{time, bearing, caption, object labels}`. Rate-limited for cost and privacy.
- **Usual places**: mined from sighting history → `relations(object, usually_on, landmark)`.

### 6.7 Forgetting, decay and compression

- **Decay score** (ACT-R-like): activation = `ln(Σ t_i^-0.5)` over past uses; facts below a floor and older than N months drop out of the briefing (not deleted).
- **Compression**: episodes older than 7 days are merged into daily summaries, older than 60 days into weekly ones (RoboMemory recursive summarization, applied across time). Keep `compressed_level`.
- **Hard limits**: event log 30 days; guest data at session end unless consent; thumbnails 7 days; owner can set all of these.
- **Contradiction handling**: by INVALIDATE with `valid_to`, never silent overwrite.

### 6.8 Reflection and routines (cold path)

- **Reflection** (Generative Agents): nightly, per person, ask the LLM for ≤ 3 insights from the day's episodes with evidence ids. Stored as `reflections` (inferred, confidence < 1). Used to set tone, never stated as fact without hedging.
- **Routine mining**: cluster presence and episode start times per weekday; if a pattern holds ≥ 3 of the last 4 weeks, propose it to the user ("I noticed you take a break around 3 pm. Want me to remind you to stretch?"). Only enabled after a yes. Procedural memory without consent is creepy.

---

## 7. Feature list and roadmap

Each phase is shippable alone. Acceptance tests are spoken scenarios plus a scripted test set (§8).

### Phase 1 — People, facts, episodes (foundation)
| # | Feature | Notes |
|---|---|---|
| 1.1 | Schema migration from current `facts/sightings/people` to §6.2 (keep data) | Reversible; back up `memory.db` first. |
| 1.2 | Event log + async write queue | Never blocks a turn. |
| 1.3 | Owner enrolment: "I'm Tuan" → voice + face embeddings | Reuses the speaker gate. |
| 1.4 | Identity per turn (face + voice + DoA fusion) → `speaker_id` | Unknown → "guest". |
| 1.5 | Fact consolidator (extract → retrieve → ADD/UPDATE/INVALIDATE/NOOP) | Mem0/RoboMemory pattern. |
| 1.6 | Hybrid `recall` (FTS5 + sqlite-vec + RRF + MMR) with time/person filters | Replace keyword `LIKE`. |
| 1.7 | Briefing block (core profile + top facts + follow-ups) | ≤ 400 tokens. |
| 1.8 | Episode summary at session end + follow-ups | "How did the exam go?" |
| 1.9 | Memory transparency: "what do you know about me?", "forget X", "forget everything" | Owner-only for writes. |
| **Acceptance** | Robot greets owner by name after restart; answers 20 scripted memory questions (§8) with ≥ 85% correct and 0 invented answers on "unknown" questions; turn latency increase ≤ 150 ms p95. | |

### Phase 2 — Objects and places
| # | Feature | Notes |
|---|---|---|
| 2.1 | Sightings as rays + landmarks ("this is my desk") | §6.6 |
| 2.2 | Memory-aware `find_object` (fresh/old/none behaviour) | RAVEN-style switching. |
| 2.3 | Object ownership and custom objects ("this is my lucky coin" + thumbnail) | Fix "my mug" failures. |
| 2.4 | Keyframe verification on demand | STaR. |
| 2.5 | Usual-place mining; "it's not where you usually leave it" | |
| **Acceptance** | "Where is my phone?" answered from memory in ≤ 3 s when seen < 10 min ago; F5 full search still ≤ 15 s; correct landmark ≥ 80%. | |

### Phase 3 — Consolidation, reflection, forgetting
| # | Feature | Notes |
|---|---|---|
| 3.1 | Cold-path job (idle/nightly): dedupe, compress episodes, decay | Idempotent from event log. |
| 3.2 | Reflections with evidence | Hedged use only. |
| 3.3 | Retention policies + guest expiry | Settings per person. |
| 3.4 | Idle glance visual episodes (opt-in) | Rate-limited. |
| 3.5 | Memory inspector CLI: `reachy-demo memory list/show/forget/export` | Debugging + user trust. |
| **Acceptance** | After 2 weeks of simulated logs: briefing stays ≤ 400 tokens; no duplicate facts on a 50-fact update script; knowledge-update questions ≥ 85%. | |

### Phase 4 — Proactive and procedural
| # | Feature | Notes |
|---|---|---|
| 4.1 | Reminders (`at_time` / `when_present`) | Presence from WorldState. |
| 4.2 | Routine proposals with consent | |
| 4.3 | Behaviour preference learning (moves liked, speech speed, "don't nod so much") | Writes `settings`. |
| 4.4 | Voice macros ("focus mode" → stored action sequence) | Procedural memory, owner-only. |
| 4.5 | Memory-based safety/privacy rules enforced in code | SafeMem idea. |
| 4.6 | Daily/weekly recap; language-tutor spaced repetition | Optional apps. |

### Explicitly not recommended now
- 3D scene graphs / point-cloud memory (no depth sensor, no mobility).
- A separate graph database (Neo4j/Graphiti) — `relations` table is enough below ~100k rows.
- Full Information-Bottleneck clustering — MMR first; revisit if retrieval quality fails with > 50k items.
- Storing raw audio/video long-term — captions + thumbnails only.
- Fine-tuning the LLM on user data.

---

## 8. Evaluation plan

Write a **Reachy memory test set** (YAML, ~100 items) using LongMemEval / LoCoMo categories, run offline with a scripted event log (no robot needed), plus a few live checks.

| Category | Example | Target |
|---|---|---|
| Single-session recall | "What's my dog's name?" | ≥ 95% |
| Cross-session recall | Told on day 1, asked on day 9 | ≥ 90% |
| Knowledge update | "I moved to Hanoi" after "I live in Saigon" → "Where do I live?" | ≥ 90% |
| Temporal reasoning | "What did we talk about last Tuesday?" / "When did I last see Mai?" | ≥ 80% |
| Multi-hop | "What does my sister's dog eat?" | ≥ 75% |
| Abstention | Never-mentioned fact | 100% "I don't know" |
| Person isolation | Guest asks "what's Tuan's address?" | 100% refusal |
| Object location | "Where did I put my glasses?" | ≥ 80% landmark correct |
| Vietnamese/English mixed | Fact told in VN, asked in EN | ≥ 85% |

Metrics also tracked: p50/p95 added turn latency, LLM calls per turn on warm path, briefing tokens, DB size per week, cost per day. Use an LLM judge only for open answers; exact-match for names/dates.

---

## 9. Privacy and safety checklist

- Local-only storage by default; cloud calls send only the minimum (a crop, a question), never the whole memory.
- Owner-only memory writes and deletes, enforced in code (existing permission policy).
- Retrieval always filtered by the current speaker's scope (Hindsight-style `all_strict` isolation): a guest can never read owner facts.
- Sensitive categories (health, finance, passwords, addresses, children's data) refused unless the owner explicitly asks, then flagged and excluded from auto-briefing.
- Face/voice recognition only for people who consented; guests are anonymous by default.
- "What do you know about me?", "forget that", "forget everything", export to JSON.
- Visible indicator (antenna pose or LED-like move) while the camera is recording a memory, if feasible.
- Retention limits per data class (§6.7).
- No pickle in the memory store (RoboMemory's graph pickle is an untrusted-input risk).

---

## 10. Open questions for the owner

1. Is the target **personal home companion**, **office/demo booth**, or both? This changes guest retention and proactivity defaults.
2. Is it acceptable to call the cloud LLM (DeepSeek) for memory extraction, or must extraction run locally?
3. Should guests be recognizable across visits (with spoken consent), or always anonymous?
4. Idle visual captioning (§6.6): wanted, and at what rate/cost?
5. Which use cases in §5 matter most for the next demo? Phase order can follow that.

---

## Sources

Reference repos (local): `reachy_code/opensource/memory/RoboMemory-Code` (README, `docs/architecture.md`, `robomemory/Modules/Memories/LTM/SemanticMem.py`), `reachy_code/opensource/memory/STaR` (README, `docs/BuildMemory.md`, `src/star/memory/hybrid_reranker.py`, `src/star/agents/star_agent_aib.py`, `src/star/clio_batch/ib_cluster.py`), `reachy_code/opensource/memory/RAVEN` (README only; submodules not checked out). Existing design: `docs/architecture/05-agent-tools-memory.md`, `docs/architecture/03-perception.md`.

Papers and pages:
- RoboMemory — https://arxiv.org/abs/2508.01415
- STaR — https://ieeexplore.ieee.org/abstract/document/11456661 (arXiv 2602.09255)
- RAVEN — https://arxiv.org/pdf/2509.23563
- KARMA — https://arxiv.org/pdf/2409.14908
- Ella — https://arxiv.org/pdf/2506.24019
- MemCompiler — https://arxiv.org/pdf/2605.07594
- SafeMem — https://arxiv.org/abs/2609.08444
- MEM: Multi-Scale Embodied Memory — https://arxiv.org/pdf/2603.03596v2
- EmbodiedMemory-Bench — https://arxiv.org/pdf/2609.28236
- DELTAMEM — https://arxiv.org/pdf/2606.03083
- Embodied Agents Meet Personalization — https://arxiv.org/pdf/2505.16348
- Memory for Autonomous LLM Agents (survey) — https://arxiv.org/pdf/2603.07670
- Always-On Agents (survey) — https://arxiv.org/pdf/2606.30306
- HARMONI — https://arxiv.org/html/2601.19839
- SUMMER — https://arxiv.org/html/2604.12081v1
- EgoMem — https://arxiv.org/pdf/2509.11914v1
- Child companion memory paradigm — https://arxiv.org/pdf/2601.08128
- ACT-R + LLM on Navel — https://ceur-ws.org/Vol-4058/paper6.pdf
- iPET (ACL 2025 demo) — https://preview.aclanthology.org/ingestion-acl-25/2025.acl-demo.40.pdf
- Fact memory vs. long context — https://www.alphaxiv.org/abs/2603.04814
- Schema-grounded memory — https://arxiv.org/pdf/2604.27906
- Memory benchmarks overview (vendor, Mem0) — https://mem0.ai/blog/ai-agent-memory-benchmark
- Memory systems comparison (blog) — https://blog.devgenius.io/ai-agent-memory-systems-in-2026-mem0-zep-hindsight-memvid-and-everything-in-between-compared-96e35b818da8
- From prior knowledge (not re-fetched this session): Generative Agents arXiv 2304.03442; MemGPT arXiv 2310.08560; Mem0 arXiv 2504.19413; Zep arXiv 2501.13956; A-MEM arXiv 2502.12110; LoCoMo arXiv 2402.17753; LongMemEval arXiv 2410.10813; ReMEmbR arXiv 2409.13682; Embodied-RAG arXiv 2409.18313; ConceptGraphs arXiv 2309.16650; 3D-Mem arXiv 2411.17735; Clio arXiv 2404.13696; MemoryVLA arXiv 2508.19236.
