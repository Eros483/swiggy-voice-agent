# Dineout Voice Reservation Agent — Design Document

---

## Overview

A multilingual AI voice agent that lets Indian users call a phone number and book restaurant tables on Swiggy Dineout using natural speech. The system handles the full reservation flow: discovery, availability checks, booking, cancellations, and rebooking history.

Built on a DIY stack: Twilio for telephony, a lightweight audio language classifier routed to language-specific finetuned Moonshine models for STT, Groq for LLM inference, Sarvam Bulbul for TTS, and Swiggy Builders Club MCP for Dineout actions.

---

## Language Scope

Restaurant reservation is a fundamentally different product from food delivery. Swiggy's own voice ordering targets 11 languages because delivery has rural and semi-urban demand. Dineout is metro behaviour — the user has a smartphone, goes to dine-out restaurants, and is comfortable booking in advance. Dineout restaurant density is concentrated in Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, Kolkata, and Pune.

**Supported languages (v1):**

| Language | Cities covered |
|---|---|
| Hindi | Delhi, Mumbai, most of north India |
| English | All metros, code-switchers |
| Telugu | Hyderabad |
| Bengali | Kolkata |
| Marathi | Mumbai, Pune |

These 5 languages cover the overwhelming majority of realistic Dineout users. Expanding to Tamil (Chennai) and Kannada (Bengaluru) is the natural v2 step, driven by actual call data.

Scoping to 5 languages also directly improves ML quality — more IndicVoices data per language, simpler classifier discrimination, and less catastrophic forgetting risk during Moonshine finetuning.

---

## Stack at a Glance

| Layer | Tool | Notes |
|---|---|---|
| Telephony | Twilio Media Streams | WebSocket, mulaw 8kHz |
| VAD | Silero VAD | Per 30ms frame, barge-in handling |
| Language Classification | Finetuned audio classifier | Tiny encoder + 5-class head, runs on raw audio |
| STT | 5 language-specific Moonshine models | One per language, routed by classifier |
| STT Benchmark Target | Sarvam Saaras v1 | Commercial baseline to beat |
| LLM + Tool Use | Groq | Fast inference, streamed output |
| Commerce Layer | Swiggy Builders Club — Dineout MCP | 6 tools, real transactions |
| TTS | Sarvam Bulbul v3 | Language-matched voice from classifier output |
| Token Store | PostgreSQL | Caller ID → Swiggy OAuth token |

---

## Architecture Decisions

### Why Not Finetuned Whisper

Whisper was the first candidate because its decoder predicts a language token before transcribing — giving STT and language detection in one pass from raw audio. It solves the chicken-and-egg problem where text-based LID needs a transcript and STT needs a language code.

However Whisper has a fundamental architectural inefficiency for real-time use: it pads all inputs to a fixed 30-second window regardless of actual audio length, and pays the full compute cost every time. For short utterances (the norm in a voice agent — "book a table for 2 Saturday 8pm") this is wasteful.

Moonshine was designed specifically to fix this. It uses variable-length input windows and encoder/decoder state caching, scaling inference cost to actual audio duration rather than a fixed window. Empirically: Moonshine v2 Small achieves 148ms latency — 13.1x faster than Whisper Small on equivalent hardware.

The reason Moonshine wasn't used initially was that it has no equivalent to Whisper's language token — it's architecturally a transcription-only model, designed monolingual. But this is solvable by training a separate classifier, which is what this design does.

### Why Not a Single Multilingual Moonshine

The "Flavors of Moonshine" research shows explicitly that Moonshine's accuracy gains come from training monolingual models — a monolingual Moonshine Tiny outperforms Whisper Medium (28x larger) on target languages. A multilingual Moonshine finetuned across all 5 languages would dilute this, producing a model that's worse than 5 separate specialist models and still has no language token.

Separate models per language is the right call. At int8 quantization each Moonshine Tiny is ~27MB. Five models in memory is ~135MB total — well within budget.

### Why Not Text-Based LID (IndicLID)

```
STT needs language → to produce transcript
LID needs transcript → to detect language
→ Chicken-and-egg. Wrong language in = garbage transcript = wrong LID output.
```

The classifier must run on raw audio, before STT.

### Why Not Audio-Based LID (facebook/mms-lid)

Correct direction but a 126-language general-purpose model for a 5-class problem. Overparameterised, slower than necessary, and not trained on phone-call quality audio. A classifier trained specifically on IndicVoices for these 5 languages will be smaller, faster, and more accurate on this distribution.

### Why STT Is Not Streaming

In a reservation agent you cannot act on a partial transcript. "Book a table for" is useless until the utterance completes — you need party size, location, date, and time before doing anything. The right design is to buffer audio until end-of-utterance (VAD silence detection) and then run the full buffer through the classifier and Moonshine. This is a single batch step in an otherwise streaming pipeline, and it's the correct choice for this use case.

---

## The Classifier

A lightweight audio encoder finetuned as a 5-class language identifier. Runs on raw audio before any STT, so there is no chicken-and-egg dependency.

### Architecture

```
Input: raw audio buffer (16kHz PCM, full utterance)
  ↓
Feature extraction: mel-spectrogram or MFCC
  ↓
Tiny encoder: pretrained audio backbone
  Options:
    - wav2vec2-base (95M) with classification head  ← more accurate
    - ECAPA-TDNN (6M) trained from scratch          ← faster, smaller
    - MMS-LID encoder finetuned on 5 classes        ← good starting point
  ↓
Linear classification head: encoder_dim → 5
  ↓
Softmax → [p_hi, p_en, p_te, p_bn, p_mr]
  ↓
Output: predicted language + confidence score
```

### Training

Dataset: `ai4bharat/IndicVoices` filtered to Hindi, English, Telugu, Bengali, Marathi.

```python
# Filter IndicVoices to 5 target languages
dataset = load_dataset("ai4bharat/IndicVoices")
target_langs = {"hi", "en", "te", "bn", "mr"}
filtered = dataset.filter(lambda x: x["language"] in target_langs)

# Balance classes — IndicVoices is not equally distributed
# Undersample majority, oversample minority to equal class size
```

Split: 80% train / 10% val / 10% test (same held-out test set as Moonshine benchmark).

### Decision Rules Per Utterance

| Condition | Action |
|---|---|
| conf > 0.80 AND utterance > 5 words | Update `active_language`, route to that Moonshine model |
| conf < 0.80 OR utterance ≤ 5 words | Keep previous `active_language` |
| First utterance of call | Default to Hindi if conf < 0.80 |
| Mid-call language switch (conf > 0.80) | Update `active_language` and Bulbul TTS params |

The short-utterance guard matters most at the start of a call — "haan", "ok", "nahi" are 1-word responses that don't carry enough signal for reliable classification.

---

## Moonshine STT — Per-Language Models

Five separate Moonshine Tiny models, each finetuned on a single language from IndicVoices.

### Why Moonshine Tiny (not Small or Medium)

```
Moonshine Tiny:   27M params, ~50ms latency, ~27MB at int8
Moonshine Small: 130M params, ~148ms latency, ~130MB at int8

Five Tiny models in memory:  ~135MB, ~50ms per inference
Five Small models in memory: ~650MB, ~148ms per inference
```

Tiny is the right starting point. Move to Small for a specific language only if WER benchmarks show Tiny is insufficient for that language.

### Finetuning

```python
from moonshine import MoonshineModel
from datasets import load_dataset

# Per-language finetuning loop
for lang in ["hi", "en", "te", "bn", "mr"]:
    model = MoonshineModel.from_pretrained("UsefulSensors/moonshine-tiny")

    lang_dataset = load_dataset(
        "ai4bharat/IndicVoices",
        split="train"
    ).filter(lambda x: x["language"] == lang)

    # Standard seq2seq finetuning
    # 80/10/10 split per language
    # Track per-language WER after every eval epoch
    # Early stopping on validation WER
```

### Edge Optimization — ONNX Export

Moonshine's official repo provides ONNX export. Each model at int8:

```python
# From moonshine repo
python -m moonshine.export --model moonshine-tiny-hi --format onnx --quantize int8
# → moonshine-tiny-hi-int8.onnx (~27MB)
```

All 5 models loaded at startup, kept warm in memory. No cold-start latency per call.

### Memory Layout at Runtime

```
Process memory:
  Classifier model (ECAPA-TDNN int8):   ~6MB
  Moonshine Tiny Hindi (int8):          ~27MB
  Moonshine Tiny English (int8):        ~27MB
  Moonshine Tiny Telugu (int8):         ~27MB
  Moonshine Tiny Bengali (int8):        ~27MB
  Moonshine Tiny Marathi (int8):        ~27MB
  ─────────────────────────────────────────────
  Total STT stack in memory:            ~141MB
```

---

## Using Detected Language for TTS

The classifier's output language drives the Sarvam Bulbul TTS call directly — `target_language_code` and `speaker` are set from `active_language` on every response:

```python
LANGUAGE_TO_BULBUL = {
    "hi": {"target_language_code": "hi-IN", "speaker": "anand"},
    "en": {"target_language_code": "en-IN", "speaker": "anand"},
    "te": {"target_language_code": "te-IN", "speaker": "anu"},
    "bn": {"target_language_code": "bn-IN", "speaker": "anu"},
    "mr": {"target_language_code": "mr-IN", "speaker": "anu"},
}

def get_tts_params(lang: str, confidence: float, word_count: int):
    if confidence < 0.80 or word_count <= 5:
        return None  # keep previous voice
    return LANGUAGE_TO_BULBUL.get(lang, LANGUAGE_TO_BULBUL["hi"])

# On each utterance:
tts_params = get_tts_params(detected_lang, confidence, word_count)
if tts_params:
    active_tts_params = tts_params

# TTS call:
response = requests.post(
    "https://api.sarvam.ai/text-to-speech",
    headers={"API-Subscription-Key": SARVAM_API_KEY},
    json={
        "text": llm_response,
        "target_language_code": active_tts_params["target_language_code"],
        "speaker": active_tts_params["speaker"],
        "model": "bulbul:v3",
        "pace": 1.0,
    }
)
```

The agent mirrors the caller's language automatically. Mid-call switches are caught on the next utterance via the classifier and the TTS voice updates accordingly.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          TWILIO LAYER                                │
│                                                                      │
│  Inbound call arrives                                                │
│  → Extract caller ID from Twilio webhook                             │
│  → Lookup phone_number in Postgres                                   │
│      Found + token valid   → proceed with access_token              │
│      Found + token expired → refresh via Swiggy OAuth, update DB     │
│      Not found             → IN-CALL ONBOARDING FLOW (see below)     │
│  → Open Media Stream WebSocket                                       │
│  → Receive audio as mulaw 8kHz                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      AUDIO PREPROCESSING                             │
│                                                                      │
│  mulaw 8kHz → PCM 16kHz resample                                     │
│  Frame into 30ms chunks                                              │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                         SILERO VAD                                   │
│                                                                      │
│  Per 30ms frame: speech_prob = model(chunk, 16000)                   │
│                                                                      │
│  speech_prob > 0.5 during silence  → start buffering                │
│  speech_prob > 0.5 during TTS      → BARGE-IN                       │
│    └── kill TTS stream immediately                                   │
│    └── flush old audio buffer                                        │
│    └── start fresh buffer                                            │
│  700ms silence after speech        → end of utterance               │
│    └── flush buffer → trigger classifier + STT pipeline              │
│                                                                      │
│  Metrics: false positive rate, false negative rate,                  │
│           barge-in response time < 200ms                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     LANGUAGE CLASSIFIER                              │
│                                                                      │
│  Input:  full utterance buffer (16kHz PCM)                           │
│  Model:  tiny audio encoder + 5-class head                           │
│          finetuned on IndicVoices (hi, en, te, bn, mr)               │
│  Output: predicted language + confidence score                       │
│                                                                      │
│  Decision rules:                                                     │
│    conf > 0.80 AND > 5 words → update active_language               │
│    conf < 0.80 OR ≤ 5 words  → keep previous active_language        │
│    first utterance, low conf → default to Hindi                      │
│                                                                      │
│  Metrics: 5-class accuracy, per-language F1,                         │
│           short-utterance accuracy tracked separately                │
└──────┬───────────────────────────────────────────────────────────────┘
       ↓  active_language → route to correct Moonshine model
┌──────────────────────────────────────────────────────────────────────┐
│              MOONSHINE STT — Language-Specific Models                │
│                                                                      │
│  5 finetuned Moonshine Tiny models, all loaded in memory:            │
│    moonshine-tiny-hi  (Hindi)                                        │
│    moonshine-tiny-en  (English)                                      │
│    moonshine-tiny-te  (Telugu)                                       │
│    moonshine-tiny-bn  (Bengali)                                      │
│    moonshine-tiny-mr  (Marathi)                                      │
│                                                                      │
│  All exported to ONNX, int8 quantized (~27MB each, ~135MB total)     │
│  Routed by classifier output — only one model runs per utterance     │
│                                                                      │
│  Input:  full utterance buffer (16kHz PCM) — NOT streaming           │
│  Output: transcript text                                             │
│  Guard:  empty transcript → skip, re-open mic                        │
│                                                                      │
│  Latency target: ~50ms per utterance (Moonshine Tiny)                │
│                                                                      │
│  Metrics: WER per language, macro WER, RTF                           │
│  Benchmark: held-out IndicVoices test set vs Sarvam Saaras v1        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                    GROQ LLM + SWIGGY DINEOUT MCP                     │
│                                                                      │
│  System prompt:                                                      │
│    - You are a table reservation assistant for Swiggy Dineout        │
│    - Respond in: {active_language}                                   │
│    - Max 2 sentences per response (phone call)                       │
│    - Always confirm details before calling book_table                │
│    - Never fabricate availability or pricing                         │
│    - On any API error → apologise and offer to try again             │
│                                                                      │
│  Dineout MCP Tools:                                                  │
│    search_restaurants       (location, cuisine, vibe, date, time)    │
│    check_table_availability (restaurant_id, date, time, party_size)  │
│    book_table               (restaurant_id, date, time, party_size)  │
│    get_booking              (booking_id)                             │
│    cancel_booking           (booking_id)                             │
│    get_booking_history      (user_id)                                │
│                                                                      │
│  Auth: per-user Swiggy access_token injected from Postgres lookup    │
│                                                                      │
│  Streaming: sentence-level — first sentence triggers TTS immediately │
│  Timeout:   > 3s → play filler ("ek second...") → retry once        │
│  Fallback:  2 retries fail → "let me connect you to someone"         │
│                                                                      │
│  Metrics: TTFT, containment rate, escalation rate,                   │
│           bookings placed, cancellations, FCR                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     SARVAM TTS — Bulbul v3                           │
│                                                                      │
│  target_language_code + speaker driven by classifier active_language │
│    hi → hi-IN, anand  |  en → en-IN, anand                          │
│    te → te-IN, anu    |  bn → bn-IN, anu    |  mr → mr-IN, anu      │
│                                                                      │
│  Updated mid-call if classifier detects language switch              │
│  Sentence-level streaming — first sentence plays before Groq done    │
│  Barge-in → halt audio stream immediately                            │
│                                                                      │
│  Metrics: TTS latency, MOS score (human eval)                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                         TWILIO LAYER                                 │
│                                                                      │
│  Re-encode PCM → mulaw 8kHz                                          │
│  Stream audio back to caller over WebSocket                          │
└──────────────────────────────────────────────────────────────────────┘


WEB ONBOARDING (primary — do this before calling)
  1. Enter phone number → Twilio OTP verify
  2. OTP confirmed → proves ownership of number
  3. "Connect Swiggy" → OAuth redirect
     Scopes requested: Dineout bookings + history only
     Not requested:    payments, food orders
  4. OAuth callback → exchange auth_code for tokens
  5. Write to Postgres (phone_number, tokens, expiry)
  6. Success screen: "Call [number] to start booking"


IN-CALL ONBOARDING (fallback — unknown callers)
  Caller ID not found in Postgres
  ↓
  Agent: "You'll need to link your Swiggy account first.
          Sending a link to your number now."
  ↓
  Twilio SMS → onboarding URL?ref={phone_number}
  ↓
  Agent: "I'll wait while you do that."
  ↓
  Poll Postgres every 2s, max 90s (circuit breaker: hard stop at 90s)
    ├── Token appears  → "You're linked! How can I help?"
    └── 90s timeout   → "The link stays valid — call back once done."
                        → graceful hangup


POSTGRES (token store)
  Table: user_tokens
    phone_number   PK, E.164 format (+919876543210)
    swiggy_user_id
    access_token   encrypted at rest
    refresh_token  encrypted at rest
    token_expiry   triggers refresh flow
    created_at
    updated_at

  On every call:
    token valid    → inject into Groq MCP auth header
    token expired  → refresh → update DB → proceed
    phone unknown  → trigger in-call onboarding


LATENCY BUDGET
  Silero VAD + 700ms silence          ~700ms  ← dominates, unavoidable
  Language classifier                   ~10ms
  Moonshine Tiny (ONNX int8)            ~50ms
  Groq (time to first token)           ~200ms
  Sarvam Bulbul (first sentence)       ~300ms
  ──────────────────────────────────────────
  Total to first audio byte           ~1.26s
```

---

## Benchmark Plan

### Classifier Benchmark

On held-out 10% of IndicVoices (5 target languages):

| Metric | Target |
|---|---|
| 5-class accuracy | > 95% |
| Per-language F1 | Tracked separately |
| Short utterance accuracy (< 5 words) | Tracked separately |
| Inference latency | < 15ms |

### Moonshine vs Sarvam Saaras Benchmark

On held-out 10% of IndicVoices per language:

| Metric | Moonshine Tiny (finetuned) | Sarvam Saaras v1 |
|---|---|---|
| WER — Hindi | | |
| WER — English | | |
| WER — Telugu | | |
| WER — Bengali | | |
| WER — Marathi | | |
| Macro avg WER | | |
| RTF on deployment hardware | | |
| Latency p50 / p95 | | |
| WER on noisy audio | | |
| WER on short utterances | | |
| WER on code-switched samples | | |

### Decision from benchmark

```
Moonshine WER competitive + RTF < 0.3 → ship, Sarvam not used in prod
Sarvam wins significantly on a language → use Sarvam for that language only
Moonshine RTF > 0.3                    → profile; try Moonshine Small for that lang
```

---

## Conversation Flow

```
Turn 1 — Intent + details
  "book a table for 2 at a rooftop place in Koramangala, Saturday 8pm"

Turn 2 — Search + present options
  search_restaurants(location="Koramangala", vibe="rooftop", ...)
  "Found two options — Skyye Rooftop and The Permit Room. Which one?"

Turn 3 — Check availability
  check_table_availability(restaurant_id, date, time, party_size=2)
  "Skyye has availability at 8pm for 2. Shall I confirm the booking?"

Turn 4 — Confirm + book
  book_table(...)
  "Done! Booking confirmed at Skyye Rooftop, Saturday 8pm, 2 guests.
   Booking ID: SW-4821."
```

**Supported intents:** new reservation, cancel, check existing booking, rebook past restaurant.

**Edge cases:** no availability → suggest alternatives; multiple matches → disambiguation turn; API failure → retry or escalate.

---

## Swiggy Builders Club

Access model: invite-led. Submit use case, architecture, auth redirect URIs, static IP, data-handling declaration.

**Dineout MCP tools:**

| Tool | Purpose |
|---|---|
| `search_restaurants` | Location, cuisine, vibe, date/time |
| `check_table_availability` | Party size, date, time |
| `book_table` | Place reservation |
| `get_booking` | Read booking by ID |
| `cancel_booking` | Cancel reservation |
| `get_booking_history` | Past bookings for rebooking |

**OAuth scopes:** Dineout bookings + history only. No payment or food order scopes.

---

## Database

```sql
CREATE TABLE user_tokens (
  phone_number     VARCHAR(15) PRIMARY KEY,  -- E.164
  swiggy_user_id   TEXT        NOT NULL,
  access_token     TEXT        NOT NULL,     -- encrypted at rest
  refresh_token    TEXT        NOT NULL,     -- encrypted at rest
  token_expiry     TIMESTAMPTZ NOT NULL,
  created_at       TIMESTAMPTZ DEFAULT now(),
  updated_at       TIMESTAMPTZ DEFAULT now()
);
```

---

## Metrics

### System Health

| Metric | Target |
|---|---|
| End-to-end latency | < 1.5s |
| Time to first audio byte | < 800ms |
| VAD false positive rate | < 5% |
| VAD false negative rate | < 3% |
| Barge-in response time | < 200ms |
| Classifier accuracy (5-class) | > 95% |
| Classifier latency | < 15ms |
| Moonshine WER per language | Tracked separately |
| Moonshine macro WER | < 15% |
| Moonshine RTF | < 0.3 |
| TTS naturalness (MOS) | > 3.5/5 |
| WebSocket drop rate | < 1% |

### Business

| Metric | Target |
|---|---|
| Call containment rate | > 70% |
| Escalation rate | < 30% |
| First Call Resolution (FCR) | > 60% |
| CSAT | > 3.8/5 |
| Call abandonment rate | < 8% |
| Repeat call rate (7 days) | < 20% |
| Onboarding conversion | Track from day 1 |
| Cost per call | Track from day 1 |

---

## Test Plan

### Functional Tests

| Test | Description |
|---|---|
| Happy path | Hindi query → classifier → Moonshine → Groq → booking placed |
| Barge-in | User speaks mid-TTS, TTS stops, new utterance processed correctly |
| Mid-call language switch | Starts Hindi, switches to Telugu — classifier detects, Bulbul voice updates |
| Short utterances | "haan", "ok", "nahi" — classifier must not update active_language |
| Low confidence routing | conf < 0.80 — keep previous language, don't misroute to wrong Moonshine |
| First utterance default | Unknown caller, low confidence first word → defaults to Hindi |
| Silence | User quiet 5s — agent prompts, no crash |
| Repeated barge-in | 3 interruptions — no state corruption |
| Reorder flow | "Book that place I went to last month" → history lookup |
| Cancellation | Cancel by booking ID |

### Classifier Benchmark (IndicVoices held-out)

| Sub-test | Description |
|---|---|
| 5-class accuracy | Per-language + overall |
| Short utterance slice | < 5 word samples |
| Code-switched samples | Hinglish, Telugu-English etc. |
| Noisy audio slice | Synthetic noise added to simulate Twilio quality |
| Confusion matrix | Which languages get confused with each other |

### Moonshine Benchmark (IndicVoices held-out vs Sarvam Saaras)

Per-language WER, macro WER, RTF, latency p50/p95, WER on noisy/short/code-switched audio.

### Load Tests

| Test | Description |
|---|---|
| Concurrent calls | 10, 50, 100 simultaneous WebSocket connections |
| Latency under load | p95 < 2s at 50 concurrent |
| Model inference under load | Classifier + Moonshine RTF stays < 0.3 |
| API rate limits | Groq, Swiggy MCP — behaviour when limits hit |

### Edge Cases

| Case | Expected behaviour |
|---|---|
| Unsupported language caller | Classifier low confidence → Hindi default → agent asks to speak one of 5 supported languages |
| Angry caller, fast speech | VAD and classifier still function |
| Background noise | Silero VAD suppresses, classifier degrades gracefully |
| Empty transcript | Skip pipeline, re-open mic |
| Groq timeout | Filler audio, retry once, escalate on second failure |
| Swiggy MCP error | Apologise, retry — never fabricate a booking |
| Polling circuit breaker | In-call onboarding hard-stops at 90s |

### Regression (every deploy)

- Classifier accuracy does not drop
- Per-language Moonshine WER does not degrade
- Moonshine RTF does not increase > 10%
- E2E latency p95 does not increase > 10%
- Onboarding SMS delivery success > 95%

---

## Observability

**Per-call trace:** latency at each stage, full transcript + classifier output (language + confidence) per utterance + Groq tool calls + TTS language used + outcome.

**Business dashboard:** bookings placed, cancellations, repeat callers, onboarding conversion, containment rate, escalation rate, FCR, cost per call, CSAT.

**Alerts:** p95 latency > 2s, classifier accuracy drop, Moonshine WER regression, Moonshine RTF spike, Swiggy MCP error rate spike, WebSocket drop rate > 1%, SMS delivery failure > 5%.

---

## Build Order

**Track 1 — Auth + Onboarding (independent)**
Postgres schema → web onboarding UI → Twilio OTP → Swiggy OAuth → token storage + refresh. Unblocks all Dineout MCP calls.

**Track 2 — Classifier (independent)**
Filter IndicVoices to 5 languages → balance classes → train audio classifier → evaluate accuracy + confusion matrix → quantize → benchmark latency.

**Track 3 — Moonshine Finetuning (independent)**
Per-language finetuning loop on IndicVoices → per-language WER tracking → ONNX export → int8 quantize → RTF benchmark vs Sarvam Saaras on held-out test set. All 5 models need to be ready before Track 4.

**Track 4 — Voice Pipeline (depends on Track 2 + 3)**
Twilio WebSocket → mulaw resampling → Silero VAD → classifier → Moonshine routing → Groq + Swiggy MCP → Sarvam Bulbul → Twilio output. Test with stub LLM before wiring Groq.

**Integration**
Wire Track 1 + Track 4. Add Groq + Dineout MCP. Full happy-path end-to-end.
