# Dineout Voice Reservation Agent — Design Document

---

## Overview

A multilingual AI voice agent that lets Indian users call a phone number and book restaurant tables on Swiggy Dineout using natural speech — in Hindi, Tamil, Telugu, Kannada, or any other supported Indian language. The system handles the full reservation flow: discovery, availability checks, booking, cancellations, and rebooking history.

Built on a DIY stack: Twilio for telephony, a finetuned Whisper model for STT and language detection, Groq for LLM inference, Sarvam Bulbul for TTS, and Swiggy Builders Club MCP for Dineout actions.

---

## Stack at a Glance

| Layer | Tool | Notes |
|---|---|---|
| Telephony | Twilio Media Streams | WebSocket, mulaw 8kHz |
| VAD | Silero VAD | Per 30ms frame, barge-in handling |
| STT + Language Detection | Finetuned Whisper (whisper-small) | Single model, transcript + language token in one pass |
| STT Benchmark Target | Sarvam Saaras v1 | Commercial baseline to beat |
| LLM + Tool Use | Groq | Fast inference, streamed output |
| Commerce Layer | Swiggy Builders Club — Dineout MCP | 6 tools, real transactions |
| TTS | Sarvam Bulbul v3 | Language-matched voice, driven by Whisper's detected language |
| Token Store | PostgreSQL | Caller ID → Swiggy OAuth token |

---

## Why Finetuned Whisper (not Sarvam STT + a separate LID model)

Every other STT + language detection combination has a fundamental problem:

```
Text-based LID (IndicLID):
  STT needs language → to produce transcript
  LID needs transcript → to detect language
  → Chicken-and-egg. Wrong language in = garbage transcript = wrong LID output.

Audio-based LID (facebook/mms-lid):
  Correct direction but two separate models, two inference calls, added latency.
  You don't own or control either model.

Sarvam auto-detect:
  Black box — no visibility into confidence, no way to improve it,
  and no guarantee the detected language is exposed in the API response.
```

Whisper solves all of this in a single forward pass. Its decoder predicts a language token **before** transcribing — so transcript and language come out together, from raw audio, with no dependency on each other:

```
Audio → Finetuned Whisper → [<|ta|> "naan oru table book pannanum"]
                              ↑ language token  ↑ transcript
                              both in one pass, no chicken-and-egg
```

Sarvam Saaras becomes the **benchmark target**, not a dependency. You pay per-call for Sarvam. You pay once for compute to run your own Whisper. At any meaningful call volume, self-hosted Whisper wins on cost and you own the model.

---

## Why Not Moonshine

Moonshine was evaluated as an alternative to Whisper. It is genuinely faster — but for this specific use case, the speed advantage comes at a cost that makes it the wrong choice.

### What Moonshine Gets Right

Moonshine was built specifically for real-time, edge-deployed ASR. Its key architectural innovation is a variable-length encoder — unlike Whisper, which pads all inputs to a fixed 30-second window regardless of actual audio length and pays the full compute cost every time, Moonshine only processes what's actually there. This produces substantial real-world speedups:

```
Moonshine v2 Small:  148ms latency — 13.1x faster than Whisper Small
Moonshine v2 Tiny:    50ms latency —  5.8x faster than Whisper Tiny
```

There is also prior art on finetuning it for non-English languages — a French finetuning repo exists and achieved 21.8% WER, which establishes that the finetuning pathway is viable.

### Why It Doesn't Work Here

**Moonshine has no language token.** Whisper's decoder is trained to predict a language identifier as its first output before generating any transcript. That single architectural feature is the entire reason Whisper was chosen — it gives STT and language detection in one forward pass with no chicken-and-egg dependency.

Moonshine is designed as a monolingual model. The research backing it ("Flavors of Moonshine") explicitly shows that its accuracy gains come from training a single-language model per target language, not a multilingual one. A multilingual Moonshine finetuned across 22 Indian languages would lose that advantage and still produce no language token.

```
Finetuned Whisper:
  Audio → [<|ta|> "naan oru table book pannanum"]
           ↑ language  ↑ transcript
           one pass, no separate model needed

Finetuned Moonshine:
  Audio → "naan oru table book pannanum"
           no language signal at all
           → need a separate audio LID model
           → two inference calls, added latency
           → chicken-and-egg problem returns
```

### When Moonshine Would Be the Right Call

If this were a Hindi-only or Tamil-only agent — single language, no detection needed — Moonshine would be the correct choice. Faster, smaller, better edge performance, and no language ID overhead. But the multilingual requirement is load-bearing for this product, and Whisper's language token is the cleanest solution to it.

### Summary

| | Finetuned Whisper | Finetuned Moonshine |
|---|---|---|
| Language token (LID built-in) | ✓ — free, one pass | ✗ — no equivalent |
| Multilingual design | ✓ | ✗ — monolingual wins |
| Speed | Good (RTF ~0.25 int8) | Better (5–13x faster) |
| Edge deployment | ✓ | ✓ better |
| Works for this use case | ✓ | ✗ |

Speed is not the bottleneck here anyway — the 700ms VAD silence window before STT runs dominates latency, not inference time. Whisper with CTranslate2 int8 quantization already hits the RTF target.

---

## Finetuning Plan

### Dataset — `ai4bharat/IndicVoices`

- Real naturalistic speech, not studio recordings
- 22 Indian languages, multiple speakers
- Code-switching samples (Hinglish, Tanglish)
- Varied acoustic conditions — closer to real phone call audio than most datasets

Split strategy:
```
Train:      80%
Validation: 10%  ← hyperparameter tuning, early stopping
Test:       10%  ← held-out, never touched until final benchmark day
```

### Base Model

```
whisper-small  (244M params) → good balance, edge-deployable  ← start here
whisper-medium (769M params) → better accuracy, heavier       ← fallback if WER gap too large
```

### Training

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

dataset = load_dataset("ai4bharat/IndicVoices", split="train")

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    generation_max_length=225,
    learning_rate=1e-5,
    warmup_steps=500,
    gradient_checkpointing=True,   # saves memory
    fp16=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
)
```

### What to track during training

- **Per-language WER** — log separately per language after every eval epoch. Some languages will be underrepresented in IndicVoices and will perform worse. Don't let them hide in the aggregate.
- **Language token accuracy** — track separately from transcript WER. These can diverge.
- **Code-switching samples** — Whisper predicts one language token but audio is mixed. Track these as a separate slice.

---

## Edge Optimization

### Step 1 — faster-whisper (CTranslate2)

Convert finetuned model to CTranslate2 format. 2–4x speedup, no accuracy loss:

```python
from ctranslate2.converters import OpenAIWhisperConverter

converter = OpenAIWhisperConverter("./finetuned-whisper-small")
converter.convert("./whisper-small-ct2", quantization="float16")
```

Inference:
```python
from faster_whisper import WhisperModel

model = WhisperModel("./whisper-small-ct2", device="cpu", compute_type="int8")
segments, info = model.transcribe(audio, beam_size=5)

detected_language = info.language            # e.g. "ta", "hi", "kn"
lang_confidence   = info.language_probability # used for short-utterance guard
transcript        = " ".join([s.text for s in segments])
```

### Step 2 — Quantization levels

| Level | Speedup | WER impact | Decision |
|---|---|---|---|
| `float16` | ~2x | Negligible | Start here |
| `int8` | ~4x | Small | Test and benchmark |
| `int4` | ~6x | Noticeable | Probably too much |

Benchmark WER at each level on the held-out test set. Pick the fastest level where WER stays within acceptable range.

### Step 3 — ONNX (if truly edge-constrained)

```python
from optimum.exporters.onnx import main_export

main_export(
    "./finetuned-whisper-small",
    output="./whisper-onnx",
    task="automatic-speech-recognition"
)
```

**RTF (Real-Time Factor)** is the key deployment metric. RTF < 1.0 = faster than real-time, which is the minimum bar for a live voice agent. Target RTF < 0.3 on deployment hardware.

---

## Using Detected Language for TTS

Whisper's language token drives the Sarvam Bulbul TTS call directly. No separate language routing logic needed — the detected language maps straight to `target_language_code` and `speaker`:

```python
LANGUAGE_TO_BULBUL = {
    "hi": {"target_language_code": "hi-IN", "speaker": "anand"},
    "ta": {"target_language_code": "ta-IN", "speaker": "shubh"},
    "te": {"target_language_code": "te-IN", "speaker": "anu"},
    "kn": {"target_language_code": "kn-IN", "speaker": "anu"},
    "ml": {"target_language_code": "ml-IN", "speaker": "anu"},
    "bn": {"target_language_code": "bn-IN", "speaker": "anu"},
    "mr": {"target_language_code": "mr-IN", "speaker": "anu"},
    "gu": {"target_language_code": "gu-IN", "speaker": "anu"},
    "pa": {"target_language_code": "pa-IN", "speaker": "anu"},
    "en": {"target_language_code": "en-IN", "speaker": "anand"},
}

def get_tts_params(whisper_lang: str, confidence: float, utterance_word_count: int):
    # Short utterance or low confidence → don't update, keep previous voice
    if confidence < 0.80 or utterance_word_count <= 5:
        return None
    return LANGUAGE_TO_BULBUL.get(whisper_lang, LANGUAGE_TO_BULBUL["hi"])

# On each utterance:
tts_params = get_tts_params(detected_language, lang_confidence, word_count)
if tts_params:
    active_tts_params = tts_params  # update voice for this and subsequent turns

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

The agent effectively mirrors the caller's language automatically — they speak Tamil, Whisper detects `"ta"`, Bulbul responds in Tamil. Mid-call switches are caught on the next utterance and the voice updates accordingly.

---

## Benchmark Plan — Finetuned Whisper vs Sarvam Saaras

On the held-out 10% of IndicVoices (never seen during training):

| Metric | Finetuned Whisper | Sarvam Saaras |
|---|---|---|
| WER per language | | |
| Macro avg WER | | |
| Language ID accuracy | | |
| RTF (Real-Time Factor) | | |
| Latency p50 / p95 | | |
| WER on noisy audio (+synthetic noise) | | |
| WER on short utterances (< 5 words) | | |
| WER on code-switched samples | | |

### Decision tree from benchmark results

```
Finetuned Whisper wins WER + RTF < 0.3  → ship it, Sarvam STT not used in prod
Sarvam wins WER, Whisper RTF acceptable  → dig into which languages Whisper loses
Whisper RTF > 0.5                        → move to whisper-medium or optimize more
Both roughly equal                       → prefer Whisper (no API cost, you own it)
```

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
│    └── flush buffer → trigger STT pipeline                           │
│                                                                      │
│  Metrics: false positive rate, false negative rate,                  │
│           barge-in response time < 200ms                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│            FINETUNED WHISPER (faster-whisper, CTranslate2)           │
│                                                                      │
│  Base:   openai/whisper-small                                        │
│  Tuned:  ai4bharat/IndicVoices (80/10/10 split)                      │
│  Format: CTranslate2, int8 quantization                              │
│                                                                      │
│  Input:  buffered 16kHz PCM (full utterance after VAD silence)       │
│  Output: transcript text                                             │
│          info.language          → detected language code ("hi","ta") │
│          info.language_probability → confidence score                │
│                                                                      │
│  Guard:  empty transcript → skip pipeline, re-open mic               │
│                                                                      │
│  Language routing rules:                                             │
│    conf > 0.80 AND utterance > 5 words                               │
│      → update active_language                                        │
│      → update active_tts_params (target_language_code + speaker)     │
│      → inject active_language into Groq system prompt context        │
│    conf < 0.80 OR utterance ≤ 5 words                                │
│      → keep previous active_language and active_tts_params           │
│                                                                      │
│  Metrics: WER per language, macro WER, language ID accuracy,         │
│           RTF, short-utterance accuracy tracked separately           │
│                                                                      │
│  Benchmark: held-out 10% of IndicVoices vs Sarvam Saaras v1         │
│    WER per language, macro WER, language ID accuracy, RTF, latency   │
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
│  target_language_code: from active_tts_params["target_language_code"]│
│  speaker:              from active_tts_params["speaker"]             │
│  Both driven by Whisper's detected language token                    │
│                                                                      │
│  Language → voice mapping (example defaults):                        │
│    hi → hi-IN, anand   ta → ta-IN, shubh                            │
│    te → te-IN, anu     kn → kn-IN, anu                              │
│    ml → ml-IN, anu     bn → bn-IN, anu                              │
│    en → en-IN, anand                                                 │
│                                                                      │
│  Sentence-level streaming — don't wait for full Groq response        │
│  Barge-in signal from VAD → halt audio stream immediately            │
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
    │                   → resume normal call flow
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
  Silero VAD + 700ms silence      ~700ms
  Finetuned Whisper (int8, CPU)   ~250ms   ← RTF target < 0.3
  Groq (time to first token)      ~200ms
  Sarvam Bulbul (first sentence)  ~300ms
  ─────────────────────────────────────────
  Total to first audio byte       ~1.45s
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
  book_table(restaurant_id, date="Saturday", time="20:00", party_size=2)
  "Done! Booking confirmed at Skyye Rooftop, Saturday 8pm, 2 guests.
   Booking ID: SW-4821."
```

**Supported intents:**
- New reservation
- Cancel reservation (`cancel_booking`)
- Check existing booking (`get_booking`)
- Rebook a past restaurant (`get_booking_history` → `book_table`)

**Edge cases:**
- No availability at requested time → suggest alternative slots
- Multiple restaurants match → disambiguation turn
- Restaurant not found → broaden search or clarify
- Booking API failure → apologise, offer retry or escalate

---

## Swiggy Builders Club

Access model: invite-led. Submit use case, integration architecture, auth redirect URIs, static IP, and data-handling declaration.

**Dineout MCP tools used:**

| Tool | Purpose |
|---|---|
| `search_restaurants` | Location, cuisine, vibe, date/time filters |
| `check_table_availability` | Party size, date, time for a specific restaurant |
| `book_table` | Place the reservation |
| `get_booking` | Read back booking details by ID |
| `cancel_booking` | Cancel an existing reservation |
| `get_booking_history` | Fetch past bookings for rebooking |

**OAuth scopes:** Dineout bookings + history only. Do not request food order or payment scopes — overbroad permissions hurt onboarding conversion.

**Ground rules (ToS):** Cannot resell MCP access, scrape beyond APIs, misrepresent pricing/availability, or obscure Swiggy's brand.

---

## Database

```sql
CREATE TABLE user_tokens (
  phone_number     VARCHAR(15) PRIMARY KEY,  -- E.164, e.g. "+919876543210"
  swiggy_user_id   TEXT        NOT NULL,
  access_token     TEXT        NOT NULL,     -- encrypted at rest
  refresh_token    TEXT        NOT NULL,     -- encrypted at rest
  token_expiry     TIMESTAMPTZ NOT NULL,
  created_at       TIMESTAMPTZ DEFAULT now(),
  updated_at       TIMESTAMPTZ DEFAULT now()
);
```

**Token resolution on every call:**
```python
async def resolve_user(caller_id: str):
    record = await db.fetchrow(
        "SELECT * FROM user_tokens WHERE phone_number = $1", caller_id
    )
    if not record:
        return None  # trigger onboarding

    if record["token_expiry"] < now():
        new_tokens = await swiggy_oauth.refresh(record["refresh_token"])
        await db.execute(
            "UPDATE user_tokens SET access_token=$1, refresh_token=$2, "
            "token_expiry=$3, updated_at=now() WHERE phone_number=$4",
            new_tokens.access, new_tokens.refresh,
            new_tokens.expiry, caller_id
        )
        return new_tokens.access

    return record["access_token"]
```

**Security:**
- Tokens encrypted at rest
- Tokens never logged in call traces
- Phone number is the only identity key — Twilio already verified caller ownership

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
| Whisper WER (macro avg) | < 15% |
| Whisper WER per language | Tracked separately |
| Whisper language ID accuracy | > 0.90 macro F1 |
| Whisper RTF | < 0.3 on deployment hardware |
| TTS naturalness (MOS) | > 3.5/5 |
| WebSocket drop rate | < 1% |

### Business

| Metric | Target |
|---|---|
| Call containment rate | > 70% |
| Escalation rate | < 30% |
| First Call Resolution (FCR) | > 60% |
| Customer Satisfaction (CSAT) | > 3.8/5 |
| Call abandonment rate | < 8% |
| Repeat call rate (same issue, 7 days) | < 20% |
| Onboarding conversion (SMS sent → linked) | Track from day 1 |
| Cost per call | Track from day 1 |

---

## Test Plan

### Functional Tests

| Test | Description |
|---|---|
| Happy path | Hindi query, full pipeline, correct booking placed |
| Barge-in | User speaks mid-TTS, TTS stops, new utterance processed correctly |
| Mid-call language switch | Starts Hindi, switches to Tamil — Whisper detects, Bulbul voice updates |
| Code-switching | Hinglish, Tanglish — Whisper returns one language token, track accuracy |
| Short utterances | "haan", "ok", "nahi" — confidence guard must prevent lang update |
| Silence | User quiet for 5s — agent prompts, no crash |
| Long utterance | 30s monologue — VAD buffering and Whisper handle correctly |
| Repeated barge-in | User interrupts 3 times — no state corruption |
| Reorder flow | "Book that place I went to last month" — history lookup works |
| Cancellation | User cancels existing booking by booking ID |

### Whisper Benchmark (IndicVoices held-out 10%)

| Sub-test | Finetuned Whisper | Sarvam Saaras |
|---|---|---|
| WER per language | | |
| Macro avg WER | | |
| Language ID accuracy | | |
| RTF on deployment hardware | | |
| Latency p50 / p95 | | |
| WER on noisy audio (synthetic noise) | | |
| WER on short utterances (< 5 words) | | |
| WER on code-switched samples | | |

### Quantization Benchmark

| Level | RTF | WER delta vs float32 | Decision |
|---|---|---|---|
| float16 | | | |
| int8 | | | |
| int4 | | | |

### Stress / Load Tests

| Test | Description |
|---|---|
| Concurrent calls | 10, 50, 100 simultaneous WebSocket connections |
| Latency under load | p95 latency stays < 2s at 50 concurrent calls |
| Whisper under load | RTF stays < 0.3 at concurrent inference load |
| API rate limits | Groq, Swiggy MCP — behaviour when limits hit |
| WebSocket reconnect | Drop mid-call, verify graceful session handling |

### Edge Cases

| Case | Expected behaviour |
|---|---|
| Unsupported language | Whisper returns low confidence — keep previous language, ask caller to speak Hindi or English |
| Angry caller, fast speech | VAD and Whisper still function |
| Background noise | Silero VAD suppresses, Whisper degrades gracefully |
| Empty transcript | Skip pipeline, re-open mic — don't send blank to Groq |
| Groq timeout | Play filler ("ek second..."), retry once, escalate on second failure |
| Swiggy MCP error | Apologise, offer retry — never fabricate a booking confirmation |
| No availability | Suggest alternative times or nearby restaurants |
| Polling circuit breaker | In-call onboarding hard-stops at 90s |

### Regression (every deploy)

- Whisper WER does not degrade on benchmark set
- Language ID macro F1 does not drop
- RTF does not increase > 10%
- E2E latency p95 does not increase > 10%
- Onboarding SMS delivery success rate stays > 95%

---

## Observability

**Per-call trace:**
- Latency at each pipeline stage
- Full transcript per call: Whisper output + detected language + confidence per utterance + Groq tool calls made
- TTS language/speaker used per turn
- Outcome: booking placed / cancelled / escalated / abandoned

**Business metrics dashboard:**
- Bookings placed, cancellations, repeat callers
- Onboarding conversion rate (SMS sent → account linked)
- Web vs in-call onboarding split
- Containment rate, escalation rate, FCR
- Cost per call (Twilio + Sarvam Bulbul + Groq — Whisper runs locally)
- CSAT via post-call SMS ("Rate your experience 1–5")

**Alerts:**
- p95 latency > 2s
- Whisper WER regression on benchmark set
- Language ID F1 drop
- Whisper RTF spike (model inference slowing down)
- Swiggy MCP error rate spike
- WebSocket drop rate > 1%
- Onboarding SMS delivery failure rate > 5%

---

## Build Order

**Track 1 — Auth + Onboarding (independent)**
Postgres schema → web onboarding UI → Twilio OTP → Swiggy OAuth → token storage and refresh. Unblocks all Dineout MCP calls.

**Track 2 — Whisper Finetuning (independent)**
Pull IndicVoices → preprocess audio to 16kHz → finetune whisper-small → evaluate per-language WER + language ID accuracy → convert to CTranslate2 → quantize → benchmark RTF. Run Sarvam Saaras on the same held-out test set and compare.

**Track 3 — Voice Pipeline (depends on Track 2 model)**
Twilio WebSocket handler → mulaw resampling → Silero VAD → finetuned Whisper inference → language routing → Groq + Swiggy MCP → Sarvam Bulbul TTS → Twilio output. Test with a stub LLM before wiring Groq and Swiggy MCP.

**Integration**
Wire Track 1 + Track 3 (caller ID → token resolution at call start). Add Groq + Swiggy Dineout MCP. Run full happy-path flow end-to-end.
