<p align="center">
  <img src="./unclaw-animated.svg" alt="Unclaw animated logo" />
</p>

<h1 align="center">Unclaw 🦐</h1>

<p align="center"><strong>Local-first AI runtime for a real personal assistant.</strong></p>

<p align="center">No cloud. No lock-in. No unnecessary framework bloat.</p>

Unclaw is a lightweight runtime for local AI models, built to become a **real autonomous personal agent** that runs on your own machine.

Today, it already gives you a solid local runtime with terminal usage, Telegram remote access, local tools, session persistence, logs, and safer defaults.

The direction is simple: **turn a strong local chatbot MVP into a serious local-first AI agent** — while staying lightweight, private, secure, and usable on normal personal hardware.

---

## Why Unclaw

Most “AI agent” projects assume cloud APIs, huge frameworks, and expensive hosted models.

Unclaw takes the opposite path:

- **local-first**
- **privacy-first**
- **secure-by-default**
- **lightweight**
- **open source**
- **built for real local models and real computers**

---

## Project status

### What already works

- [x] Local runtime with Ollama
- [x] Interactive terminal via `unclaw start`
- [x] Telegram remote control
- [x] Multiple local model profiles
- [x] Session persistence and local memory foundations
- [x] Local file tools
- [x] Web fetch / search foundations
- [x] Runtime logs and tracing
- [x] Guided onboarding
- [x] Safer local defaults

### What is being built now

- [ ] True autonomous agent behavior
- [ ] Intelligent tool use without requiring slash commands
- [ ] Better model-aware behavior for tool-friendly vs non-tool-friendly models
- [ ] Cleaner agent architecture after MVP audit
- [ ] Stronger search quality and grounding

### What comes next

- [ ] Easier install for non-technical users
- [ ] More messaging channels (Signal, WhatsApp, ...)
- [ ] Better document and app actions
- [ ] Local research memory / caching
- [ ] More capable personal assistant workflows

---

## What Unclaw is today

Unclaw is currently an **early but real MVP**.

It is already useful.
But it is **not pretending to be a finished autonomous agent yet**.

That is exactly what the next transformation phase is about.

---

## Quick start

Unclaw targets **Python 3.12+** and uses **Ollama** for local models.

```bash
 git clone https://github.com/nidrajud/unclaw.git
 cd unclaw
 python3.12 -m venv .venv
 source .venv/bin/activate
 python -m pip install -e .[dev]
 ollama serve
 unclaw onboard
 unclaw start
```

---

## Main commands

```bash
unclaw start
unclaw telegram
unclaw onboard
unclaw logs
unclaw logs full
unclaw update
unclaw help
```

### In-session commands

```bash
/help
/new
/sessions
/use <session_id>
/model
/model fast
/model main
/model deep
/model codex
/think on
/think off
/tools
/read <path>
/ls [path]
/fetch <url>
/search <query>
/session
/summary
```

---

## Current model profiles

- **fast** → quick lightweight replies
- **main** → default everyday assistant
- **deep** → heavier reasoning
- **codex** → code-oriented tasks

Default lineup:

- `fast` → `llama3.2:3b`
- `main` → `qwen3.5:4b`
- `deep` → `qwen3.5:9b`
- `codex` → `qwen2.5-coder:7b`

---

## Security mindset

Unclaw follows one rule:

**safe defaults first, more power later by explicit choice.**

Current protections already include deny-by-default Telegram access, local secrets handling, local file boundaries, and safer web access defaults.

---

## Long-term goal

Unclaw aims to become a **serious local AI assistant for everyday use**:

- search the web
- summarize information
- read and manipulate files
- help write code and text
- interact through multiple channels
- stay private, local, and under user control

If the project succeeds, Unclaw should feel like having a small personal assistant on your own machine — not renting one from the cloud.
