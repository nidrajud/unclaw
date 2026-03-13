# Unclaw

**OpenClaw spirit, rebuilt as a local-first, lightweight agent runtime for real consumer hardware.**

Unclaw is a practical Python agent shell for running small and medium local models without mandatory cloud services, recurring platform lock-in, or opaque hosted orchestration. It focuses on a sharp terminal experience, a useful Telegram control channel, simple local persistence, and a clean runtime you can inspect, modify, and extend.

Today, Unclaw already gives you:

- a local-first runtime backed by Ollama
- a polished terminal experience with `unclaw start`
- a Telegram bot channel with guided setup
- live local logs with `unclaw logs simple` and `unclaw logs full`
- model routing across `fast`, `main`, `deep`, and `codex` profiles
- built-in tool commands such as `/read`, `/ls`, `/fetch`, and `/tools`
- local sessions, summaries, and a lightweight memory base
- project-local Telegram secrets storage
- no mandatory cloud dependency

It is intentionally transparent, modular, and practical rather than pretending to be magic.

## Why Unclaw

Many agent projects assume hosted infra, heavyweight stacks, or expensive models by default. Unclaw takes the opposite path:

- **Local-first by design.** Your runtime, sessions, logs, and secrets stay on your machine.
- **Built for realistic hardware.** The defaults target small and medium Ollama models you can actually run.
- **Simple where it matters.** Clear commands, clear config, clear logs, no giant framework tax.
- **Useful now, extensible later.** Terminal chat, Telegram control, tools, memory, and live traces already work.
- **No cloud lock-in.** You can inspect the code, change the prompts, swap models, and evolve the workflow locally.

## Current Features

### Runtime and UX

- Guided onboarding with `unclaw onboard`
- Interactive terminal runtime with `unclaw start`
- Streaming terminal replies
- Shared session system with local persistence
- Session summaries and memory refresh
- Polished banners and startup/preflight output
- Safe local update flow with `unclaw update`

### Models and Routing

- Four shipped model profiles: `fast`, `main`, `deep`, `codex`
- Live model selection with `/model`
- Optional thinking mode with `/think on|off`
- Runtime routing across local profiles
- Defaults tuned for Ollama-backed local models

### Tools

- `/tools` to inspect available built-in tools
- `/read <path>` to read one local file
- `/ls [path]` to inspect one local directory
- `/fetch <url>` to fetch a URL
- Structured tool execution traces in the local log stream

### Channels

- Terminal channel
- Telegram bot channel sharing the same local runtime and sessions
- Telegram slash-command support, including session and tool commands

### Observability

- `unclaw logs simple` for a concise human view
- `unclaw logs full` for the structured runtime stream
- Model, tool, routing, and Telegram events recorded locally

## Quick Start

### 1. Install

Unclaw targets **Python 3.12+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2. Install and start Ollama

Unclaw is built around a local Ollama runtime.

```bash
ollama serve
```

Pull at least the default model if you want to skip guided setup:

```bash
ollama pull qwen3.5:4b
```

### 3. Run onboarding

```bash
unclaw onboard
```

The onboarding flow helps you:

- choose channels
- confirm the model lineup
- store a Telegram bot token locally if you want the Telegram channel
- write local config files for the current project

### 4. Start the local runtime

```bash
unclaw start
```

Useful companion commands:

```bash
unclaw logs simple
unclaw logs full
unclaw update
```

## Model Profiles

Unclaw ships with four practical profiles in `config/models.yaml`:

| Profile | Default model | Purpose |
| --- | --- | --- |
| `fast` | `llama3.2:3b` | Quick low-cost replies |
| `main` | `qwen3.5:4b` | Default everyday assistant profile |
| `deep` | `qwen3.5:9b` | Heavier reasoning when you want more depth |
| `codex` | `qwen2.5-coder:7b` | Coding-oriented local profile |

Switch profiles at runtime:

```text
/model
/model fast
/model deep
/think on
/think off
```

## Channels

### Terminal

The terminal channel is the core Unclaw experience:

- startup preflight checks
- session-aware chat
- streaming assistant replies
- slash commands for models, sessions, memory, and tools

Start it with:

```bash
unclaw start
```

### Telegram

Telegram is for remote control of the same local runtime, not for turning Unclaw into a cloud service.

Start it with:

```bash
unclaw telegram
```

What already works:

- guided Telegram token onboarding
- local token storage in `config/secrets.yaml`
- shared slash commands
- shared sessions and logs
- secure-by-default authorization

Telegram is now intentionally **deny-by-default**:

- `allowed_chat_ids: []` means **no chats are authorized**
- unauthorized chats are rejected without running the model
- rejected chat IDs are logged so you can allowlist them later

To authorize Telegram access, edit `config/telegram.yaml` and add numeric chat IDs:

```yaml
bot_token_env_var: TELEGRAM_BOT_TOKEN
polling_timeout_seconds: 30
allowed_chat_ids:
  - 123456789
```

## Security Philosophy

Unclaw is local-first, but local-first does not mean careless. The current direction is:

- **secure defaults first**
- **dangerous behavior only by explicit opt-in**
- **simple mechanisms before complicated security theater**

Current shipped defaults include:

- Telegram access is deny-by-default unless a chat ID is explicitly allowlisted
- `config/secrets.yaml` is written with owner-only permissions
- reasoning text is not logged unless `logging.include_reasoning_text: true`
- runtime logs remain useful without storing more sensitive detail than necessary

This is not the end of the hardening work. It is the foundation for stronger permissions, safer tools, and clearer policy controls.

## Why Local-First

Local-first matters for more than cost.

- **Privacy:** your sessions, logs, summaries, and secrets remain under your control
- **Transparency:** you can inspect exactly how the runtime behaves
- **Reliability:** no required hosted control plane
- **Modularity:** swap models and evolve the runtime without waiting on a vendor roadmap
- **Practicality:** small and medium local models are improving quickly, and they are already useful for real workflows

Unclaw is designed for that reality.

## Project Status

Unclaw is an early but real local runtime. It already supports:

- local-first execution
- terminal and Telegram channels
- guided onboarding
- live logs
- session persistence
- memory summaries
- built-in local file and web tools

It is not pretending to be finished. Current work in progress includes:

- security hardening
- stronger permissions for tools
- better safe defaults across channels
- richer policy and prompt management
- a broader tool ecosystem
- more agentic behaviors
- stronger audit coverage and tests
- additional channels and messengers later

The roadmap is about making the runtime more trustworthy and more useful without losing the lightweight local-first core.

## Contributing / Current State

Contributions are welcome, especially if you care about:

- local model ergonomics
- clean Python architecture
- runtime observability
- safety and permission boundaries
- practical agent workflows on consumer hardware

Before opening large changes, keep the current shape of the project in mind:

- small and readable beats clever
- local-first is a product constraint, not marketing copy
- shipping honest behavior matters more than making grand claims
- better defaults are preferred over optional complexity

## Command Snapshot

```bash
unclaw start
unclaw telegram
unclaw onboard
unclaw logs simple
unclaw logs full
unclaw update
```

Inside the runtime, useful slash commands include:

```text
/help
/new
/sessions
/use <session_id>
/model <profile>
/think on
/think off
/tools
/read README.md
/ls .
/fetch https://example.com
/session
/summary
```

## License

Apache-2.0
