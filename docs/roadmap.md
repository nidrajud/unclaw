# Roadmap

## Project strategy

The project is built in layers.

The first goal is not to ship everything. The first goal is to ship a strong local-first core that is:
- useful
- fast
- clean
- observable
- extensible

Once the core is solid, more capabilities can be added safely.

## Phase 0 - Project foundation

Goal:
- create a clean repository
- define project direction
- define architecture
- define coding guidelines

Deliverables:
- README
- core docs
- repository structure
- local Python environment
- first commits

## Phase 1 - MVP core runtime

Goal:
- build a minimal but real agent runtime

Scope:
- CLI chat
- session management
- model provider abstraction
- model profiles
- command handler
- routing
- live logging
- simple local persistence

Deliverables:
- terminal interaction
- slash commands
- fast and deep modes
- structured runtime flow

## Phase 2 - Tools and file/web capabilities

Goal:
- make the agent useful beyond chat

Scope:
- web search
- URL fetch
- file reading
- file listing
- tool registry
- tool dispatcher
- structured tool execution

Deliverables:
- real agent tasks
- visible tool traces
- better routing outcomes

## Phase 3 - Basic memory

Goal:
- allow the agent to keep useful state without bloating prompts

Scope:
- session summaries
- user memory basics
- project memory basics
- memory selection
- memory writing rules

Deliverables:
- memory-aware conversations
- better continuity
- no blind memory injection

## Phase 4 - Telegram channel

Goal:
- interact with the agent outside the terminal

Scope:
- Telegram bot integration
- same sessions and commands as CLI
- shared runtime
- shared logging model

Deliverables:
- local-first agent accessible from Telegram

## Phase 5 - Stabilization and polish

Goal:
- make the MVP solid and demonstrable

Scope:
- cleanup
- tests
- better docs
- better error handling
- installation instructions for Linux and macOS

Deliverables:
- strong demo-ready MVP

## Phase 6 - Advanced memory and planning

Goal:
- make the agent more capable on longer workflows

Scope:
- better retrieval
- memory scoring
- memory namespaces
- task planning improvements
- replay and auditability

## Phase 7 - More tools and automation

Goal:
- move closer to a full local assistant

Scope:
- browser automation
- system actions
- note creation
- mail drafts
- richer permissions
- skill packages

## Phase 8 - Multi-machine local orchestration

Goal:
- use stronger machines when available

Scope:
- wake-on-LAN
- task dispatch to a gaming PC
- result retrieval
- remote local worker pattern

## Phase 9 - Voice and richer UX

Goal:
- make the agent more natural to use

Scope:
- local STT
- local TTS
- voice session handling
- richer UI
- execution timeline visualization

## Long-term direction

Long-term, unclaw should become:
- a strong local-first personal agent runtime
- easy to install
- easy to inspect
- easy to extend
- powerful on both modest and strong local hardware
