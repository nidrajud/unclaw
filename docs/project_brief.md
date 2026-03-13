# Unclaw Project Brief

## 1. Project identity

**Name:** unclaw
**Tagline:** AI agents unleashed. Local-first. No cloud lock-in. No recurring fees. Built for real local models.
**Emoji identity:** 🦐

Unclaw is a local-first agent runtime designed for real use on consumer hardware. It is inspired by the ambition of projects like OpenClaw, but it is not intended to be a clone. Its goal is to become a lighter, faster, cleaner, more transparent, and more practical reference for local AI agents.

The core idea is simple:
- work well with local open models,
- stay fast on modest hardware,
- scale up on stronger machines,
- expose the agent's reasoning and execution clearly,
- remain simple to install, understand, modify, and extend.

Unclaw must feel like a real personal local assistant, not like a cloud agent that has been forced to run locally.

---

## 2. Project goals

### Primary goal
Build a local-first agent system that can:
- answer simple questions quickly,
- search the web,
- read local files,
- use tools,
- manage multi-turn conversations,
- route intelligently between fast and deep modes,
- support local models with very different capabilities,
- remain transparent through full live logs,
- run on both Mac and PC.

### Long-term goal
Turn unclaw into a serious open-source reference for local agents, with:
- adaptive context management,
- modular memory,
- structured tool execution,
- multi-device orchestration,
- optional voice,
- clear UX,
- very low friction install and setup.

### What unclaw is not
Unclaw is not meant to be:
- a cloud-first product,
- a wrapper around one single model,
- a giant monolithic framework,
- an enterprise RAG platform,
- a copy of OpenClaw,
- a benchmark project that ignores UX.

---

## 3. Product philosophy

### Local-first
The project must be designed around local execution from day one.
Cloud services are not the target. External APIs may be useful for inspiration, but the project itself must not depend on paid remote inference.

### Model-agnostic
Unclaw must support multiple local model families and backends. It must not be locked to one model, one vendor, or one inference engine.

### Fast by default
The default user experience must feel responsive. A large part of the project value comes from perceived speed.

### Deep when needed
Unclaw must also be able to switch into a more deliberate mode for harder problems.

### Transparent
The user must be able to see what the agent is doing:
- routing choice,
- selected model,
- memory selected,
- tools chosen,
- major execution steps,
- errors and fallbacks.

### Clean engineering
The repository must look like serious human engineering work:
- readable architecture,
- simple comments,
- explicit modules,
- low magic,
- understandable flow,
- maintainable structure.

---

## 4. Core capabilities target

### MVP capabilities
The first strong MVP must include:
- terminal chat interface,
- new conversation command,
- model switching command,
- thinking mode on/off,
- web search,
- local file reading,
- Telegram chat,
- live execution logs,
- session persistence,
- clean routing between fast answer, deep answer, and tool usage.

### Later capabilities
Later versions should add:
- richer memory,
- browser automation,
- note and mail actions,
- more local system tools,
- desktop UI or web UI,
- local voice input and output,
- cross-device orchestration,
- waking and dispatching heavy tasks to another local machine,
- skill/plugin system,
- compatibility layers with selected external ecosystems.

---

## 5. Main design principles

### 5.1 Runtime over chatbot
Unclaw should be built as a runtime, not as a chat wrapper.
The core job of the runtime is to:
1. receive input,
2. inspect context,
3. choose route,
4. select the right model,
5. select the right memory,
6. select the right tools,
7. execute,
8. trace everything,
9. return a clean answer,
10. write useful memory only when justified.

### 5.2 Dual-mode behavior
Unclaw must support at least two interaction modes:
- **Fast mode**: short, fast, lightweight, minimum overhead.
- **Deep mode**: more deliberate, more context, more planning, more tool usage.

This must be visible to the user and easy to control.

### 5.3 Hybrid routing
Routing must not depend only on one tiny model.
The intended architecture is:
- deterministic pre-routing rules,
- small LLM router when useful,
- fallback to stronger model if ambiguity remains.

### 5.4 Selective context
The project must avoid the classic local-agent failure mode:
- giant prompts,
- too much conversation injected,
- too many tools described at once,
- too much memory injected automatically.

The system should inject less context, but better context.

### 5.5 Model-aware tool use
Some local models handle structured tools well.
Some do not.
Some do not support tool calling at all.

Unclaw must support multiple execution modes:
- native tool calling when reliable,
- strict JSON action plans,
- parser fallback for weaker models.

### 5.6 Memory as a modular system
Memory must not be a single undifferentiated dump.
Memory should be logically separated into layers such as:
- conversation memory,
- user memory,
- project memory,
- ephemeral memory,
- execution memory.

### 5.7 Observability is a feature
The logs are not a developer afterthought.
They are part of the product identity.

---

## 6. Target users

### Primary user
A technical power user who wants a serious local agent on personal hardware.

### Secondary users
- developers who want a clean local agent base,
- makers and tinkerers,
- people who want privacy and no recurring AI bills,
- users with stronger PCs who want to scale up locally.

### Hardware targets
The system should be designed to run across a range of machines, for example:
- compact Apple Silicon machines,
- gaming PCs,
- general consumer desktops and laptops.

The initial real-world working setup for this project is:
- development on a Linux desktop,
- testing and local-usage validation on a Mac mini,
- optional later heavy-task execution on a stronger gaming PC.

---

## 7. Technical scope

### Recommended language
Python is the primary language for the project.
Reasons:
- fast development speed,
- good ecosystem for local AI,
- good tooling for APIs, files, web, automation, and voice,
- good readability,
- good portability.

### Recommended supporting formats
- YAML for configuration,
- JSON for schemas and contracts,
- Markdown for prompts and documentation,
- SQLite for the first persistent data layer.

### Backend direction
The backend should expose a clean local runtime and API. The terminal interface is the first-class interface for the MVP. A browser-based UI is optional later and must not be required for the agent to run.

### Model backends
The architecture should keep room for several local providers. The code must be written around an abstraction layer, not hard-coded around one backend.

---

## 8. Proposed architecture

### Main areas
- `core/` for orchestration,
- `llm/` for model providers and profiles,
- `tools/` for skill execution,
- `memory/` for memory management,
- `channels/` for CLI, Telegram, and future interfaces,
- `logs/` for tracing and live observability,
- `db/` for persistence,
- `compat/` for later adapters.

### Core runtime flow
A typical runtime flow should look like this:
1. input arrives,
2. slash command check,
3. fast pre-router,
4. small router model if needed,
5. memory candidate selection,
6. context building,
7. model selection,
8. optional tool use,
9. answer build,
10. memory write,
11. log final trace.

---

## 9. Memory vision

### Guiding principle
Memory should help the model think better, not simply increase prompt size.

### Logical memory layers
The architecture should support at least:
- **session memory** for current conversation flow,
- **user memory** for stable facts and preferences,
- **project memory** for ongoing structured work,
- **ephemeral memory** for short-lived recent context,
- **execution memory** for what tools and flows worked before.

### Important memory rules
- no blind full-memory injection,
- retrieve selectively,
- score by relevance and freshness,
- allow namespace-based memory,
- allow decay or expiration for low-value temporary items,
- keep a path open for later contradiction handling.

### Practical note
Memory may be logically fragmented without becoming physically chaotic. The implementation may begin with SQLite plus lightweight structured tables and tags, then evolve later if needed.

---

## 10. Tool system vision

### Short-term tools
The first tools should be practical and safe:
- web search,
- fetch URL content,
- read file,
- list files,
- summarize session.

### Later tools
Possible later extensions:
- browser automation,
- note creation,
- mail workflows,
- local app launch,
- system actions,
- heavy-task dispatch to another machine,
- code/project tools,
- plugin skills.

### Tool execution rules
Every tool must have:
- a clear name,
- a contract,
- argument validation,
- defined permission level,
- timeout,
- traceability.

---

## 11. Logging and trace vision

### Why this matters
One of unclaw's strongest differentiators should be the ability to observe the runtime in real time.

### Logs should expose
- incoming message,
- session identifier,
- command detection,
- routing decision,
- thinking mode,
- selected model,
- candidate memory,
- injected memory,
- selected tools,
- tool outputs or summary,
- timings,
- errors,
- final answer.

### Views
There should eventually be at least two views:
- a compact readable user trace,
- a deeper engineering trace.

---

## 12. UX guidelines

### Interface goals
The UX should be:
- simple,
- fast,
- readable,
- not overloaded,
- transparent,
- pleasant in terminal.

### Commands
The CLI must support commands such as:
- `/new`
- `/sessions`
- `/use`
- `/model`
- `/think on`
- `/think off`
- `/logs on`
- `/logs off`
- `/tools`
- `/memory`
- `/session`
- `/help`

### Style rules
- no emojis in code or code comments,
- emojis are allowed only in user-facing interfaces,
- comments should be short and human-looking,
- avoid over-engineered abstractions,
- keep naming explicit and professional.

---

## 13. Engineering rules for contributors and AI coding assistants

This section is especially important for AI-assisted coding.

### Code style expectations
- Write readable Python.
- Prefer explicit control flow.
- Use small focused modules.
- Avoid clever but opaque patterns.
- Do not add unnecessary abstraction layers.
- Keep functions reasonably short.
- Add simple human-style comments only where useful.
- Avoid decorative comments.
- Avoid emoji in code, comments, filenames, and commit messages.

### Architecture expectations
- Respect the repository structure.
- Do not mix provider logic, tool logic, and orchestration logic.
- Keep storage behind clear functions or repositories.
- Keep models behind provider abstractions.
- Keep tools behind contracts and a registry.

### Logging expectations
- Important runtime steps must emit events.
- Debug information must not be hidden in random print statements.
- Prefer structured logging and a central event/tracing flow.

### Testing expectations
- Add unit tests for core routing and command behavior.
- Add lightweight integration tests for the main flow.
- Keep tests understandable.

### Documentation expectations
- Update docs when architecture changes.
- Keep README aligned with real project state.
- Avoid documenting features that do not exist yet unless clearly marked as planned.

### AI assistant behavior expectations
When an AI coding assistant works on this project, it should:
- read this file first,
- follow the repository structure,
- keep implementations practical,
- optimize for clarity and maintainability,
- not rewrite unrelated modules,
- not introduce unnecessary frameworks,
- not assume cloud services,
- not assume browser UI is required,
- not assume one model backend will solve everything,
- preserve local-first design.

---

## 14. Installation and packaging philosophy

A major project goal is low-friction setup.

### Installation goals
- minimal manual steps,
- reproducible environment,
- clean dependency management,
- easy local run commands,
- clear bootstrap scripts.

### Packaging direction
The project should aim to provide later:
- simple local install,
- simple upgrade path,
- few mandatory dependencies,
- clean separation between optional extras.

---

## 15. Security and safety direction

### MVP safety
The MVP should stay conservative.
Unsafe system actions should not be rushed.

### Safe defaults
- start with safe tools,
- validate paths and arguments,
- avoid free shell execution in the first versions,
- keep risky operations behind explicit permission boundaries,
- log important actions.

### Later direction
Later versions can introduce:
- stronger permissions,
- sandboxing,
- controlled automation,
- action confirmation checkpoints.

---

## 16. Non-goals for the first milestone

The first strong milestone should explicitly avoid trying to do all of this at once:
- advanced browser automation,
- full voice agent,
- WhatsApp and Signal from day one,
- full plugin marketplace,
- perfect long-term memory,
- full OpenClaw compatibility,
- every possible local backend,
- free-form shell power everywhere.

These can come later.

---

## 17. First milestone target

The first milestone should produce a working local agent with:
- terminal chat,
- sessions,
- slash commands,
- model profiles,
- fast/deep mode,
- basic routing,
- file reading,
- web search,
- Telegram channel,
- live execution logs,
- basic memory summary,
- a clean, professional repository.

This milestone must already be:
- usable,
- testable,
- demo-friendly,
- understandable by an external contributor.

---

## 18. Long-term ambition

The long-term ambition is not just to make another local assistant.
The ambition is to make unclaw one of the cleanest and most convincing local-first agent runtimes for real users and real machines.

That means:
- strong engineering,
- low friction,
- practical speed,
- transparent behavior,
- adaptable model support,
- memory that is actually useful,
- tools that work even with imperfect local models,
- a project architecture that can survive growth.

---

## 19. Working principle for development

Development should follow this rule:

1. build the smallest clean version that works,
2. make it observable,
3. make it reliable,
4. then extend it.

Avoid building wide before building solid.

---

## 20. Short project summary

Unclaw is a local-first AI agent runtime for consumer hardware.
It aims to be fast, transparent, modular, and usable with real local models.
It is built to support both simple fast interactions and deeper tool-based workflows.
Its value comes from adaptive routing, selective context, structured tools, modular memory, and strong execution transparency.
