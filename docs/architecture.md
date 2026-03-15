# Architecture

## Purpose

This document defines the **target architecture** for Unclaw and clarifies the gap between the **current MVP** and the **intended product**.

Unclaw must become a **lightweight, secure, privacy-first, local-only AI agent runtime**.
It must **not** drift into being a chatbot with manual tools.

This document is both:
- a design reference for future engineering work,
- and a reality check to prevent documentation from promising features that do not yet exist.

---

## Current reality vs target state

### Current MVP reality
At the time of writing, the MVP is best described as:
- a local-first conversational runtime,
- with useful manual tools,
- strong observability,
- local persistence,
- and clean module boundaries,
- **but not yet a true autonomous agent runtime**.

The current system still depends too heavily on:
- explicit slash commands,
- hardcoded tool flows,
- single-turn model execution,
- and manual user-driven tool invocation.

### Target state
The target system is:
- a **real autonomous local-first agent runtime**,
- able to decide when to use tools,
- able to execute multi-step observation-action loops,
- able to adapt routing and depth to the request,
- able to stay fast on modest hardware,
- and able to remain transparent, secure, and easy to maintain.

---

## Non-negotiable architecture principles

### 1. Agent runtime, not chatbot wrapper
Unclaw must be designed as a runtime that can:
1. receive a user request,
2. understand the intent,
3. decide whether direct answering is enough,
4. decide whether one or more tools are needed,
5. execute tool calls safely,
6. observe the results,
7. continue or stop,
8. produce a natural final answer,
9. persist only useful state,
10. trace the whole execution clearly.

If a capability only works through explicit slash commands, that capability is not yet fully integrated into the agent runtime.

### 2. Local-first and local-only for core intelligence
The core runtime must work with:
- local models,
- local memory,
- local tools,
- local persistence,
- local logs,
- local control.

Cloud APIs must not become a required dependency for core behavior.

### 3. Privacy-first
The architecture must assume that:
- user data stays local by default,
- logs may contain sensitive information,
- session storage must be minimal and understandable,
- secrets must be handled conservatively,
- web content and tool outputs must not silently leak private context.

### 4. Security-first
Every capability must be designed with explicit trust boundaries.
This includes:
- path safety,
- SSRF protection,
- prompt injection resistance,
- tool permission levels,
- safe defaults,
- bounded loops and timeouts,
- constrained persistence,
- explicit high-risk action gates.

### 5. Lightweight by design
Unclaw must remain lightweight:
- few dependencies,
- no heavyweight orchestration framework,
- no giant agent SDK unless absolutely required,
- no architecture that assumes large cloud-grade models,
- no prompt stuffing as a substitute for runtime design.

### 6. Scalable without brittle deterministic lists
The architecture must avoid depending on long hand-maintained lists of words, phrases, or cases whenever a more adaptive mechanism is reasonable.

Examples of preferred direction:
- capability routing based on model/tool reasoning or compact semantic classification,
- provider abstractions instead of vendor-specific logic in core runtime,
- schemas and contracts instead of regex-heavy glue everywhere,
- bounded adaptive logic instead of giant lists of language-specific triggers.

### 7. Mainstream UX, hidden machinery
The user should not have to think in terms of internal tools.
The default experience should feel like a real assistant.
Internal steps should remain visible in logs and traces, but **not leak awkwardly into the final answer**.

---

## Architecture layers

## channels/
Channels are thin I/O layers.

Examples:
- terminal CLI,
- Telegram,
- future local API,
- future local UI.

Channels should:
- receive user input,
- display output,
- surface logs when requested,
- delegate actual intelligence to the runtime.

Channels should **not** contain business logic, routing logic, or tool orchestration logic.

## core/
The core runtime is the brain of the system.

It is responsible for:
- request classification,
- capability routing,
- model selection,
- tool orchestration,
- observation-action loop control,
- context assembly,
- grounding policies,
- session flow,
- memory selection,
- final answer composition.

This is where the agent behavior must live.

## llm/
This layer abstracts model providers and model capabilities.

It must expose clean support for:
- chat,
- structured outputs,
- tool calling when supported,
- reasoning/thinking modes,
- model capability detection,
- profile selection,
- fallback behavior for weaker models.

Provider-specific details must not leak into high-level runtime logic.

## tools/
Tools are capability endpoints, not UX features.

Each tool must define:
- a name,
- a clear description,
- input schema,
- permission level,
- timeout or execution bounds,
- traceability,
- safe failure behavior.

Tools should be as dumb and reliable as possible.
They should return useful data, not try to impersonate the final assistant voice.

The runtime, not the tool module, should own final answer composition.

## memory/
Memory must stay modular and selective.

Logical layers may include:
- session memory,
- user memory,
- project memory,
- ephemeral memory,
- execution memory,
- research/session caches.

Memory should be:
- retrieved selectively,
- freshness-aware,
- bounded,
- and never injected blindly.

## db/
Persistence should remain simple and local-first.

Initial persistence should stay compatible with:
- SQLite,
- structured tables,
- explicit repositories,
- easy inspection,
- easy export,
- easy cleanup.

## logs/
Observability is a first-class product feature.

Logs and traces must expose:
- route decisions,
- selected model,
- tool decisions,
- tool execution steps,
- timings,
- failures,
- memory injections,
- and final outputs.

The engineering trace may be richer than the user trace, but both should remain coherent.

---

## Target runtime flow

The long-term default runtime flow should look like this:

1. input arrives from a channel,
2. command handling happens only for explicit runtime commands,
3. the runtime classifies the task,
4. the runtime decides between direct answer vs agent execution,
5. the runtime selects the model/profile,
6. the runtime builds a bounded context,
7. the model may call tools,
8. the runtime executes tools safely,
9. the runtime feeds results back into the loop,
10. the runtime stops when a final answer is ready or a safe bound is reached,
11. the final answer is rendered naturally,
12. useful traces and state are persisted.

This loop must stay:
- bounded,
- observable,
- model-aware,
- and lightweight.

---

## Tool execution strategy

Unclaw must support multiple tool execution modes because local models differ a lot.

### Preferred order
1. native tool calling when reliable,
2. structured output / action schema when native tools are weak or unavailable,
3. conservative fallback behavior for weaker models,
4. direct answer when tools are unnecessary.

The runtime must not assume that one approach works for every model family.

---

## Search and research architecture direction

Web search must be treated as a retrieval capability inside the agent loop, not as a standalone user-facing formatting engine.

The preferred long-term structure is:
- discovery/search,
- page fetch,
- extraction/cleaning,
- safety filtering,
- evidence selection,
- temporary caching,
- synthesis into natural assistant output.

Important rules:
- tools return data,
- the agent loop decides whether more research is needed,
- final answers should sound like assistant answers,
- raw tool mechanics should remain mostly hidden from the user.

---

## Current architectural gaps to keep visible

This document must stay honest.

At the time of writing, the main gaps between MVP and target architecture are:
- no real autonomous tool loop yet,
- routing still too limited,
- tool selection not yet fully model-driven,
- search stack too monolithic,
- prompt-injection hardening still incomplete,
- memory still basic,
- UX still exposes too much manual control for important capabilities.

These gaps must remain visible in docs until they are genuinely resolved.

---

## What Unclaw must never become

Unclaw must not become:
- a cloud-dependent wrapper,
- a giant framework-first project,
- a prompt-stuffed fake agent,
- a brittle keyword router,
- an opaque system that hides unsafe behavior,
- a pile of deterministic lists pretending to be intelligence,
- or a manual-tool chatbot marketed as agentic.

---

## Success criteria for the architecture

A successful architecture for Unclaw should make it possible to:
- run well on modest local hardware,
- support multiple local model profiles,
- use tools autonomously when useful,
- stay safe and bounded,
- keep the codebase understandable,
- evolve without massive rewrites,
- and deliver a polished mainstream assistant experience.
