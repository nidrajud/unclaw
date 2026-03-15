# Roadmap

## Strategy

Unclaw must be developed in layers.

But the sequencing matters:
we should not keep adding surface features to a runtime that is still structurally closer to a chatbot with manual tools than to a real agent.

The first major objective now is:
**repair the MVP foundation so Unclaw becomes a serious local-first autonomous agent runtime baseline.**

This roadmap therefore distinguishes clearly between:
- what exists,
- what is being repaired,
- and what comes after the runtime becomes truly agentic.

---

## Guiding priorities

Every roadmap decision should preserve these constraints:
- lightweight,
- secure-first,
- privacy-first,
- local-first,
- multi-model aware,
- mainstream-usable,
- and scalable without heavy frameworks or brittle deterministic logic.

---

## Phase 0 — Foundation and documentation

### Goal
Create a clean repository and define the project direction.

### Status
Largely complete.

### Deliverables
- README,
- core docs,
- repository structure,
- local Python environment,
- first commits.

### Remaining expectation
Docs must stay aligned with reality and with the target agent-runtime direction.

---

## Phase 1 — Base runtime MVP

### Goal
Build the first local-first conversational runtime.

### Status
Complete as a conversational MVP, but incomplete as an agent runtime.

### What exists
- CLI chat,
- session management,
- model profiles,
- local persistence,
- basic routing,
- live logging,
- provider abstraction.

### Architectural limitation
This phase produced a strong assistant foundation, but not yet a true autonomous runtime.

---

## Phase 2 — Initial tools and utility

### Goal
Make the system useful beyond plain chat.

### Status
Partially complete.

### What exists
- web search,
- URL fetch,
- file reading,
- file listing,
- tool registry,
- tool dispatcher,
- traceable manual tool execution.

### Architectural limitation
These tools are still too manual from the user perspective.
Tool existence alone does not make the system agentic.

---

## Phase 3 — Runtime repair: chatbot MVP to agent MVP

### Goal
This is now the most urgent phase.
Transform Unclaw from a chatbot with manual tools into a serious local-first agent MVP.

### Priority
Highest.
Nothing is more important than this phase now.

### Main objectives
- connect models and tools properly,
- introduce a bounded observation-action loop,
- make tool use increasingly autonomous,
- keep the UX natural,
- keep the runtime lightweight,
- keep the system safe and transparent.

### Expected outcomes
- the model can request tool usage,
- the runtime can execute tools and continue the turn,
- search becomes a capability inside the runtime loop, not a separate fake-agent path,
- routing becomes more adaptive,
- final answers look like assistant answers, not tool dumps.

### Example sub-phases

#### 3.1 Tool-calling and runtime loop
- add model/tool integration,
- parse native or structured tool calls,
- implement bounded multi-step execution,
- persist and trace loop steps clearly.

#### 3.2 Search stack repair
- split the current search monolith,
- make search discovery/fetch/synthesis cleaner,
- harden against prompt injection,
- improve answer shaping.

#### 3.3 Adaptive capability routing
- reduce reliance on slash commands for normal behavior,
- support capability-aware routing,
- avoid giant deterministic trigger lists,
- keep routing lightweight and bounded.

#### 3.4 UX cleanup
- hide internal mechanics better,
- keep explicit commands for power users,
- make the default experience feel like a real assistant.

---

## Phase 4 — Stronger memory foundations

### Goal
Introduce memory that helps the runtime think better without bloating prompts.

### Status
Basic session memory exists, but richer memory remains future work.

### Scope
- better session summaries,
- user memory basics,
- project memory basics,
- selective retrieval,
- memory write rules,
- bounded memory injection.

### Important note
Memory must not become a dump.
It must stay selective, inspectable, and lightweight.

---

## Phase 5 — Stabilization and quality hardening

### Goal
Make the post-repair MVP solid, demonstrable, and trustworthy.

### Scope
- cleanup,
- tests,
- better docs,
- stronger safety checks,
- better error handling,
- installation polish,
- E2E validation,
- performance checks.

### Deliverable
A strong demo-ready MVP that is honestly agentic, not only marketed that way.

---

## Phase 6 — Research depth and caching

### Goal
Strengthen web and information workflows without bloating the runtime.

### Scope
- local caching of research results,
- better research/session state,
- quick vs deep research profiles,
- deeper multi-step research,
- stronger synthesis quality,
- still-bounded retrieval behavior.

### Constraint
This phase must keep the search/research stack lightweight and maintainable.

---

## Phase 7 — More tools and safe automation

### Goal
Move closer to a real local assistant while preserving safety.

### Scope
- browser automation,
- note creation,
- draft generation,
- local system actions,
- code/project tools,
- richer permission boundaries,
- possible skill packages.

### Constraint
Every new tool must fit the runtime cleanly.
Do not widen the tool surface faster than the orchestration quality can support.

---

## Phase 8 — Multi-machine local orchestration

### Goal
Use stronger local machines when available.

### Scope
- wake-on-LAN,
- heavy-task dispatch,
- local worker pattern,
- result retrieval,
- secure machine-to-machine local coordination.

### Constraint
Keep the default setup simple for single-machine users.

---

## Phase 9 — Voice and richer UX

### Goal
Make the assistant more natural for mainstream usage.

### Scope
- local STT,
- local TTS,
- voice sessions,
- richer UI,
- execution timeline visualization.

### Constraint
Voice and richer UI must be built on top of a solid runtime, not used to hide architectural weakness.

---

## Long-term direction

Long-term, Unclaw should become:
- a strong local-first autonomous personal agent runtime,
- easy to install,
- easy to inspect,
- easy to extend,
- powerful on both modest and strong local hardware,
- and trusted because its behavior stays visible and understandable.

---

## Immediate roadmap truth

At this stage, the most important truth is simple:

**do not keep piling features on top of a not-yet-agentic core.**

The next serious milestone is not “more features”.
It is:
**turn the current MVP into a real autonomous local-first agent MVP.**
