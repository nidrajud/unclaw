# Engineering Guidelines

## Purpose

This document defines the engineering rules for the Unclaw codebase.

These rules exist to protect the project from drifting away from its actual goal:
**building a lightweight, secure, privacy-first, local-only autonomous AI agent runtime**.

This means the codebase must not drift toward:
- a cloud wrapper,
- a framework-heavy experiment,
- a brittle rules engine,
- or a chatbot with manual tools that only looks agentic on paper.

---

## Core engineering priorities

The project must remain:
- lightweight,
- understandable,
- modular,
- secure,
- privacy-first,
- local-first,
- scalable in the right way,
- and usable by real people on consumer hardware.

When trade-offs appear, prefer:
1. correctness,
2. safety,
3. architectural honesty,
4. maintainability,
5. speed and simplicity,
6. feature breadth later.

---

## Non-negotiable project constraints

### Local-only core intelligence
Do not introduce required cloud inference into the core runtime.
Unclaw must work with local models as its real default, not as a secondary mode.

### Lightweight by default
Keep dependencies limited and justified.
Do not add heavyweight frameworks casually.
Do not solve architectural weaknesses with giant orchestration libraries unless absolutely necessary.

### Privacy-first
Assume user data is sensitive.
Minimize what is stored, minimize what is logged, and keep storage explicit.

### Secure-first
Every new tool or runtime path must be reviewed through a security lens:
- what input is trusted,
- what is untrusted,
- what may be dangerous,
- what must be bounded,
- what must be logged,
- what must be blocked by default.

### Agent runtime target
Always code with the target in mind:
Unclaw is supposed to become a **true autonomous agent runtime**.
Do not normalize manual-tool-only behavior as the final architecture.

---

## Architecture rules

### Keep channels thin
CLI, Telegram, and future channels must stay thin.
They handle transport and UX, not intelligence.

### Core owns orchestration
Routing, context assembly, tool orchestration, observation-action loops, and answer shaping belong in `core/`.

### Keep provider logic isolated
Provider-specific logic must stay in `llm/`.
Do not leak backend quirks into the rest of the runtime unless wrapped behind a clear abstraction.

### Keep tools focused
Tools should do one thing well.
They should return reliable data, not simulate assistant-style final prose.

### Keep storage explicit
Persistence must go through clear storage/repository code.
No hidden state.
No magical side channels.

### Keep observability central
Important runtime decisions must be traced through the logging/tracing system, not scattered prints.

---

## Scalability rules

### Avoid brittle deterministic lists when a more adaptive mechanism is reasonable
Do not build large, hand-maintained lists of phrases as the main source of intelligence when the problem is really one of routing, intent classification, or model/tool coordination.

Examples of what to avoid:
- giant keyword trigger lists for web search,
- language-specific hardcoded phrase lists as the main routing method,
- long rule tables that become impossible to maintain.

This does **not** mean “no deterministic logic ever”.
It means deterministic logic should be:
- minimal,
- justified,
- bounded,
- and not the only intelligence layer for a capability that must scale across languages and future tools.

### Build extension points early when they are obviously needed
If a capability is certain to grow, create a clean seam early.
Examples:
- provider abstractions,
- tool contracts,
- routing/capability interfaces,
- search profiles,
- cache namespaces,
- permission levels.

### Do not future-proof by overengineering
Create clean seams, not giant abstractions.
“Future-proof” does not mean building a framework before the need is real.

---

## Coding rules

### Prefer explicit code
Use readable, explicit Python.
Avoid cleverness that hides control flow.

### Small focused modules
Each module should have a clear responsibility.
If a file becomes a multi-subsystem monolith, split it.

### Functions should do one job
Prefer short focused functions with clear inputs/outputs.
If a function does routing, planning, formatting, and validation at once, it is probably doing too much.

### Make trust boundaries obvious
Whenever code crosses a trust boundary, the code should make that visible.
Examples:
- user input,
- fetched web content,
- filesystem access,
- secrets,
- tool outputs,
- persistence.

### Keep type expectations clear
Use clear function signatures and data contracts.
Avoid `Any` when a stronger contract is practical.

### Kill dead config and dead code
Do not keep architectural promises in config or docs that have no implementation behind them unless they are clearly labeled as planned.

---

## AI-generated code rules

### AI is an implementation assistant, not the architect
Any AI-generated code must obey project constraints.
If the generated code conflicts with the architecture, the architecture wins.

### Do not let the assistant drift the project
Reject changes that:
- add unnecessary complexity,
- silently introduce cloud assumptions,
- lock the code to one provider,
- replace runtime design with prompt tricks,
- or degrade the project into a manual-tool chatbot.

### Ask for bounded implementations
Prefer prompts that request:
- exact scope,
- file targets,
- tests,
- constraints,
- explicit non-goals,
- and architectural compatibility.

### Always audit generated changes against the target product
Before accepting a generated change, ask:
- does this make Unclaw more autonomous or more brittle?
- does this improve security/privacy/local-first behavior?
- does this improve maintainability?
- does this hide a structural problem instead of solving it?

---

## Tooling and dependency rules

### Keep dependencies limited
Every dependency must have a clear reason.
No dependency should be added just because it is fashionable.

### Prefer standard library when reasonable
If the standard library can solve the problem cleanly, prefer it.

### Prefer proven lightweight libraries
If a dependency is justified, prefer stable and widely used libraries.

### Avoid heavyweight agent frameworks too early
Unclaw should not depend on a giant framework unless the project has already proven that the architecture truly needs it.
The default bias should be: build the loop, not the framework.

---

## Security rules

### Safe defaults first
New capabilities must default to the safest reasonable behavior.

### Validate arguments aggressively
Every tool input must be validated.
Paths, URLs, modes, sizes, identifiers, and bounds must be checked.

### Treat fetched content as untrusted
Web content and file content are data, not trusted instructions.
The runtime must protect itself against prompt injection and unsafe context contamination.

### Bound loops and resources
Any agent loop, tool retry logic, search expansion, or recursion must be bounded.

### Make risky actions explicit
High-risk actions must not happen silently.
Permission boundaries must be visible in code and in runtime behavior.

---

## Privacy rules

### Store as little as practical
Persist only what is needed for functionality, continuity, or debugging.

### Keep sensitive values out of logs
Reasoning, secrets, tokens, and private content should be redacted or minimized by default.

### Make storage inspectable
The user or developer should be able to understand where data is stored and why.

### Keep retention intentional
If a capability stores search results, session data, or cache data, its retention policy should be explicit.

---

## Testing rules

### Test what matters architecturally
Prioritize tests for:
- routing,
- tool invocation,
- observation-action loop behavior,
- session/state persistence,
- safety checks,
- grounding,
- and failure modes.

### Test real behavior, not only mocked happy paths
Mocking is useful, but avoid false confidence.
If a capability depends on real provider behavior or real parsing behavior, add realistic tests where practical.

### Keep tests understandable
Tests should help another engineer understand how the system is supposed to behave.

### Add regression tests for architectural failures
When a bug reveals that the system drifted away from the target agent behavior, encode that as a regression test.

---

## Logging and observability rules

### Logging is part of the product
Logs are not only for developers.
They are part of how Unclaw stays transparent and trustworthy.

### Trace major decisions
Log:
- route decisions,
- selected model/profile,
- tool decisions,
- search depth or research mode,
- timings,
- failures,
- important safety blocks,
- and answer completion.

### Avoid noisy logs without meaning
Do not log everything blindly.
Log what helps explain runtime decisions and failures.

---

## Documentation rules

### Keep docs aligned with reality
If something is target-state but not implemented, say so clearly.
Do not document future architecture as if it already exists.

### Keep the project goal explicit everywhere that matters
The docs must repeatedly reinforce that Unclaw is trying to become:
- local-first,
- privacy-first,
- secure-first,
- lightweight,
- autonomous,
- scalable without heavy frameworks,
- and mainstream-friendly.

### Use docs to prevent drift
Docs are not decoration.
They are part of project control.
They must keep human contributors and AI coding assistants aligned.

---

## Practical rule for contributors

Before merging or accepting any non-trivial change, ask:

1. does this move Unclaw toward a real autonomous local agent runtime?
2. does it preserve lightweight local-first architecture?
3. does it improve or weaken security/privacy?
4. does it create brittle deterministic behavior?
5. does it keep the codebase clean and extensible?
6. does it remain understandable to another engineer?

If the answer is unclear, the change is not ready.
