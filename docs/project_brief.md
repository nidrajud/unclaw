# Unclaw Project Brief

## 1. Project identity

**Name:** Unclaw  
**Tagline:** AI agents unleashed. Local-first. No cloud lock-in. No recurring fees. Built for real local models.  
**Emoji identity:** 🦐

Unclaw is intended to become a **lightweight, secure, privacy-first, local-only AI agent runtime** designed for real use on consumer hardware.

It is inspired by the ambition of larger agent systems, but it must not become:
- a clone,
- a giant framework,
- a cloud wrapper,
- or a chatbot with manual tools dressed up as an agent.

The project goal is to become a serious reference for local AI agents by being:
- lighter,
- clearer,
- safer,
- more transparent,
- more practical,
- and easier to extend.

Unclaw must feel like a real personal assistant that genuinely works locally.

---

## 2. Current status and target status

### Current status
The MVP already provides:
- local model execution through Ollama,
- terminal and Telegram interaction,
- session persistence,
- local file tools,
- web research capabilities,
- observability and logs,
- and a relatively clean modular codebase.

But the current MVP is still **not yet a true autonomous agent runtime**.
It is closer to:
- a strong local-first assistant MVP,
- with useful manual tools,
- but without a complete agent loop.

### Target status
The target product is a runtime that can:
- understand requests naturally,
- decide when tools are needed,
- execute tools autonomously,
- observe tool results,
- continue reasoning in bounded multi-step loops,
- answer naturally without exposing awkward internal mechanics,
- and stay lightweight enough for modest local hardware.

This gap must remain explicit until it is truly closed.

---

## 3. Primary goals

### Primary goal
Build a local-first agent system that can:
- answer simple questions quickly,
- handle harder requests in a deeper mode,
- read local files,
- search and fetch web information,
- use tools intelligently,
- manage multi-turn conversations,
- route adaptively between fast and deeper execution,
- support local models with different capabilities,
- remain transparent through live logs and traces,
- and run well on both Mac and PC.

### Long-term goal
Turn Unclaw into a serious open-source reference for local-first autonomous agents, with:
- adaptive routing,
- model-aware tool use,
- modular memory,
- research workflows,
- optional voice,
- stronger local automation,
- multi-machine local orchestration,
- clean UX,
- and a very low-friction setup.

---

## 4. Non-negotiable product constraints

Unclaw must remain:
- **local-first**,
- **privacy-first**,
- **secure-first**,
- **lightweight**,
- **open source**,
- **free to run locally**,
- **mainstream-usable**,
- and **architecturally honest**.

That means:
- no required cloud inference,
- no dependency on heavy orchestration frameworks unless clearly justified,
- no product direction built around giant prompts and fake “agentic” behavior,
- no silent drift into a chatbot-with-tools architecture.

---

## 5. What Unclaw is not

Unclaw is not meant to be:
- a cloud-first product,
- a thin wrapper around one model,
- a giant monolithic framework,
- an enterprise RAG platform,
- a benchmark-only project that ignores UX,
- a brittle keyword router,
- or a manual-tool chatbot pretending to be autonomous.

---

## 6. Product philosophy

### 6.1 Local-first
The core runtime must work locally from day one.
Cloud services may inspire some ideas, but they must not define the real architecture.

### 6.2 Privacy-first
User data stays local by default.
Storage, logs, sessions, research traces, and caches must remain inspectable and bounded.

### 6.3 Security-first
Every capability must respect strong safety boundaries.
This includes:
- safe defaults,
- argument validation,
- SSRF resistance,
- path protections,
- prompt-injection awareness,
- permission levels,
- and bounded execution loops.

### 6.4 Fast by default
The default experience must feel responsive.
Most requests should not pay the cost of deep reasoning or heavy orchestration.

### 6.5 Deep when needed
Harder tasks may use deeper routing, more steps, or more tool use — but still within bounded and observable execution.

### 6.6 Transparent
The user must be able to inspect what the system is doing through logs and traces:
- route choice,
- model choice,
- tool usage,
- search or research depth,
- major execution steps,
- timings,
- failures,
- and fallbacks.

### 6.7 Clean engineering
The repository must remain understandable and maintainable.
It should look like careful human engineering work.

---

## 7. Core product requirement: runtime over chatbot

Unclaw must be built as a **runtime**, not as a chat wrapper.

The runtime’s job is to:
1. receive the input,
2. inspect the request and state,
3. decide the route,
4. select the model/profile,
5. select the relevant memory,
6. decide whether tools are needed,
7. execute tools safely,
8. observe results,
9. continue or stop,
10. produce a natural final answer,
11. persist only useful information,
12. trace the whole execution.

If a capability is only accessible through an explicit user command, it is not yet fully integrated into the runtime.

---

## 8. MVP expectations

### Strong MVP target
The first strong MVP should include:
- terminal interaction,
- session management,
- model profiles,
- fast/deep behavior,
- local persistence,
- web research,
- local file reading,
- Telegram access,
- live execution logs,
- bounded tool use,
- and a clean professional repository.

### But the strong MVP must also start the agent transition
The MVP must not stop at “chat + slash commands + tools”.
It must begin the transition toward:
- automatic tool selection,
- autonomous bounded execution,
- adaptive routing,
- natural answers that hide tool mechanics,
- and model-aware capability handling.

---

## 9. Main design principles

### 9.1 Dual-mode behavior
Unclaw must support at least two broad interaction styles:
- **Fast mode**: short, lightweight, responsive, minimum overhead.
- **Deep mode**: more deliberate, more context, more tool usage, more bounded multi-step execution.

### 9.2 Adaptive routing
Routing must not rely on a giant hand-maintained keyword list.
The target architecture should support:
- minimal deterministic pre-routing where clearly justified,
- adaptive capability routing,
- small routing logic or structured model assistance when useful,
- stronger fallback behavior when ambiguity remains.

### 9.3 Model-aware tool use
Different local models have different strengths.
Unclaw must support multiple execution modes such as:
- native tool calling when reliable,
- structured outputs when needed,
- conservative fallback behavior for weaker models.

### 9.4 Selective context
Avoid the classic local-agent failure mode:
- giant prompts,
- too much conversation injected,
- too many tools described at once,
- too much memory injected blindly.

The system must inject less context, but better context.

### 9.5 Memory as a modular system
Memory must not be one undifferentiated dump.
Target layers may include:
- session memory,
- user memory,
- project memory,
- ephemeral memory,
- execution memory,
- research cache.

### 9.6 Observability is part of the product
Logs are not a developer afterthought.
They are part of how Unclaw earns trust.

### 9.7 Scalable without deterministic brittleness
Avoid building key capabilities around giant lists of terms or hardcoded language-specific triggers when a more robust adaptive mechanism is possible.

---

## 10. Target users

### Primary user
A technical power user who wants a serious local agent on personal hardware.

### Secondary users
- developers who want a clean local agent base,
- makers and tinkerers,
- privacy-conscious users,
- users who want local AI without recurring fees,
- users with modest machines as well as stronger local hardware.

### Hardware target
The system should run well across:
- compact Apple Silicon machines,
- gaming PCs,
- consumer desktops and laptops.

The design must remain compatible with the reality of modest local hardware.

---

## 11. Technical scope

### Recommended language
Python remains the primary language.

Reasons:
- development speed,
- strong local AI ecosystem,
- good tooling for APIs, files, automation, and voice,
- portability,
- readability.

### Recommended formats
- YAML for config,
- JSON for structured outputs/contracts where useful,
- Markdown for docs and prompts,
- SQLite for the first persistent data layer.

### Backend direction
The backend must expose a clean local runtime and API.
The terminal remains a first-class interface.
A local web UI may come later, but must not become mandatory.

### Model backends
The architecture should preserve support for multiple local providers and profiles.
The runtime must not be hardcoded around one provider forever.

---

## 12. Tool system vision

### Short-term tools
The first tools should remain practical and safe:
- web search,
- fetch URL content,
- read file,
- list directory,
- session or memory inspection where useful.

### Later tools
Possible extensions:
- browser automation,
- note creation,
- mail drafting,
- local app/system actions,
- code/project tools,
- richer skills/plugins,
- multi-machine task dispatch.

### Tool rules
Every tool must have:
- a clear name,
- a contract/schema,
- argument validation,
- a permission level,
- timeout or execution bounds,
- traceability,
- safe failure behavior.

And most importantly:
**tools should return useful data, not own the assistant voice.**
The runtime should own the final natural answer.

---

## 13. Logging and trace vision

One of Unclaw’s strongest differentiators should be real-time observability.

Logs should expose:
- incoming message,
- session identifier,
- routing decision,
- selected model/profile,
- memory decisions,
- selected tools,
- tool summaries/results,
- timings,
- errors,
- final answer.

The system should eventually offer at least:
- a compact readable trace,
- and a deeper engineering trace.

---

## 14. UX direction

The UX should be:
- simple,
- fast,
- readable,
- transparent when useful,
- but not overloaded by internal mechanics.

### Important UX rule
The user should not need to think like the runtime.
The user should not have to know internal tool names or manual workflows for normal use.

Slash commands may exist for power use, debugging, or forcing behavior.
But the default experience should increasingly move toward natural autonomous assistance.

---

## 15. Engineering rules for contributors and AI coding assistants

When a human contributor or AI coding assistant works on Unclaw, it should:
- read this brief first,
- respect repository structure,
- keep implementations practical,
- protect local-first/privacy-first/security-first constraints,
- avoid unnecessary frameworks,
- not assume browser UI is mandatory,
- not assume one model backend solves everything,
- avoid drifting toward brittle keyword logic,
- and keep the agent-runtime goal visible at all times.

If a change makes Unclaw more manual, more brittle, or more misleadingly “agentic”, it is probably the wrong change.

---

## 16. Security and safety direction

### MVP safety
The MVP must stay conservative.
Unsafe system actions must not be rushed.

### Safe defaults
- start with safe tools,
- validate arguments aggressively,
- avoid unrestricted shell execution early,
- keep risky operations behind explicit permission boundaries,
- treat fetched content as untrusted,
- log important actions.

### Later direction
Later versions may add:
- stronger permissions,
- sandboxing,
- controlled automation,
- action confirmation checkpoints,
- richer safety boundaries.

---

## 17. Non-goals for the first strong milestone

Do not try to do everything at once.
Avoid prematurely expanding into:
- full browser automation,
- full voice agent,
- broad messaging platform support,
- plugin marketplace,
- perfect long-term memory,
- full compatibility layers,
- unrestricted shell power.

Build solid first.

---

## 18. Long-term ambition

The long-term ambition is not just to build another local assistant.
The ambition is to make Unclaw one of the clearest, safest, and most convincing local-first autonomous agent runtimes for real users and real machines.

That means:
- strong engineering,
- low friction,
- practical speed,
- transparent behavior,
- adaptable model support,
- useful memory,
- safe and effective tools,
- and an architecture that can survive growth.

---

## 19. Working development principle

Development should follow this order:
1. build the smallest clean version that works,
2. make it observable,
3. make it reliable,
4. make it safe,
5. then extend it.

Do not build wide before building solid.
