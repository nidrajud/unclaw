# Architecture

## Overview

Unclaw is built as a modular local-first agent runtime.

The architecture separates:
- channels
- orchestration
- model providers
- memory
- tools
- logging
- persistence

This separation keeps the project clean, testable, and extensible.

## High-level flow

A typical request follows this path:

1. user input enters from a channel
2. command handler checks for slash commands
3. pre-router applies fast deterministic rules
4. LLM router is used only if needed
5. memory candidates are selected
6. relevant tools are shortlisted
7. context is assembled with a token budget
8. the selected model is called
9. tools are executed if needed
10. the final answer is returned
11. memory is optionally updated
12. all important events are logged

## Main modules

### channels
Responsible for user interaction.

Examples:
- CLI
- Telegram
- local API
- future web UI

### core
Responsible for orchestration and execution.

Core responsibilities:
- command handling
- routing
- planning
- execution
- context building
- permissions
- session management

### llm
Responsible for model abstraction.

This layer hides provider-specific details and exposes a common interface for:
- chat
- generation
- structured output
- capabilities
- model profiles

### memory
Responsible for memory management.

Memory is split logically into:
- session memory
- user memory
- project memory
- ephemeral memory
- future execution memory

### tools
Responsible for all agent skills.

Each tool must be registered with:
- a name
- a description
- argument schema
- permission level
- execution handler
- timeout and safety rules

### logs
Responsible for observability.

This layer captures:
- route decisions
- model calls
- tool calls
- timings
- errors
- memory selection

### db
Responsible for local persistence.

The first storage backend is SQLite.

## Design principles

### Thin channels
Channels should stay thin. Business logic must live in the core runtime, not in channel adapters.

### Model-agnostic orchestration
The runtime must not depend on one model family only. It must support multiple local models and multiple providers over time.

### Minimal context by default
The system should avoid large static prompts and excessive context injection. Context must be assembled dynamically.

### Explicit tools
Tool use must not depend purely on magic model behavior. The runtime should support:
- structured JSON tool selection
- native tool calling when available
- fallback parsing for weaker models later

### Memory is selective
Memory is never blindly injected. It is retrieved, scored, and optionally included.

### Observability is a first-class feature
Logs and traces are not only for debugging. They are part of the product experience.

## Initial storage strategy

For the first versions:
- SQLite stores sessions, messages, runs, tool calls, and memory items
- local files store configs, prompts, and exportable artifacts

Later, vector search can be added behind an abstraction layer.

## Initial runtime targets

The first implementation should support:
- terminal chat
- Telegram chat
- local model profiles
- web search
- file reading
- live logs
- basic memory
- slash commands

## Future architecture extensions

Planned future additions:
- better memory retrieval
- browser automation
- local voice pipeline
- gaming PC delegation
- richer permissions system
- local web UI
- skill manifests
- compatibility adapters
