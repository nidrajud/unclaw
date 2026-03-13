# Engineering Guidelines

## Purpose

This document defines the engineering rules for the unclaw codebase.

The project must remain:
- clean
- understandable
- modular
- practical
- human-looking

The code should look like it was written by a careful engineer, not generated carelessly.

## General coding rules

### Keep it simple
Prefer simple and explicit code over clever abstractions.

### Write readable code
Names should be clear. Functions should be focused. Modules should have a clear purpose.

### Avoid premature complexity
Do not add advanced abstractions before they are needed by the project.

### No hidden magic
Important behavior should be easy to follow through the codebase.

## Project constraints

### Local-first only
Do not introduce cloud dependencies into the core runtime.

### Consumer hardware aware
Code should be designed with modest local hardware in mind:
- avoid unnecessary overhead
- avoid prompt bloat
- avoid loading too many features at once

### Transparency matters
The runtime should expose decisions, tool usage, and important execution steps in logs.

## File and module structure

### One clear responsibility per module
A module should have one clear purpose.

### Thin entrypoints
CLI, Telegram, and future channel adapters should remain thin.

### Core logic belongs in the runtime
Routing, orchestration, context building, and execution logic belong in the core modules.

## Comments and style

### Comments should be simple
Use short and useful comments when they help explain intent or a non-obvious step.

### No noisy comments
Do not explain trivial code line by line.

### No emojis in code or comments
Emojis are allowed only in user-facing interfaces where they improve UX.

### Human tone
Code and comments should feel written by a human engineer.

## Prompting and AI-generated code

### AI is a coding assistant, not the architect
Generated code must follow the project structure and constraints already defined in the docs.

### Do not let the assistant redesign the project casually
When generating a file, the assistant should respect the existing architecture unless explicitly asked to change it.

### Prefer minimal complete implementations
For early phases, prefer small but correct implementations over ambitious incomplete ones.

## Testing and validation

### Test what matters first
Prioritize:
- routing behavior
- command handling
- tool execution
- session persistence
- memory selection

### Keep smoke tests easy to run
Basic checks should be simple to execute on both Linux and macOS.

## Dependencies

### Keep dependencies limited
Every dependency should have a clear reason to exist.

### Prefer stable and well-known libraries
Avoid niche dependencies unless they clearly solve a real problem.

### Avoid heavyweight dependencies too early
Do not add large frameworks before they are justified.

## Logging

### Logging is a product feature
Logs are not only for developers. They help users understand the agent.

### Capture meaningful events
Log:
- routing decisions
- selected model
- selected memory
- tool invocations
- timing
- errors

### Avoid noisy logs with no value
Logs should be useful, not overwhelming.

## Documentation

### Docs should be practical
Documentation should help both humans and coding assistants work effectively.

### Keep docs aligned with reality
When architecture changes, update the relevant docs.

## Git and commits

### Small clean commits
Prefer many small commits with clear intent.

### Good commit messages
Write short professional commit messages in English.

Examples:
- Add initial tool registry
- Implement CLI session manager
- Add basic router decision schema

## Future-proofing

### Build extension points carefully
Add clear interfaces for:
- model providers
- tools
- channels
- memory backends

### Do not overbuild too early
Future-proofing should not turn into overengineering.
