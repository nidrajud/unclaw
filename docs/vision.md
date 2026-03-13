# Vision

## What is unclaw

Unclaw is a local-first agent runtime designed for real local models and real consumer hardware.

The goal is not to build another cloud-dependent assistant, nor to clone existing agent frameworks. The goal is to build a fast, transparent, modular, and practical local agent that works well on machines such as a Mac mini, gaming PCs, and other personal computers.

Unclaw should feel like a real personal assistant:
- fast for simple tasks
- capable for complex tasks
- transparent in how it thinks, routes, and uses tools
- easy to install and use
- fully local-first, with no cloud lock-in

## Long-term ambition

Unclaw aims to become a reference project for local AI agents:
- clean architecture
- strong developer experience
- useful for real users
- optimized for small and mid-sized local models
- extensible to larger local models on more powerful hardware

The project should eventually offer capabilities comparable to larger agent systems, while staying lighter, clearer, and better adapted to local-first usage.

## Core product principles

### Local-first
The core of the project must work without cloud services. Local models, local memory, local tools, and local execution are the default.

### Adaptive intelligence
Unclaw should not send the same amount of context, memory, and tools to the model every time. It should adapt dynamically to the user request.

### Fast and deep dual-mode
The agent must support two broad behaviors:
- a fast mode for quick answers and simple tasks
- a deep mode for reasoning, planning, and multi-step execution

### Transparency
The user must be able to inspect what the agent is doing:
- routing
- model choice
- tool usage
- memory selection
- timing
- execution steps

### Clean extensibility
The project should remain readable and extensible. New tools, channels, memory layers, and model providers must be easy to add without rewriting the core runtime.

## Target users

The main target users are:
- developers
- makers
- local AI enthusiasts
- power users
- people who want control over their tools and data

## What unclaw is not

Unclaw is not:
- a cloud SaaS product
- a thin wrapper over remote APIs
- a monolithic chatbot with too much prompt stuffing
- an opaque autonomous system that hides what it is doing

## Success criteria

A successful unclaw project should:
- run well on modest local hardware
- feel responsive
- remain understandable to developers
- support multiple local model sizes
- handle both simple and complex requests
- provide clear logs and execution traces
- offer a strong base for future growth
