# InCoRo Demo

Minimal robotics stack inspired by the InCoRo paper. This repository implements a basic control-loop shape with a simulated robot, simulated scene perception, and OpenAI-based task decomposition and action generation.

## Current Scope

- task decomposition with the OpenAI SDK
- structured scene objects and robot/action models
- a feedback-loop orchestrator
- safety validation and action clipping
- a simulated robot interface
- a simulated perception module with optional camera hookup via OpenCV

## Entry Points

`main.py`

Runs the actual flow:

1. decompose a natural-language task
2. read scene state and robot state
3. ask the LLM for the next action
4. safety-check the action
5. execute it on the simulated robot

`examples.py`

Runs small local demos for:

- the simulated robot interface
- the safety filter

## Run

```bash
uv sync
cp .env.example .env
uv run main.py
uv run examples.py
```

## Structure

```text
.
├── main.py
├── examples.py
└── src/
    ├── llm.py
    ├── models.py
    ├── orchestrator.py
    └── runtime.py
```

## Modules

### `src/models.py`

Shared dataclasses for configuration, scene objects, robot state, actions, and task plans.

### `src/llm.py`

Contains:

- `TaskPreprocessor` for decomposing natural-language tasks
- `LLMController` for generating the next low-level action

### `src/runtime.py`

Contains:

- `PerceptionModule` with a simulated scene
- `SafetyFilter` for bounds and delta checks
- `RobotInterface` for simulated execution

### `src/orchestrator.py`

Coordinates the feedback loop across the LLM, perception, safety, and robot layers.
