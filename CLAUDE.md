# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Gymnasium environments and wrappers example repository containing educational implementations of reinforcement learning environments and common wrapper patterns.

## Key Components

- **Environment**: `GridWorldEnv` - A simple grid-based environment where an agent navigates to reach a target
- **Wrappers**: Educational implementations of common RL wrapper patterns:
  - `ClipReward` - Clips rewards to a valid range
  - `DiscreteActions` - Restricts action space to finite subset
  - `RelativePosition` - Computes relative position observations
  - `ReacherRewardWrapper` - Weights reward terms for reacher environment

## Development Commands

### Installation
```bash
cd gymnasium_env
pip install -e .
```

### Code Quality
- Set up pre-commit hooks: `pre-commit install`
- Pre-commit hooks will run automatically on commits

## Architecture

The codebase follows standard Gymnasium conventions:

- `gymnasium_env/envs/` - Environment implementations
- `gymnasium_env/wrappers/` - Wrapper implementations  
- `gymnasium_env/__init__.py` - Registers the GridWorld environment with Gymnasium
- Environment registration ID: `"gymnasium_env/GridWorld-v0"`

## Dependencies

- Python >= 3.11
- gymnasium (core RL framework)
- pygame >= 2.1.3 (for rendering)
- pre-commit (code quality)

## Testing and Usage

This is primarily an educational/example repository. Test new environments and wrappers by importing and instantiating them in Python scripts using the standard Gymnasium API.