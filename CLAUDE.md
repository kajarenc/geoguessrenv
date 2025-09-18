# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains custom Gymnasium environments and wrappers, including a GeoGuessr-style environment that's currently in development. The project aims to implement a street-level panorama navigation environment for reinforcement learning research.

## Current Components

- **Environment**: `GeoGuessrWorldEnv` - A Gymnasium-compatible environment (in development) for panorama navigation
- **GridWorldEnv** - A simple grid-based environment example
- **OpenAI Agent** - An agent that uses OpenAI's vision models for navigation
- **Wrappers** - Educational implementations of common RL wrapper patterns
- **Demo Scripts** - Example usage of the environments
- **Tests** - Test suite covering environment functionality

## Development Commands

### Installation
```bash
# Clone the repo
git clone https://github.com/kajarenc/geoguessrenv.git
cd geoguessrenv

# Create & activate virtual environment
uv venv
source .venv/bin/activate

# Install project and dependencies
uv pip install -e .
```

### Running Demos
```bash
# Basic GeoGuessr environment demo
uv run python geoguessr_env_demo.py

# OpenAI agent (requires OPENAI_API_KEY)
uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 10 --input_lat 47.620908 --input_lon -122.353508 --render
```

### Testing
```bash
uv run pytest tests/ -v
```

### Code Quality
- Set up pre-commit hooks: `pre-commit install`
- Pre-commit hooks will run automatically on commits

## Current Architecture

The codebase follows standard Gymnasium conventions:

- `gymnasium_env/` - Main package directory
- `gymnasium_env/envs/` - Environment implementations
  - `geoguessr_world.py` - GeoGuessr-style environment (in development)
  - `grid_world.py` - Simple grid world example
- `gymnasium_env/wrappers/` - Wrapper implementations
- `agents/` - Agent implementations including OpenAI-powered agent
- `tests/` - Test suite
- `scripts/` - Utility scripts

## Dependencies

- Python >= 3.11 (as specified in pyproject.toml)
- gymnasium >= 1.2.0 (core RL framework)
- numpy >= 1.24.0 (numerical computing)
- requests >= 2.32.5 (HTTP client for API calls)
- pygame >= 2.1.3 (for rendering)
- pillow >= 11.3.0 (image processing)
- openai >= 1.41.0 (for OpenAI agent)
- streetview >= 0.0.16 (street view API access)
- streetlevel >= 0.12.4 (street level imagery)
- pytest >= 8.4.2 (testing framework)
- pre-commit (code quality)

## Current Environment State

The `GeoGuessrWorldEnv` is currently implemented with basic functionality:

### Action Format (Current Implementation)
Actions use a dictionary format with operation-specific keys:
- **Click**: `{"op": "click", "click": [x, y]}`
- **Answer**: `{"op": "answer", "answer": [lat_deg, lon_deg]}`

### Demo Usage
The current demo shows:
- Environment registration with ID `"GeoGuessrWorld-v0"`
- Basic click navigation and answer submission
- Local caching in `cache/` directory
- Integration with street view data providers

## User Preferences & Code Style

### Python Development Standards
- **Supported Python versions**: 3.11+ (as specified in pyproject.toml)
- **Linter & Formatter**: Ruff 0.x (enforces PEP 8 compliance)
- **Testing Framework**: pytest 8.x
- **Dependency Management**: Use `uv` exclusively for dependency installation, synchronization, and locking. Never use `pip`, `pip-tools`, or `poetry` directly.

### Key Principles
- **Code Style**: Follow PEP 8 guidelines strictly with Ruff enforcement
- **Pythonic Code**: Write elegant, readable code following the Zen of Python
- **Architecture**: Prefer composition over inheritance
- **Naming**: Use snake_case for all Python folders and filenames
- **Self-Documenting Code**: Name functions and variables clearly to minimize need for comments
- **Comments**: When needed, capitalize comments with proper grammar and punctuation
- **Testing**: Write comprehensive unit tests using pytest
- **Modern Python**: Prioritize features available in Python 3.9+

## Testing and Usage

This is an educational/development repository for custom Gymnasium environments, with a focus on GeoGuessr-style street view navigation.

### Current Testing
The project includes:
- Basic environment functionality tests in `tests/test_geoguessr_env.py`
- OpenAI agent tool tests in `tests/test_openai_agent_tools.py`
- Test configuration in `tests/conftest.py`

### Usage Examples
1. **Basic Demo**: Run `geoguessr_env_demo.py` for a simple environment test
2. **OpenAI Agent**: Use the agent runner script with vision models for navigation
3. **Environment Integration**: Import and use environments via standard Gymnasium API

### Development Status
This is an active development project working toward implementing:
- Full street view panorama navigation
- Deterministic episode replay
- Complete caching system
- Comprehensive test coverage

Reference `TaskDescription.md` for the complete specification and end goals.
