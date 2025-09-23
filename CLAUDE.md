# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains custom Gymnasium environments and wrappers, including a GeoGuessr-style environment that's currently in development. The project aims to implement a street-level panorama navigation environment for reinforcement learning research.

## Current Components

- **Environment**: `GeoGuessrEnv` - A Gymnasium-compatible environment for panorama navigation
- **OpenAI Agent** - Vision-based agent using OpenAI's GPT-4 Vision for intelligent navigation
- **Baseline Agents** - Simple benchmark agents for testing and comparison
- **VLM Broker** - Standardized interface for Vision-Language Model interactions
- **Agent Runners** - CLI scripts for running different agent implementations
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
uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 5 --input_lat 47.620908 --input_lon -122.353508 --render

# Baseline agent evaluation
uv run python geoguess_env/run_baseline.py --episodes 2 --out results.csv --geofence geofences/seattle_15km.json --max-nav-steps 10 --seed 123
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

- `geoguess_env/` - Main package directory (Note: actual dir is `geoguess_env` not `gymnasium_env`)
  - `geoguessr_env.py` - Main environment implementation
  - `config.py` - Configuration dataclasses with validation
  - `action_parser.py` - Action parsing and validation logic
  - `asset_manager.py` - Manages caching and loading of panorama assets
  - `geometry_utils.py` - Geographic and geometric calculations
  - `vlm_broker.py` - Vision-Language Model broker for standardized prompting
  - `baseline_agent.py` - Baseline agent implementations for benchmarking
  - `run_baseline.py` - CLI runner for baseline agent evaluation
  - `providers/` - Street view data provider implementations
    - `base.py` - Abstract base provider class
    - `google_streetview.py` - Google Street View API integration
- `agents/` - Agent implementations including OpenAI-powered agent
  - `openai_agent.py` - Vision-based navigation agent using OpenAI models
  - `openai_models.py` - Pydantic models for structured tool calling
  - `base.py` - Base agent interface
  - `utils.py` - Agent utility functions for caching and encoding
- `tests/` - Comprehensive test suite
- `scripts/` - Utility and demo scripts
  - `run_openai_agent.py` - OpenAI agent runner
  - `cache_test_data.py` - Test data caching utility

## Dependencies

- Python >= 3.10 (as specified in pyproject.toml)
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

The `GeoGuessrEnv` is currently implemented with basic functionality:

### Action Format (Current Implementation)
Actions use a dictionary format with a unified value key:
- **Click**: `{"op": "click", "value": [x, y]}`
- **Answer**: `{"op": "answer", "value": [lat_deg, lon_deg]}`

### Demo Usage
The current demo shows:
- Environment registration with ID `"GeoGuessrWorld-v0"`
- Basic click navigation and answer submission
- Local caching in `cache/` directory
- Integration with street view data providers

## Agent Implementations

The project includes several agent implementations for different use cases:

### OpenAIVisionAgent
- **Purpose**: Vision-based navigation using OpenAI's GPT-4 Vision models
- **Strategy**: Uses structured tool calling with vision analysis for intelligent navigation
- **Features**:
  - Episodic memory with navigation history
  - Structured prompting via VLMBroker
  - Caching for repeated scenarios
  - Link snapping for improved navigation accuracy
  - Force-answer logic when approaching step limits
- **Requirements**: OpenAI API key (`OPENAI_API_KEY`)
- **Usage**: Suitable for research and high-performance navigation

### BaselineAgent
- **Purpose**: Minimal arrow follower for benchmarking and testing
- **Strategy**:
  - Sweeps horizontally across the image to find navigation links
  - Follows links for up to K steps or until loop detection
  - Answers with continent centroid heuristics
- **Features**:
  - Loop detection via panorama ID tracking
  - Configurable maximum navigation steps
  - Simple geographic priors for guessing
  - Reproducible with seed control
- **Usage**: Baseline for performance comparison and environment testing

### ImprovedBaselineAgent
- **Purpose**: Enhanced baseline with learning capabilities
- **Strategy**: Extends BaselineAgent with success tracking
- **Features**:
  - Learns from successful navigation clicks
  - Enhanced answer generation (placeholder for future improvements)
  - Could incorporate text detection, architectural analysis, etc.
- **Usage**: Development platform for baseline improvements

## User Preferences & Code Style

### Python Development Standards
- **Supported Python versions**: 3.10+ (as specified in pyproject.toml)
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
- Action parser tests in `tests/test_action_parser.py`
- Asset manager tests in `tests/test_asset_manager.py`
- Configuration validation tests in `tests/test_config.py`
- Geometry utilities tests in `tests/test_geometry_utils.py`
- Provider tests in `tests/test_google_streetview_provider.py`
- Integration tests in `tests/test_integration.py`
- Environment retry logic tests in `tests/test_env_retry_logic.py`
- Blocklist functionality tests in `tests/test_blocklist_functionality.py`
- Test configuration in `tests/conftest.py`
- Reusable test fixtures in `tests/test_fixtures.py`

### Usage Examples
1. **Basic Demo**: Run `geoguessr_env_demo.py` for a simple environment test
2. **OpenAI Agent**: Use the agent runner script with vision models for navigation
3. **Baseline Agent**: Run baseline evaluation for benchmarking and testing
4. **Environment Integration**: Import and use environments via standard Gymnasium API

### Development Status
This is an active development project working toward implementing:
- Full street view panorama navigation
- Deterministic episode replay
- Complete caching system
- Comprehensive test coverage

Reference `TaskDescription.md` for the complete specification and end goals.

## Key Implementation Details

### Configuration System
The project uses a structured configuration system based on dataclasses (`config.py`):
- `GeoGuessrConfig` - Main configuration with validation
- `GeofenceConfig` - Geographic boundary constraints
- `ProviderConfig` - Data provider settings
- `RenderConfig` - Rendering parameters
- `NavigationConfig` - Navigation behavior settings

### Caching Strategy
- Street view panoramas are cached locally in `cache/` directory
- Cache structure:
  - `cache/images/` - Panorama images stored as `{panorama_id}.jpg`
  - `cache/metadata/` - Metadata stored as `{root_panorama_id}_mini.jsonl` files
  - `cache/replays/` - Episode replay data
- Each metadata file contains panorama info and navigation links for connected panoramas
- Images are stored flat without provider subdirectories (simplifies access across providers)
- Caching reduces API calls and improves test reliability

### Provider Architecture
- Abstract `PanoramaProvider` base class defines the interface
- Google Street View provider implemented with rate limiting
- Providers handle fetching, caching, and metadata management
- Easy to extend with new providers (e.g., Mapillary, Bing)

### Action Space Design
Actions are dictionary-based with operation types:
- Click actions include pixel coordinates in the "value" array: [x, y]
- Answer actions provide latitude/longitude guesses in the "value" array: [lat, lon]
- Action parser validates and normalizes all inputs with fallback to safe defaults

### VLM Broker Architecture
The `VLMBroker` provides standardized I/O between vision-language models and the environment:
- **Prompt Building**: Generates consistent prompts explaining the task, image context, and allowed actions
- **Action Parsing**: Extracts and validates actions from VLM response text
- **Fallback Handling**: Provides safe default actions when parsing fails
- **JSON Schema**: Uses the single format `{"op":"operation","value":[...]}` for all actions
- **Error Recovery**: Gracefully handles malformed responses with center-click fallbacks

## Testing Strategy

### Test Organization
- `tests/conftest.py` - Shared fixtures and test utilities
- `tests/test_fixtures.py` - Reusable test data and assertions
- `tests/fixtures/` - Static test data (panorama samples, etc.)
- Unit tests for each major component
- Integration tests for end-to-end workflows

### Running Tests
```bash
# Run all tests with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_geoguessr_env.py -v

```

## CI/CD Pipeline

### GitHub Actions Workflow
Located in `.github/workflows/python-package.yml`:
- Runs on push to main and pull requests
- Tests against Python 3.10, 3.11, and 3.12
- Environment secrets required:
  - `GOOGLE_MAPS_API_KEY` - For Street View API access
  - `OPENAI_API_KEY` - For OpenAI agent testing
- Workflow steps:
  1. Linting with Ruff
  2. Format checking with Ruff
  3. Full test suite with pytest

### Pre-commit Hooks
Configured in `.pre-commit-config.yaml`:
- Automatically runs Ruff linting and formatting
- Ensures code quality before commits
- Install with: `pre-commit install`

## Development Workflow

### Setting Up Development Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install in editable mode with all dependencies
uv pip install -e .

# Install pre-commit hooks
pre-commit install

# Set up environment variables
cp .env.example .env
# Edit .env to add your API keys
```

### Common Development Tasks
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check . --fix

# Run specific test
uv run pytest tests/test_geoguessr_env.py::test_click_action -v

# Run baseline agent
uv run python geoguess_env/run_baseline.py --episodes 2 --out test_results.csv --geofence geofences/seattle_15km.json

# Test OpenAI agent (requires API key)
uv run python scripts/run_openai_agent.py --model gpt-4o --max_nav_steps 5 --render
```

### Debugging Tips
- Enable verbose logging: Set environment variable `DEBUG=1`
- Visualize episodes: Use `--render` flag with demo scripts
- Check cache integrity: Examine `cache/` directory structure
- API debugging: Monitor rate limits and response codes

## API Keys and Environment Variables

### Required API Keys
- **GOOGLE_MAPS_API_KEY**: Required for Google Street View access
  - Obtain from: [Google Cloud Console](https://console.cloud.google.com/)
  - Enable: Street View Static API
- **OPENAI_API_KEY**: Required for OpenAI agent functionality
  - Obtain from: [OpenAI Platform](https://platform.openai.com/)

### Environment File Structure
Create `.env` file in project root:
```bash
GOOGLE_MAPS_API_KEY=your_google_maps_key_here
OPENAI_API_KEY=your_openai_key_here
```

## Known Issues and Limitations

### Current Limitations
- Only Google Street View provider fully implemented
- Limited to static panorama images (no dynamic/video support)
- Action space currently uses pixel coordinates (not normalized)
- No multi-agent support yet

### Common Issues
- **Rate Limiting**: Google API has quotas; implement backoff strategies
- **Cache Size**: Can grow large; periodic cleanup may be needed
- **Test Flakiness**: Network-dependent tests may fail; use cached fixtures
- **Memory Usage**: Large panorama images can consume significant RAM

## Future Development Roadmap

### Planned Features
- Additional providers (Mapillary, Bing Street Side)
- Normalized action space (0-1 coordinates)
- Episode recording and replay system
- Performance benchmarking suite
- Web-based visualization dashboard

### Extension Points
- New agent architectures in `agents/`
- Additional wrappers for observation/action transformations
- Provider-specific optimizations
- Enhanced caching strategies
