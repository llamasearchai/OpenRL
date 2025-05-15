# RL-Engineering System

<!-- A brief one-paragraph description of the project. -->
This project is a production-ready reinforcement learning engineering system designed for fine-tuning Large Language Models (LLMs) using techniques like RLHF (Reinforcement Learning from Human Feedback).

**Note:** This README provides an overview of the project's capabilities and how to get started. We are continuously working on improving the documentation and features.

## Table of Contents
- [Motivation](#motivation)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Cloning the Repository](#cloning-the-repository)
  - [Setting up the Environment](#setting-up-the-environment)
  - [Building Rust Extensions](#building-rust-extensions)
- [Usage](#usage)
  - [Running the CLI](#running-the-cli)
  - [Training a Model](#example-training-a-model)
- [Running Tests](#running-tests)
- [Docker](#docker)
  - [Building the Docker Image](#building-the-docker-image)
  - [Running with Docker Compose](#running-with-docker-compose)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Motivation

<!-- Describe the problem this project solves or its main purpose.
     What makes it special or innovative?
     E.g., "Training large language models with RLHF is complex and resource-intensive.
     This system aims to provide a streamlined, scalable, and extensible platform for researchers and engineers..." -->
**Provide a compelling motivation for your project here. Explain the core problem it solves and its unique value proposition.**

## Features

<!-- List key features or capabilities of the system with a bit more detail. -->
- **Modular Design:** Easily extendable components for algorithms, data handling, and models.
- **Scalable Training:** Support for distributed training using DeepSpeed and Accelerate.
- **Rust-Powered Performance:** Critical operations potentially accelerated with Rust extensions for efficiency.
- **Comprehensive Experiment Tracking:** Integrated with Weights & Biases (W&B) and TensorBoard.
- **Flexible Configuration:** Utilizes Hydra for managing complex configurations.
- **End-to-End RLHF Workflow:** Covers data preprocessing, model training (SFT, Reward Modeling, PPO/DPO), and evaluation.
- **CLI & API:** Provides both a command-line interface for ease of use and a FastAPI-based API for programmatic access (if applicable).
- **Training Dashboard:** (If functional) A web-based dashboard to monitor training progress.

## Project Structure

```
OpenRL/
├── Cargo.toml          # Rust dependencies and build configuration for rlhf-rust-ops
├── Dockerfile          # For building Docker images
├── docker-compose.yml  # For running services with Docker Compose
├── examples/           # Example scripts and notebooks showcasing functionalities
├── pyproject.toml      # Python dependencies and project metadata (Poetry)
├── README.md           # This file
├── scripts/            # Utility scripts (e.g., data processing, deployment helpers)
├── src/
│   ├── rlhf/           # Main Python package for the RLHF system
│   │   ├── __init__.py
│   │   ├── version.py
│   │   ├── algorithms/   # RL algorithms (PPO, DPO, etc.)
│   │   ├── api/          # FastAPI application components
│   │   ├── core/         # Core components and utilities
│   │   ├── data/         # Data loading and preprocessing
│   │   ├── evaluation/   # Evaluation scripts and metrics
│   │   ├── langchain/    # LangChain integrations
│   │   ├── models/       # Model definitions (actors, critics, reward models)
│   │   ├── optimization/ # Optimization techniques and schedulers
│   │   ├── rust_bindings/# Python bindings for Rust extensions
│   │   ├── tracking/     # Experiment tracking integrations (W&B, Tensorboard)
│   │   ├── training/     # Training loops and orchestration
│   │   └── utils/        # Common utility functions
│   └── (other Rust crates source if any, e.g., lib.rs for rlhf-rust-ops if Cargo.toml points here)
├── tests/              # Unit, integration, and performance tests
│   ├── integration/
│   ├── performance/
│   └── unit/
└── training_dashboard/ # Source for the training monitoring dashboard UI
```

**Important Note on Structure:** The primary Python package `rlhf` is located in `src/rlhf/`. If you had a previous top-level `rlhf/` directory, its contents should be merged into `src/rlhf/` or relocated to other appropriate places like `examples/` or `scripts/`. Ensure `src/rlhf/__init__.py` and `src/rlhf/version.py` are in place.

## Prerequisites

- Python >=3.9, <3.12
- Poetry (>=1.4.0, for Python dependency management)
- Rust (latest stable version, if modifying or building Rust components from source, including `cargo`)
- Docker and Docker Compose (latest versions, if using containerized deployment)

## Installation

### Cloning the Repository
```bash
git clone https://github.com/llamasearchai/OpenRL.git
cd OpenRL
```

### Setting up the Environment
It is recommended to use a virtual environment. Poetry will handle this automatically.

1.  **Install Python dependencies:**
    ```bash
    poetry install
    ```
    This command installs all base dependencies.

2.  **For GPU support (optional):**
    If you have NVIDIA GPUs and want to use features like Triton, bitsandbytes, etc., install the `gpu` extras:
    ```bash
    poetry install --extras gpu
    ```
    Ensure you have the appropriate CUDA drivers and toolkit installed on your system.

3. **Activate the virtual environment (if not already active):**
   ```bash
   poetry shell
   ```

### Building Rust Extensions
Maturin, listed in `pyproject.toml`, handles the Rust extensions. Poetry should build them automatically during `poetry install`. If you need to build them manually or during development:
```bash
poetry run maturin develop # For an editable install in the current environment
# OR for a release build (creates wheels in target/wheels & installs):
# poetry run maturin build --release
```
The `rlhf-rust-ops` library defined in `Cargo.toml` will be built. Ensure its source code (e.g., `src/lib.rs` for the library root) is correctly located, typically within a `src/` directory relative to `Cargo.toml` if not a workspace.

## Usage

### Running the CLI
The project includes a CLI entry point defined in `pyproject.toml`. You can access it via Poetry:
```bash
poetry run rlhf --help
```
This should display available commands and options.

### Example: Training a Model

To run a training job, you might use a command like:
```bash
# Example command (replace with actual command and config path)
poetry run rlhf train --config-path path/to/your/configs --config-name training_config.yaml
```

Check the `examples/` directory for more detailed scripts. For instance, to run the simple DPO example:
```bash
poetry run python examples/simple_dpo.py # (Adjust path/command if needed)
```

## Running Tests

To run the entire test suite (unit, integration tests):
```bash
poetry run pytest
```

To get a coverage report:
```bash
poetry run pytest --cov=rlhf --cov-report=xml --cov-report=term-missing
```
Refer to `pyproject.toml` for `pytest` marker configurations (e.g., `slow`, `gpu`, `distributed`).

## Docker

### Building the Docker Image
Ensure Docker is running. From the project root directory:
```bash
docker build -t rlhf-engineering-system:latest .
```
Consider tagging it `llamasearchai/openrl:latest` or similar if you plan to publish to a container registry.

### Running with Docker Compose
Docker Compose can be used to run the application, especially if it involves multiple services or specific configurations.
```bash
# Ensure docker-compose.yml is configured correctly
docker-compose up
```
To run in detached mode:
```bash
docker-compose up -d
```

## Documentation

Detailed documentation is a work in progress and will be expanded in the `docs/` directory. Future plans include:
- **API Reference:** Generated from source code comments (e.g., using Sphinx for Python and `cargo doc` for Rust).
- **Guides:** Practical guides for common tasks and system architecture.
- **Tutorials:** Step-by-step walkthroughs of specific use cases and advanced features.

Contributions to documentation are highly welcome!

## Contributing

Contributions are welcome! We appreciate your help in improving this system.
Please read our [CONTRIBUTING.md](CONTRIBUTING.md) guidelines to understand our development process, coding standards, and how to submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
