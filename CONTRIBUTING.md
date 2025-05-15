# Contributing to RL-Engineering System

First off, thank you for considering contributing to the RL-Engineering System! Your help is greatly appreciated. 

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth and effective collaboration process.

## Table of Contents
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Code Contributions](#code-contributions)
  - [Documentation Improvements](#documentation-improvements)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
  - [Python](#python)
  - [Rust](#rust)
  - [General](#general)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

## How Can I Contribute?

### Reporting Bugs
If you encounter a bug, please open an issue on our GitHub repository. When reporting a bug, include:
- A clear and descriptive title.
- Steps to reproduce the bug.
- Expected behavior.
- Actual behavior.
- Your system environment (OS, Python version, relevant library versions).
- Any relevant logs or screenshots.

### Suggesting Enhancements
For suggestions or feature requests, please also open an issue on GitHub. Describe the enhancement, its potential benefits, and any implementation ideas you might have.

### Code Contributions
- Fork the repository and create your branch from `main` (or the relevant development branch).
- Ensure your code adheres to the [Coding Standards](#coding-standards).
- Write comprehensive [Tests](#testing) for your changes.
- Update documentation as necessary.
- Make sure your commits are atomic and have clear, descriptive messages.
- Submit a Pull Request (PR) to the `main` branch.

### Documentation Improvements
Improvements to documentation are always welcome. This includes correcting typos, clarifying explanations, adding examples, or expanding existing guides and tutorials.

## Development Setup
1. Follow the [Installation](#installation) instructions in the main `README.md` file.
2. Ensure you have `pre-commit` hooks installed and configured if the project uses them (`pre-commit install`).

## Coding Standards

### Python
- Follow PEP 8 guidelines.
- Use Black for code formatting and isort for import sorting (configurations are in `pyproject.toml`).
- Add type hints to your code and ensure it passes Mypy checks.
- Write clear and concise docstrings (e.g., Google style or NumPy style).
- Aim for readable, maintainable, and efficient code.

### Rust
- Follow standard Rust conventions (e.g., `rustfmt`, `clippy`).
- Write clear documentation comments.
- Ensure code is safe and handles errors appropriately.

### General
- Keep code modular and well-organized.
- Comment complex or non-obvious parts of the code.
- Avoid unnecessary dependencies.

## Testing
- All new features and bug fixes should include corresponding tests.
- Unit tests should cover individual components.
- Integration tests should verify interactions between components.
- Ensure all tests pass (`poetry run pytest`) before submitting a PR.
- Strive for high test coverage.

## Pull Request Process
1. Ensure your PR addresses an existing issue or discusses a new feature/bug fix.
2. Provide a clear title and detailed description of the changes in your PR.
3. Link to any relevant issues.
4. Ensure your branch is up-to-date with the target branch before submitting.
5. Respond to any feedback or review comments promptly.
6. Once approved and all checks pass, your PR will be merged.

## Code of Conduct
This project and everyone participating in it is governed by a Code of Conduct. We expect everyone to follow these guidelines to help foster an open and welcoming environment.

(We recommend adopting the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). You can copy the text from the link and create a `CODE_OF_CONDUCT.md` file, then link to it here.)

For now, please be respectful and constructive in all interactions.

Thank you for contributing! 