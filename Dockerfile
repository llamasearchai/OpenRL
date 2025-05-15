# Stage 1: Builder
FROM python:3.9-slim as builder

# Set environment variables to prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for Rust, Python builds, and potentially some ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    # Add any other system dependencies your Python packages or Rust build might need
    # For example, for some torch features or other libraries: libgl1-mesa-glx libglib2.0-0
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install Poetry
RUN pip install poetry==1.4.2 # Consider using a version consistent with your team

# Copy dependency definition files
COPY pyproject.toml poetry.lock* ./
# Copy Cargo.toml for Rust dependencies (and Cargo.lock if not gitignored and used)
COPY Cargo.toml ./
# If Rust source code is in a specific directory (e.g., src/), copy it early if Cargo.toml needs it for dep resolution
# Or copy it later just before building.

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install Python dependencies (excluding optional GPU extras for this generic Dockerfile)
# For GPU, you might have a separate Dockerfile or build arg
RUN poetry install --no-root --no-interaction --no-dev # Or include --only main if you don't need other groups for runtime

# Copy the rest of the application code
# Python package source is now in src/rlhf and copied via `COPY src ./src`
# COPY rlhf ./rlhf # This line will be removed
# Rust source code (assuming it's in a top-level src directory)
COPY src ./src

# Build Rust extensions
# Ensure maturin is available (it's listed in pyproject.toml, poetry install should get it)
RUN poetry run maturin build --release --out dist  # Build wheels into dist/
# Note: `maturin develop` installs into site-packages. `maturin build` creates wheels.
# We will install these wheels in the final stage.

# Stage 2: Final image
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Docker image user" appuser

WORKDIR /app

# Copy installed Python packages from builder image's site-packages
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Copy built Rust extension wheels from builder and install them
COPY --from=builder /app/dist /tmp/dist/
RUN pip install --no-cache-dir /tmp/dist/*.whl && rm -rf /tmp/dist/

# Copy application code (Python package)
COPY --from=builder --chown=appuser:appuser /app/src/rlhf ./rlhf

# Change ownership of the app directory and switch to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port if this is a web service (e.g., FastAPI)
# Check your application if it runs a web server and on which port
# EXAMPLE: EXPOSE 8000

# Default command - could be the CLI or a web server
# Entrypoint could also be used for more complex startup logic
# This assumes your CLI is the primary way to run the application
# ENTRYPOINT ["poetry", "run", "rlhf"]
# CMD ["--help"]
# Or, if it's a FastAPI app (check rlhf.cli or other modules for server startup)
# CMD ["uvicorn", "rlhf.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 
# For now, a placeholder or a basic help command
CMD ["python", "-m", "rlhf.cli", "--help"] # Assuming rlhf.cli can be run as a module

# For GPU usage, this Dockerfile would need to start from an NVIDIA CUDA base image
# (e.g., nvidia/cuda:X.Y-cudnnA-runtime-ubuntuZ.W or a PyTorch NGC image)
# and ensure GPU drivers are correctly handled in the deployment environment.
