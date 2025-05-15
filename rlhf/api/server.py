import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Literal
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Body, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator, root_validator
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
from uuid import uuid4, UUID

# Import from the correct package structure
from rlhf.algorithms.base import RLHFAlgorithmType
from rlhf.algorithms.dpo import DPOConfig
from rlhf.algorithms.ppo import PPOConfig
from rlhf.algorithms.kto import KTOConfig
from rlhf.core.logging import get_logger
from rlhf.models.policy import PolicyModelWrapper
from rlhf.models.reference import ReferenceModelWrapper
from rlhf.models.reward import RewardModelWrapper
from rlhf.models.critic import ValueModelWrapper
from rlhf.training.trainer import RLHFTrainer
from rlhf.data.dataset import create_dataset_from_jsonl
from rlhf.tracking.experiment import create_experiment_tracker
from rlhf.core.metrics import metrics_registry

# Configure logger
logger = get_logger("API")

# Define API Key security scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Initialize API app with more comprehensive OpenAPI documentation
app = FastAPI(
    title="RLHF Engineering System API",
    description="""
    API for the RLHF Engineering System that enables training and deploying LLM models using 
    Reinforcement Learning from Human Feedback (RLHF) techniques.
    
    This API allows you to:
    - Create and manage training experiments
    - Select models and datasets
    - Configure and run RLHF training jobs
    - Monitor training progress and metrics
    - Manage trained models
    """,
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    contact={
        "name": "Nik Jois",
        "email": "nikjois@llamasearch.ai",
        "url": "https://github.com/llamasearchai/OpenRL",
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/llamasearchai/OpenRL/blob/main/LICENSE",
    },
)

# Add CORS middleware with more secure settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

# Initialize metrics with configurable port
metrics_port = int(os.environ.get("METRICS_PORT", "8001"))
metrics_registry.initialize(expose_http=True, port=metrics_port)

# Define environment variables for configuration
DEBUG_MODE = os.environ.get("DEBUG", "false").lower() == "true"
EXPERIMENTS_DIR = os.environ.get("EXPERIMENTS_DIR", os.path.join(os.getcwd(), "experiments"))

# Define Pydantic models
class StatusEnum(str, Literal["created", "running", "completed", "failed", "stopped"]):
    """Status enum for experiment status tracking"""
    pass


class AlgorithmEnum(str, Literal["ppo", "dpo", "kto", "sft"]):
    """Available RLHF training algorithms"""
    pass


class ModelTypeEnum(str, Literal["policy", "reward", "reference", "value"]):
    """Types of models used in RLHF"""
    pass


class DatasetTypeEnum(str, Literal["preference", "instruction", "completion", "mixed"]):
    """Types of datasets used in RLHF training"""
    pass


class ExperimentInfo(BaseModel):
    """Basic information about a training experiment"""
    id: str = Field(..., description="Unique identifier for the experiment")
    name: str = Field(..., description="Human-readable name for the experiment")
    algorithm: AlgorithmEnum = Field(..., description="RLHF algorithm used for training")
    status: StatusEnum = Field(..., description="Current status of the experiment")
    start_time: str = Field(..., description="ISO timestamp when the experiment started")
    end_time: Optional[str] = Field(None, description="ISO timestamp when the experiment ended, if completed")
    metrics: Optional[Dict[str, float]] = Field(None, description="Summary metrics for the experiment")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "name": "DPO Training Run 1",
                "algorithm": "dpo",
                "status": "running",
                "start_time": "2023-09-21T13:45:30.123Z",
                "end_time": None,
                "metrics": {"loss": 0.42, "accuracy": 0.85}
            }
        }


class ExperimentDetails(BaseModel):
    """Detailed information about a training experiment"""
    id: str = Field(..., description="Unique identifier for the experiment")
    name: str = Field(..., description="Human-readable name for the experiment")
    description: str = Field(..., description="Detailed description of the experiment's purpose")
    algorithm: AlgorithmEnum = Field(..., description="RLHF algorithm used for training")
    status: StatusEnum = Field(..., description="Current status of the experiment")
    start_time: str = Field(..., description="ISO timestamp when the experiment started")
    end_time: Optional[str] = Field(None, description="ISO timestamp when the experiment ended, if completed")
    config: Dict[str, Any] = Field(..., description="Complete configuration for the experiment")
    metrics: Optional[Dict[str, float]] = Field(None, description="Summary metrics for the experiment")
    dataset: str = Field(..., description="Name of the dataset used for training")
    model: str = Field(..., description="Name of the primary model being trained")
    duration_seconds: Optional[int] = Field(None, description="Total duration of training in seconds, if completed")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "name": "DPO Training Run 1",
                "description": "Training Llama-2-7B with DPO using human preference pairs",
                "algorithm": "dpo",
                "status": "completed",
                "start_time": "2023-09-21T13:45:30.123Z",
                "end_time": "2023-09-21T15:45:30.123Z",
                "config": {
                    "algorithm": "dpo",
                    "algorithm_config": {"beta": 0.1},
                    "policy_model_id": "llama-7b",
                    "reference_model_id": "llama-7b",
                    "batch_size": 8,
                    "learning_rate": 1e-5,
                    "max_steps": 1000
                },
                "metrics": {"final_loss": 0.32, "accuracy": 0.89},
                "dataset": "Human Preference Pairs",
                "model": "Llama-2-7B",
                "duration_seconds": 7200
            }
        }


class TrainingState(BaseModel):
    """Current state of a training experiment"""
    experiment_id: str = Field(..., description="Unique identifier for the experiment")
    status: StatusEnum = Field(..., description="Current status of the experiment")
    current_step: int = Field(..., description="Current training step")
    total_steps: int = Field(..., description="Total number of training steps")
    current_loss: float = Field(..., description="Current training loss value")
    learning_rate: float = Field(..., description="Current learning rate")
    remaining_time_estimate: str = Field(..., description="Estimated time remaining for training")
    
    @validator('current_step')
    def current_step_valid(cls, v, values):
        if 'total_steps' in values and v > values['total_steps']:
            raise ValueError(f"Current step ({v}) cannot exceed total steps ({values['total_steps']})")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "experiment_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "running",
                "current_step": 523,
                "total_steps": 1000,
                "current_loss": 0.4231,
                "learning_rate": 1e-5,
                "remaining_time_estimate": "01:23:45"
            }
        }


class TrainingMetric(BaseModel):
    """Training metric datapoint"""
    step: int = Field(..., description="Training step for this metric")
    timestamp: str = Field(..., description="ISO timestamp when this metric was recorded")
    value: float = Field(..., description="Value of the metric")
    
    class Config:
        schema_extra = {
            "example": {
                "step": 500,
                "timestamp": "2023-09-21T14:45:30.123Z",
                "value": 0.4231
            }
        }


class ModelInfo(BaseModel):
    """Information about an available model"""
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name for the model")
    type_name: ModelTypeEnum = Field(..., description="Type of model (policy, reward, reference, value)")
    parameters: int = Field(..., description="Number of parameters in the model")
    description: str = Field(..., description="Description of the model")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "llama-7b",
                "name": "Llama-2-7B",
                "type_name": "policy",
                "parameters": 7000000000,
                "description": "Meta's Llama 2 7B parameter model"
            }
        }


class DatasetInfo(BaseModel):
    """Information about an available dataset"""
    id: str = Field(..., description="Unique identifier for the dataset")
    name: str = Field(..., description="Human-readable name for the dataset")
    type_name: DatasetTypeEnum = Field(..., description="Type of dataset")
    size: int = Field(..., description="Number of examples in the dataset")
    description: str = Field(..., description="Description of the dataset")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "preference-pairs",
                "name": "Human Preference Pairs",
                "type_name": "preference",
                "size": 50000,
                "description": "Pairs of chosen and rejected completions"
            }
        }


class ExperimentConfig(BaseModel):
    """Configuration for creating a new experiment"""
    name: str = Field(..., description="Human-readable name for the experiment")
    description: str = Field(..., description="Detailed description of the experiment's purpose")
    algorithm: AlgorithmEnum = Field(..., description="RLHF algorithm to use for training")
    algorithm_config: Dict[str, Any] = Field(..., description="Algorithm-specific configuration")
    policy_model_id: str = Field(..., description="ID of the policy model to train")
    reference_model_id: str = Field(..., description="ID of the reference model to use")
    reward_model_id: Optional[str] = Field(None, description="ID of the reward model to use (required for PPO)")
    dataset_id: str = Field(..., description="ID of the dataset to use for training")
    batch_size: int = Field(4, description="Batch size for training")
    learning_rate: float = Field(1e-5, description="Learning rate for training")
    max_steps: int = Field(1000, description="Maximum number of training steps")
    seed: int = Field(42, description="Random seed for reproducibility")
    use_lora: bool = Field(False, description="Whether to use LoRA for parameter-efficient fine-tuning")
    lora_rank: Optional[int] = Field(8, description="Rank for LoRA adapters if use_lora is True")
    
    @root_validator
    def check_algorithm_requirements(cls, values):
        """Validate that required models are provided based on algorithm"""
        algorithm = values.get('algorithm')
        if algorithm == 'ppo' and not values.get('reward_model_id'):
            raise ValueError("reward_model_id is required for PPO algorithm")
        
        # Validate algorithm-specific config
        algo_config = values.get('algorithm_config', {})
        if algorithm == 'dpo' and 'beta' not in algo_config:
            raise ValueError("DPO algorithm requires 'beta' in algorithm_config")
        elif algorithm == 'ppo' and ('vf_coef' not in algo_config or 'cliprange' not in algo_config):
            raise ValueError("PPO algorithm requires 'vf_coef' and 'cliprange' in algorithm_config")
        
        return values
    
    class Config:
        schema_extra = {
            "example": {
                "name": "DPO Training Run",
                "description": "Training Llama-2-7B with DPO using human preference pairs",
                "algorithm": "dpo",
                "algorithm_config": {"beta": 0.1},
                "policy_model_id": "llama-7b",
                "reference_model_id": "llama-7b",
                "dataset_id": "preference-pairs",
                "batch_size": 8,
                "learning_rate": 1e-5,
                "max_steps": 1000,
                "seed": 42,
                "use_lora": True,
                "lora_rank": 8
            }
        }


# Additional model and dataset management

class ModelRegistration(BaseModel):
    """Request model for registering a new model"""
    name: str = Field(..., description="Human-readable name for the model")
    type_name: ModelTypeEnum = Field(..., description="Type of model")
    parameters: int = Field(..., description="Number of parameters in the model")
    description: str = Field(..., description="Description of the model")
    path: str = Field(..., description="Path to the model files or HuggingFace model ID")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional model configuration")


class DatasetRegistration(BaseModel):
    """Request model for registering a new dataset"""
    name: str = Field(..., description="Human-readable name for the dataset")
    type_name: DatasetTypeEnum = Field(..., description="Type of dataset")
    size: int = Field(..., description="Number of examples in the dataset")
    description: str = Field(..., description="Description of the dataset")
    path: str = Field(..., description="Path to the dataset files or HuggingFace dataset ID")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional dataset configuration")


# In-memory storage (would be replaced with a database in production)
experiments: Dict[str, Dict[str, Any]] = {}
models: Dict[str, Dict[str, Any]] = {}
datasets: Dict[str, Dict[str, Any]] = {}
trainers: Dict[str, Any] = {}
training_metrics: Dict[str, List[TrainingMetric]] = {}

# Ensure experiments directory exists
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


# Authentication and utility functions
def verify_api_key(api_key: str = Depends(API_KEY_HEADER)) -> bool:
    """
    Verify the API key provided in the request.
    
    In a production environment, this would validate against a secure database.
    For now, we're using a simple environment variable check.
    """
    allowed_keys = os.environ.get("ALLOWED_API_KEYS", "dev_key,test_key").split(",")
    
    # Skip validation in debug mode if no key provided
    if DEBUG_MODE and api_key is None:
        logger.warning("API key validation skipped in debug mode")
        return True
        
    if api_key is None or api_key not in allowed_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return True


# Helper functions
def get_timestamp() -> str:
    """Get current time in ISO format with UTC timezone"""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def calculate_duration(start_time: str, end_time: str) -> Optional[int]:
    """Calculate duration between two ISO timestamps in seconds"""
    try:
        start = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        end = datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        return int((end - start).total_seconds())
    except (ValueError, TypeError):
        logger.warning(f"Could not calculate duration between {start_time} and {end_time}")
        return None


def get_models_from_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get available models from registry
    
    This would typically load from a database or model registry service
    """
    if not models:
        # Add some sample models
        models["llama-7b"] = {
            "id": "llama-7b",
            "name": "Llama-2-7B",
            "type_name": "policy",
            "parameters": 7000000000,
            "description": "Meta's Llama 2 7B parameter model"
        }
        models["llama-13b"] = {
            "id": "llama-13b",
            "name": "Llama-2-13B",
            "type_name": "policy",
            "parameters": 13000000000,
            "description": "Meta's Llama 2 13B parameter model"
        }
        models["reward-deberta"] = {
            "id": "reward-deberta",
            "name": "DeBERTa Reward Model",
            "type_name": "reward",
            "parameters": 900000000,
            "description": "DeBERTa-based reward model fine-tuned on preference data"
        }
    return models


def get_datasets_from_registry() -> Dict[str, Dict[str, Any]]:
    """
    Get available datasets from registry
    
    This would typically load from a database or dataset registry service
    """
    if not datasets:
        # Add some sample datasets
        datasets["preference-pairs"] = {
            "id": "preference-pairs",
            "name": "Human Preference Pairs",
            "type_name": "preference",
            "size": 50000,
            "description": "Pairs of chosen and rejected completions"
        }
        datasets["instruction-data"] = {
            "id": "instruction-data",
            "name": "Instruction Dataset",
            "type_name": "instruction",
            "size": 100000,
            "description": "Instructions for PPO training"
        }
    return datasets


def validate_experiment_id(experiment_id: str) -> None:
    """Validate that the experiment exists, raising appropriate exception if not"""
    if experiment_id not in experiments:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Experiment not found: {experiment_id}"
        )


def save_experiment(experiment_data: Dict[str, Any]) -> None:
    """
    Save experiment to storage
    
    Args:
        experiment_data: Dictionary containing experiment configuration and metadata
    """
    try:
        experiment_id = experiment_data["id"]
        experiments[experiment_id] = experiment_data
        
        # Also save to disk for persistence
        experiment_path = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        with open(experiment_path, "w") as f:
            json.dump(experiment_data, f, indent=2)
            
        logger.info(f"Saved experiment {experiment_id} to {experiment_path}")
    except (KeyError, IOError) as e:
        logger.error(f"Failed to save experiment: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save experiment: {str(e)}"
        )


def load_experiments() -> None:
    """
    Load experiments from disk into memory
    
    This ensures experiments persist across server restarts
    """
    if not os.path.exists(EXPERIMENTS_DIR):
        logger.warning(f"Experiments directory not found: {EXPERIMENTS_DIR}")
        return
    
    loaded_count = 0
    for filename in os.listdir(EXPERIMENTS_DIR):
        if filename.endswith(".json"):
            try:
                experiment_path = os.path.join(EXPERIMENTS_DIR, filename)
                with open(experiment_path, "r") as f:
                    experiment_data = json.load(f)
                    if "id" in experiment_data:
                        experiments[experiment_data["id"]] = experiment_data
                        loaded_count += 1
                    else:
                        logger.warning(f"Skipping experiment file {filename}: missing id field")
            except Exception as e:
                logger.error(f"Error loading experiment {filename}: {str(e)}")
    
    logger.info(f"Loaded {loaded_count} experiments from {EXPERIMENTS_DIR}")


def create_training_job(experiment_id: str) -> bool:
    """
    Create and start an actual training job for the experiment
    
    In a production environment, this would:
    1. Initialize the appropriate training components
    2. Set up logging and monitoring
    3. Start the training in a separate process or job queue
    
    Returns:
        bool: True if job started successfully, False otherwise
    """
    try:
        if experiment_id not in experiments:
            logger.error(f"Cannot create training job: experiment {experiment_id} not found")
            return False
            
        exp_data = experiments[experiment_id]
        config = exp_data.get("config", {})
        
        # This would be a real implementation in production
        # Here we're just logging what would happen
        algorithm = config.get("algorithm")
        logger.info(f"Creating {algorithm} training job for experiment {experiment_id}")
        
        # We would actually create the training components
        # policy_model = load_model(config.get("policy_model_id"))
        # dataset = load_dataset(config.get("dataset_id"))
        # trainer = create_trainer(algorithm, policy_model, dataset, config)
        # trainers[experiment_id] = trainer
        # trainer.start()
        
        trainers[experiment_id] = {
            "running": True,
            "start_time": get_timestamp(),
            "status": "running",
            "current_step": 0
        }
        
        return True
    except Exception as e:
        logger.error(f"Failed to create training job for {experiment_id}: {str(e)}")
        return False


# Load experiments at startup
load_experiments()


# API Routes
@app.get("/api/experiments", response_model=List[ExperimentInfo], tags=["Experiments"])
async def get_experiments(_: bool = Depends(verify_api_key)):
    """
    Get list of all experiments
    
    Returns a list of all training experiments in the system, with basic information
    about each experiment.
    """
    experiment_list = []
    for exp_id, exp_data in experiments.items():
        experiment_list.append(ExperimentInfo(
            id=exp_id,
            name=exp_data.get("name", "Unnamed"),
            algorithm=exp_data.get("algorithm", "unknown"),
            status=exp_data.get("status", "unknown"),
            start_time=exp_data.get("start_time", ""),
            end_time=exp_data.get("end_time"),
            metrics=exp_data.get("metrics")
        ))
    return experiment_list


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentDetails, tags=["Experiments"])
async def get_experiment_details(
    experiment_id: str = PathParam(..., description="Unique identifier of the experiment"),
    _: bool = Depends(verify_api_key)
):
    """
    Get detailed information about a specific experiment
    
    Returns comprehensive information about the experiment, including configuration,
    current status, and metrics if available.
    """
    validate_experiment_id(experiment_id)
    
    exp_data = experiments[experiment_id]
    
    # Calculate duration if available
    duration_seconds = None
    if exp_data.get("start_time") and exp_data.get("end_time"):
        duration_seconds = calculate_duration(exp_data["start_time"], exp_data["end_time"])
    
    return ExperimentDetails(
        id=experiment_id,
        name=exp_data.get("name", "Unnamed"),
        description=exp_data.get("description", ""),
        algorithm=exp_data.get("algorithm", "unknown"),
        status=exp_data.get("status", "unknown"),
        start_time=exp_data.get("start_time", ""),
        end_time=exp_data.get("end_time"),
        config=exp_data.get("config", {}),
        metrics=exp_data.get("metrics"),
        dataset=exp_data.get("dataset", "unknown"),
        model=exp_data.get("model", "unknown"),
        duration_seconds=duration_seconds
    )


@app.post("/api/experiments/{experiment_id}/start", response_model=TrainingState, tags=["Training"])
async def start_training(
    experiment_id: str = PathParam(..., description="Unique identifier of the experiment to start"),
    _: bool = Depends(verify_api_key)
):
    """
    Start training for an experiment
    
    Initiates the training process for the specified experiment. If the experiment
    is already running, an error will be returned.
    """
    validate_experiment_id(experiment_id)
    
    exp_data = experiments[experiment_id]
    
    # Check if already running
    if exp_data.get("status") == "running":
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Experiment is already running"
        )
    
    # Update status
    exp_data["status"] = "running"
    exp_data["start_time"] = get_timestamp()
    save_experiment(exp_data)
    
    # Start the actual training job
    job_started = create_training_job(experiment_id)
    if not job_started:
        # Revert status if job failed to start
        exp_data["status"] = "failed"
        save_experiment(exp_data)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start training job"
        )
    
    # Return current training state
    return TrainingState(
        experiment_id=experiment_id,
        status="running",
        current_step=0,
        total_steps=exp_data.get("config", {}).get("max_steps", 1000),
        current_loss=0.0,
        learning_rate=exp_data.get("config", {}).get("learning_rate", 1e-5),
        remaining_time_estimate="Calculating..."
    )


@app.post("/api/experiments/{experiment_id}/stop", response_model=TrainingState, tags=["Training"])
async def stop_training(
    experiment_id: str = PathParam(..., description="Unique identifier of the experiment to stop"),
    _: bool = Depends(verify_api_key)
):
    """
    Stop training for an experiment
    
    Halts the training process for the specified experiment. If the experiment
    is not currently running, an error will be returned.
    """
    validate_experiment_id(experiment_id)
    
    exp_data = experiments[experiment_id]
    
    # Check if running
    if exp_data.get("status") != "running":
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Experiment is not running"
        )
    
    # Update status
    exp_data["status"] = "stopped"
    exp_data["end_time"] = get_timestamp()
    save_experiment(exp_data)
    
    # In a real implementation, we would stop the actual training job
    if experiment_id in trainers:
        # trainers[experiment_id].stop() in a real implementation
        trainers[experiment_id]["running"] = False
        trainers[experiment_id]["status"] = "stopped"
    
    # Get current progress if available
    current_step = 0
    current_loss = 0.0
    if experiment_id in trainers:
        current_step = trainers[experiment_id].get("current_step", 0)
        current_loss = trainers[experiment_id].get("current_loss", 0.0)
    
    # Return final training state
    return TrainingState(
        experiment_id=experiment_id,
        status="stopped",
        current_step=current_step,
        total_steps=exp_data.get("config", {}).get("max_steps", 1000),
        current_loss=current_loss,
        learning_rate=exp_data.get("config", {}).get("learning_rate", 1e-5),
        remaining_time_estimate="N/A"
    )


@app.get(
    "/api/experiments/{experiment_id}/metrics/{metric_name}",
    response_model=List[TrainingMetric],
    tags=["Metrics"]
)
async def get_experiment_metrics(
    experiment_id: str = PathParam(..., description="Unique identifier of the experiment"),
    metric_name: str = PathParam(..., description="Name of the metric to retrieve"),
    _: bool = Depends(verify_api_key)
):
    """
    Get metrics for an experiment
    
    Retrieves the time series data for a specific metric of the experiment.
    """
    validate_experiment_id(experiment_id)
    
    # Check if metrics exist in our storage
    metrics_key = f"{experiment_id}_{metric_name}"
    if metrics_key not in training_metrics:
        # Generate mock metrics for demonstration
        metrics = []
        exp_data = experiments[experiment_id]
        total_steps = exp_data.get("config", {}).get("max_steps", 1000)
        status = exp_data.get("status", "created")
        
        # Determine how many steps to generate based on status
        if status == "completed":
            current_step = total_steps
        elif status == "running" and experiment_id in trainers:
            current_step = trainers[experiment_id].get("current_step", total_steps // 2)
        elif status == "stopped" and experiment_id in trainers:
            current_step = trainers[experiment_id].get("current_step", total_steps // 3)
        else:
            current_step = 0
        
        # Generate metrics only if we have steps
        if current_step > 0:
            # Create data points with appropriate intervals
            sample_rate = max(1, current_step // 100)  # Aim for ~100 data points
            for step in range(0, current_step + 1, sample_rate):
                # For loss, start high and decrease (with some noise)
                if metric_name == "loss":
                    base_value = 0.5 * (1 - step/total_steps)
                    noise = 0.05 * (torch.rand(1).item() - 0.5)
                    value = max(0.01, base_value + noise)
                # For accuracy, start low and increase (with some noise)
                elif metric_name == "accuracy":
                    base_value = 0.8 * (step/total_steps)
                    noise = 0.05 * (torch.rand(1).item() - 0.5)
                    value = min(1.0, max(0.0, base_value + noise))
                # For other metrics, generate random noise around a trend line
                else:
                    value = 0.5 + 0.1 * (step/total_steps) + 0.05 * (torch.rand(1).item() - 0.5)
                
                # Calculate timestamp based on step (assuming constant step time)
                # Start from current time and work backwards
                seconds_per_step = 10
                timestamp = get_timestamp() if step == current_step else (
                    datetime.utcnow() - (current_step - step) * seconds_per_step
                ).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                
                metrics.append(TrainingMetric(
                    step=step,
                    timestamp=timestamp,
                    value=float(value)  # Ensure it's a Python float, not tensor
                ))
            
            # Store metrics for future requests
            training_metrics[metrics_key] = metrics
    
    return training_metrics.get(metrics_key, [])


@app.get("/api/models", response_model=List[ModelInfo], tags=["Models"])
async def get_models(_: bool = Depends(verify_api_key)):
    """
    Get available models
    
    Returns a list of all models available for training in the system.
    """
    models_data = get_models_from_registry()
    return [
        ModelInfo(
            id=model_id,
            name=model_data["name"],
            type_name=model_data["type_name"],
            parameters=model_data["parameters"],
            description=model_data["description"]
        ) for model_id, model_data in models_data.items()
    ]


@app.get("/api/datasets", response_model=List[DatasetInfo], tags=["Datasets"])
async def get_datasets(_: bool = Depends(verify_api_key)):
    """
    Get available datasets
    
    Returns a list of all datasets available for training in the system.
    """
    datasets_data = get_datasets_from_registry()
    return [
        DatasetInfo(
            id=dataset_id,
            name=dataset_data["name"],
            type_name=dataset_data["type_name"],
            size=dataset_data["size"],
            description=dataset_data["description"]
        ) for dataset_id, dataset_data in datasets_data.items()
    ]


@app.post("/api/experiments", response_model=ExperimentInfo, tags=["Experiments"])
async def create_experiment(
    config: ExperimentConfig, 
    _: bool = Depends(verify_api_key)
):
    """
    Create a new experiment
    
    Creates a new training experiment with the provided configuration.
    """
    # Generate a unique ID
    experiment_id = str(uuid4())
    
    # Get current timestamp
    timestamp = get_timestamp()
    
    # Lookup associated model and dataset information
    models_data = get_models_from_registry()
    datasets_data = get_datasets_from_registry()
    
    # Validate that referenced models and datasets exist
    if config.policy_model_id not in models_data:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Policy model not found: {config.policy_model_id}"
        )
    
    if config.reference_model_id not in models_data:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Reference model not found: {config.reference_model_id}"
        )
    
    if config.reward_model_id and config.reward_model_id not in models_data:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Reward model not found: {config.reward_model_id}"
        )
    
    if config.dataset_id not in datasets_data:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Dataset not found: {config.dataset_id}"
        )
    
    policy_model = models_data.get(config.policy_model_id, {}).get("name", "Unknown")
    dataset = datasets_data.get(config.dataset_id, {}).get("name", "Unknown")
    
    # Create experiment data
    experiment_data = {
        "id": experiment_id,
        "name": config.name,
        "description": config.description,
        "algorithm": config.algorithm,
        "status": "created",
        "start_time": timestamp,
        "config": {
            "algorithm": config.algorithm,
            "algorithm_config": config.algorithm_config,
            "policy_model_id": config.policy_model_id,
            "reference_model_id": config.reference_model_id,
            "reward_model_id": config.reward_model_id,
            "dataset_id": config.dataset_id,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_steps": config.max_steps,
            "seed": config.seed,
            "use_lora": config.use_lora,
            "lora_rank": config.lora_rank
        },
        "model": policy_model,
        "dataset": dataset,
    }
    
    # Save experiment
    save_experiment(experiment_data)
    
    # Return basic info
    return ExperimentInfo(
        id=experiment_id,
        name=config.name,
        algorithm=config.algorithm,
        status="created",
        start_time=timestamp
    )


@app.get("/api/health", tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns system health information and status.
    """
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)] if gpu_available else []
    
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": get_timestamp(),
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_devices": gpu_names,
        "experiment_count": len(experiments),
        "active_trainers": sum(1 for trainer in trainers.values() if trainer.get("running", False))
    }


@app.delete(
    "/api/experiments/{experiment_id}",
    status_code=204,
    tags=["Experiments"],
    responses={
        204: {"description": "Experiment successfully deleted"},
        404: {"description": "Experiment not found"},
        400: {"description": "Cannot delete a running experiment"}
    }
)
async def delete_experiment(
    experiment_id: str = PathParam(..., description="Unique identifier of the experiment to delete"),
    _: bool = Depends(verify_api_key)
):
    """
    Delete an experiment
    
    Permanently removes an experiment and its associated data.
    Cannot delete experiments that are currently running.
    """
    validate_experiment_id(experiment_id)
    
    exp_data = experiments[experiment_id]
    
    # Check if running
    if exp_data.get("status") == "running":
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running experiment. Stop it first."
        )
    
    # Remove from memory
    del experiments[experiment_id]
    
    # Remove from disk
    try:
        experiment_path = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        if os.path.exists(experiment_path):
            os.remove(experiment_path)
    except Exception as e:
        logger.error(f"Error removing experiment file: {str(e)}")
        # Continue anyway - we've already removed it from memory
    
    # Remove associated metrics
    metrics_keys = [k for k in training_metrics.keys() if k.startswith(f"{experiment_id}_")]
    for key in metrics_keys:
        if key in training_metrics:
            del training_metrics[key]
    
    # Remove trainer if exists
    if experiment_id in trainers:
        del trainers[experiment_id]
    
    # No content response for successful deletion
    return None


# Add error handling middleware
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler for unhandled exceptions
    
    Provides a consistent error response format and logs exceptions
    """
    # Log the exception
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    # Return a standardized error response
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please contact support."}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom handler for HTTPExceptions that adds structured logging
    """
    # Log at appropriate level based on status code
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    elif exc.status_code >= 400:
        logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    else:
        logger.info(f"HTTP {exc.status_code}: {exc.detail}")
    
    # Return the exception as-is
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.get("/api/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model_details(
    model_id: str = PathParam(..., description="Unique identifier of the model"),
    _: bool = Depends(verify_api_key)
):
    """
    Get detailed information about a specific model
    
    Returns comprehensive information about the model.
    """
    models_data = get_models_from_registry()
    if model_id not in models_data:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )
    
    model_data = models_data[model_id]
    return ModelInfo(
        id=model_id,
        name=model_data["name"],
        type_name=model_data["type_name"],
        parameters=model_data["parameters"],
        description=model_data["description"]
    )


@app.post("/api/models", response_model=ModelInfo, tags=["Models"])
async def register_model(
    model: ModelRegistration,
    _: bool = Depends(verify_api_key)
):
    """
    Register a new model
    
    Adds a new model to the registry, making it available for training.
    """
    # Generate a unique ID (in production, this might follow a specific format)
    model_id = f"{model.name.lower().replace(' ', '-')}-{str(uuid4())[:8]}"
    
    # Check if a model with the same name already exists
    models_data = get_models_from_registry()
    for existing_id, existing_model in models_data.items():
        if existing_model["name"] == model.name:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"A model with name '{model.name}' already exists"
            )
    
    # Register the model
    models_data[model_id] = {
        "id": model_id,
        "name": model.name,
        "type_name": model.type_name,
        "parameters": model.parameters,
        "description": model.description,
        "path": model.path,
        "config": model.config or {}
    }
    
    # In a real implementation, we would save this to persistent storage
    # For now, it's just in memory and will be lost on restart
    
    return ModelInfo(
        id=model_id,
        name=model.name,
        type_name=model.type_name,
        parameters=model.parameters,
        description=model.description
    )


@app.get("/api/datasets/{dataset_id}", response_model=DatasetInfo, tags=["Datasets"])
async def get_dataset_details(
    dataset_id: str = PathParam(..., description="Unique identifier of the dataset"),
    _: bool = Depends(verify_api_key)
):
    """
    Get detailed information about a specific dataset
    
    Returns comprehensive information about the dataset.
    """
    datasets_data = get_datasets_from_registry()
    if dataset_id not in datasets_data:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_id}"
        )
    
    dataset_data = datasets_data[dataset_id]
    return DatasetInfo(
        id=dataset_id,
        name=dataset_data["name"],
        type_name=dataset_data["type_name"],
        size=dataset_data["size"],
        description=dataset_data["description"]
    )


@app.post("/api/datasets", response_model=DatasetInfo, tags=["Datasets"])
async def register_dataset(
    dataset: DatasetRegistration,
    _: bool = Depends(verify_api_key)
):
    """
    Register a new dataset
    
    Adds a new dataset to the registry, making it available for training.
    """
    # Generate a unique ID
    dataset_id = f"{dataset.name.lower().replace(' ', '-')}-{str(uuid4())[:8]}"
    
    # Check if a dataset with the same name already exists
    datasets_data = get_datasets_from_registry()
    for existing_id, existing_dataset in datasets_data.items():
        if existing_dataset["name"] == dataset.name:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"A dataset with name '{dataset.name}' already exists"
            )
    
    # Register the dataset
    datasets_data[dataset_id] = {
        "id": dataset_id,
        "name": dataset.name,
        "type_name": dataset.type_name,
        "size": dataset.size,
        "description": dataset.description,
        "path": dataset.path,
        "config": dataset.config or {}
    }
    
    # In a real implementation, we would save this to persistent storage
    
    return DatasetInfo(
        id=dataset_id,
        name=dataset.name,
        type_name=dataset.type_name,
        size=dataset.size,
        description=dataset.description
    )


def start_server(
    host: str = "0.0.0.0", 
    port: int = 8000, 
    reload: bool = False, 
    log_level: str = "info",
    workers: int = 1
):
    """
    Start the API server
    
    Args:
        host: Host address to bind the server to
        port: Port to listen on
        reload: Whether to enable auto-reload on code changes (development only)
        log_level: Logging level for uvicorn
        workers: Number of worker processes
    """
    logger.info(f"Starting RLHF API server on {host}:{port}")
    logger.info(f"Debug mode: {DEBUG_MODE}, Auto-reload: {reload}")
    
    # Print OpenAPI documentation URLs
    docs_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/docs"
    redoc_url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/redoc"
    logger.info(f"API Documentation: {docs_url}")
    logger.info(f"ReDoc Documentation: {redoc_url}")
    
    # Start the server
    uvicorn.run(
        "rlhf.api.server:app", 
        host=host, 
        port=port, 
        reload=reload, 
        log_level=log_level,
        workers=workers
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    parser.add_argument("--log-level", type=str, default="info", choices=["debug", "info", "warning", "error", "critical"], help="Logging level")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Start the server with the parsed arguments
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers
    )