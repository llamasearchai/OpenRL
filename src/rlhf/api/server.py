import os
from typing import Dict, List, Optional, Union, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rlhf.core.logging import get_logger

logger = get_logger("API")

# Initialize FastAPI app
app = FastAPI(
    title="RLHF Engineering System API",
    description="API for RLHF model training, evaluation, and inference",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define API models
class ModelInfo(BaseModel):
    id: str
    name: str
    type_name: str
    parameters: int
    description: str = ""

class DatasetInfo(BaseModel):
    id: str
    name: str
    type_name: str
    size: int
    description: str = ""

class ExperimentInfo(BaseModel):
    id: str
    name: str
    algorithm: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)

class ExperimentConfig(BaseModel):
    name: str
    description: str
    algorithm: str
    algorithm_config: Dict[str, Any]
    policy_model_id: str
    reference_model_id: str
    reward_model_id: Optional[str] = None
    dataset_id: str
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_steps: int = 10000
    seed: int = 42

class TrainingRequest(BaseModel):
    experiment_id: str

class GenerationRequest(BaseModel):
    model_id: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1

class GenerationResponse(BaseModel):
    generations: List[str]
    model_id: str
    prompt: str

# In-memory storage (for demo purposes)
models = {}
datasets = {}
experiments = {}
active_trainers = {}


# Routes
@app.get("/", tags=["Health"])
async def root():
    """API health check endpoint"""
    return {"status": "healthy", "service": "rlhf-engineering-system"}

@app.get("/api/models", response_model=List[ModelInfo], tags=["Models"])
async def get_models():
    """Get all available models"""
    return list(models.values())

@app.get("/api/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model(model_id: str):
    """Get model by ID"""
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return models[model_id]

@app.get("/api/datasets", response_model=List[DatasetInfo], tags=["Datasets"])
async def get_datasets():
    """Get all available datasets"""
    return list(datasets.values())

@app.get("/api/datasets/{dataset_id}", response_model=DatasetInfo, tags=["Datasets"])
async def get_dataset(dataset_id: str):
    """Get dataset by ID"""
    if dataset_id not in datasets:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found")
    return datasets[dataset_id]

@app.get("/api/experiments", response_model=List[ExperimentInfo], tags=["Experiments"])
async def get_experiments():
    """Get all experiments"""
    return list(experiments.values())

@app.get("/api/experiments/{experiment_id}", response_model=ExperimentInfo, tags=["Experiments"])
async def get_experiment(experiment_id: str):
    """Get experiment by ID"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    return experiments[experiment_id]

@app.post("/api/experiments", response_model=ExperimentInfo, tags=["Experiments"])
async def create_experiment(config: ExperimentConfig):
    """Create a new experiment"""
    import uuid
    import datetime
    
    # Generate unique ID
    experiment_id = str(uuid.uuid4())
    
    # Create experiment
    experiment = ExperimentInfo(
        id=experiment_id,
        name=config.name,
        algorithm=config.algorithm,
        status="created",
        start_time=datetime.datetime.now().isoformat(),
    )
    
    # Store experiment
    experiments[experiment_id] = experiment
    
    logger.info(f"Created experiment: {experiment_id}")
    return experiment

@app.post("/api/experiments/{experiment_id}/start", response_model=ExperimentInfo, tags=["Training"])
async def start_training(experiment_id: str, background_tasks: BackgroundTasks):
    """Start training for an experiment"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    experiment = experiments[experiment_id]
    if experiment.status == "running":
        raise HTTPException(status_code=400, detail=f"Experiment {experiment_id} is already running")
    
    # Update status
    experiment.status = "running"
    
    # Start training in background
    # In a real implementation, this would start the actual training process
    # background_tasks.add_task(run_training, experiment_id)
    
    logger.info(f"Started training for experiment: {experiment_id}")
    return experiment

@app.post("/api/experiments/{experiment_id}/stop", response_model=ExperimentInfo, tags=["Training"])
async def stop_training(experiment_id: str):
    """Stop training for an experiment"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    experiment = experiments[experiment_id]
    if experiment.status != "running":
        raise HTTPException(status_code=400, detail=f"Experiment {experiment_id} is not running")
    
    # Update status
    experiment.status = "stopped"
    
    # Stop training process
    # In a real implementation, this would stop the actual training process
    
    logger.info(f"Stopped training for experiment: {experiment_id}")
    return experiment

@app.post("/api/generate", response_model=GenerationResponse, tags=["Inference"])
async def generate_text(request: GenerationRequest):
    """Generate text using a trained model"""
    if request.model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    # In a real implementation, this would use the actual model for generation
    # Here we just return a placeholder response
    return GenerationResponse(
        generations=["This is a placeholder response from the RLHF model."],
        model_id=request.model_id,
        prompt=request.prompt,
    )

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False, log_level: str = "info"):
    """Start the API server
    
    Args:
        host: Host to bind to
        port: Port to bind on
        reload: Enable auto-reload
        log_level: Log level
    """
    uvicorn.run("rlhf.api.server:app", host=host, port=port, reload=reload, log_level=log_level)

# Add sample data for demo purposes
def add_sample_data():
    """Add sample models and datasets for demo purposes"""
    # Add sample models
    models["llama2-7b"] = ModelInfo(
        id="llama2-7b",
        name="Llama 2 (7B)",
        type_name="causal_lm",
        parameters=7_000_000_000,
        description="Llama 2 7B base model from Meta",
    )
    
    models["llama2-13b"] = ModelInfo(
        id="llama2-13b",
        name="Llama 2 (13B)",
        type_name="causal_lm",
        parameters=13_000_000_000,
        description="Llama 2 13B base model from Meta",
    )
    
    models["reward-deberta"] = ModelInfo(
        id="reward-deberta",
        name="Reward Model (DeBERTa)",
        type_name="reward_model",
        parameters=900_000_000,
        description="DeBERTa-based reward model for RLHF",
    )
    
    # Add sample datasets
    datasets["hh-rlhf"] = DatasetInfo(
        id="hh-rlhf",
        name="HH-RLHF",
        type_name="preference_pairs",
        size=50000,
        description="Helpful and Harmless RLHF dataset with preference pairs",
    )
    
    datasets["anthropic-hh"] = DatasetInfo(
        id="anthropic-hh",
        name="Anthropic Helpful & Harmless",
        type_name="preference_pairs",
        size=160000,
        description="Anthropic's Helpful and Harmless preference dataset",
    )
    
    logger.info("Added sample models and datasets")

# Add sample data at module load time
add_sample_data()
