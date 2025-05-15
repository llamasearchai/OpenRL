from typing import Any, Dict, List, Optional

from rlhf.core.logging import get_logger

logger = get_logger("LangChainIntegration")

try:
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    from langchain.schema import Generation, LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Install with 'pip install langchain'")
    LANGCHAIN_AVAILABLE = False
    # Create dummy classes for type hints
    class LLM:
        pass
    class CallbackManagerForLLMRun:
        pass


class RLHFModel(LLM):
    """LangChain integration for RLHF fine-tuned models"""
    
    model_id: str
    """ID of the RLHF model to use"""
    
    api_url: str = "http://localhost:8000"
    """URL of the RLHF API server"""
    
    api_key: Optional[str] = None
    """API key for authentication"""
    
    temperature: float = 0.7
    """Temperature for sampling"""
    
    top_p: float = 0.9
    """Top-p sampling parameter"""
    
    max_tokens: int = 256
    """Maximum number of tokens to generate"""
    
    def __init__(self, **kwargs):
        """Initialize the RLHF model
        
        Args:
            model_id: ID of the RLHF model to use
            api_url: URL of the RLHF API server
            api_key: API key for authentication
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            max_tokens: Maximum number of tokens to generate
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available. Install with 'pip install langchain'")
        
        super().__init__(**kwargs)
        self._validate_parameters()
        
        # Import here to avoid dependency issues
        import requests
        self.client = requests
        
        logger.info(f"Initialized RLHF model integration with model {self.model_id}")
    
    def _validate_parameters(self):
        """Validate model parameters"""
        if not self.model_id:
            raise ValueError("model_id is required")
        
        if self.temperature < 0 or self.temperature > 2:
            logger.warning(f"Temperature {self.temperature} outside recommended range [0, 2]")
        
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError(f"top_p must be between 0 and 1, got {self.top_p}")
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM"""
        return "rlhf_model"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get parameters used to identify this LLM"""
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """Call the RLHF API to generate text
        
        Args:
            prompt: Prompt to generate from
            stop: List of stop sequences
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        # Prepare headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Prepare request data
        data = {
            "model_id": self.model_id,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "stop": stop,
            "num_return_sequences": 1,
        }
        
        # Send request to API
        try:
            response = self.client.post(
                f"{self.api_url}/api/generate",
                json=data,
                headers=headers,
            )
            response.raise_for_status()
            
            result = response.json()
            generations = result.get("generations", [])
            
            if not generations:
                logger.warning("Empty response from API")
                return ""
            
            return generations[0]
            
        except Exception as e:
            logger.error(f"Error calling RLHF API: {e}")
            raise
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> LLMResult:
        """Generate from multiple prompts
        
        Args:
            prompts: List of prompts to generate from
            stop: List of stop sequences
            run_manager: Callback manager
            **kwargs: Additional parameters
            
        Returns:
            LLMResult with generations
        """
        generations = []
        
        for prompt in prompts:
            generations.append([
                Generation(text=self._call(prompt, stop, run_manager, **kwargs))
            ])
        
        return LLMResult(generations=generations)


def create_rlhf_chain(model_id: str, api_url: str = "http://localhost:8000", **kwargs):
    """Create a LangChain with an RLHF model
    
    Args:
        model_id: ID of the RLHF model to use
        api_url: URL of the RLHF API server
        **kwargs: Additional parameters for the model
        
    Returns:
        LangChain chain
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available. Install with 'pip install langchain'")
    
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    
    # Create RLHF model
    model = RLHFModel(
        model_id=model_id,
        api_url=api_url,
        **kwargs
    )
    
    # Create default prompt template
    prompt = PromptTemplate(
        input_variables=["input"],
        template="{input}",
    )
    
    # Create chain
    chain = LLMChain(llm=model, prompt=prompt)
    
    return chain