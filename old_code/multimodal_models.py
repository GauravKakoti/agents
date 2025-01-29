from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from typing import Optional, Dict, Any
import os

class GroqMultiModal(OpenAIMultiModal):
    def __init__(self,
                model: str = "llama-3.2-90b-vision-preview", 
                api_base: str="https://api.groq.com/openai/v1",
                api_key: Optional[str] = None,
                **kwargs):
        api_key = api_key or os.getenv("GROQ_CLOUD_API_KEY")
        super().__init__(model=model, api_base=api_base, api_key=api_key)
    
    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        if self.model not in ["llama-3.2-90b-vision-preview"]:
            raise ValueError(
                f"Invalid model {self.model}. "
                f"Available models are: llama-3.2-90b-vision-preview"
            )
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_new_tokens is not None:
            # If max_tokens is None, don't include in the payload:
            # https://platform.openai.com/docs/api-reference/chat
            # https://platform.openai.com/docs/api-reference/completions
            base_kwargs["max_tokens"] = self.max_new_tokens
        return {**base_kwargs, **self.additional_kwargs}
        