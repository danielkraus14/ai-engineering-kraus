import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from pydantic import BaseModel, SecretStr, Field, model_validator
from openai import AsyncOpenAI, OpenAIError
from anthropic import AsyncAnthropic, AnthropicError
from google import genai
from dotenv import load_dotenv
import os

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class LLMConfig(BaseModel):
    provider: Provider
    model: str
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    gemini_api_key: Optional[SecretStr] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    @model_validator(mode="after")
    def check_required_key(self):
        if self.provider == Provider.OPENAI and self.openai_api_key is None:
            raise ValueError("openai_api_key is required when provider is OPENAI")
        if self.provider == Provider.ANTHROPIC and self.anthropic_api_key is None:
            raise ValueError("anthropic_api_key is required when provider is ANTHROPIC")
        if self.provider == Provider.GEMINI and self.gemini_api_key is None:
            raise ValueError("gemini_api_key is required when provider is GEMINI")
        return self

class BaseLLMClient(ABC):
    @abstractmethod
    async def chat(self, prompt: str) -> str:
        pass

class LLMError(Exception):
    """Raised when a provider call fails, hiding SDK-specific exceptions."""
    pass

class OpenAIClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai_api_key.get_secret_value())  

    async def chat(self, prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            raise LLMError(f"OpenAI call failed: {e}") from e
            

class AnthropicClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AsyncAnthropic(api_key=config.anthropic_api_key.get_secret_value())

    async def chat(self, prompt: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except (AnthropicError, TypeError) as e:
            raise LLMError(f"AnthropicAI call failed: {e}") from e

class GeminiClient(BaseLLMClient):
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = genai.Client(api_key=config.gemini_api_key.get_secret_value())
        
    async def chat(self, prompt: str) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.config.model,
                contents=prompt
            )
            return response.text
        except genai.errors.APIError as e:
            raise LLMError(f"GeminiAI call failed: {e}") from e

class LLMFactory:
    @staticmethod
    def create_client(config: LLMConfig) -> BaseLLMClient:
        if config.provider == Provider.OPENAI:
            return OpenAIClient(config)
        elif config.provider == Provider.ANTHROPIC:
            return AnthropicClient(config)
        elif config.provider == Provider.GEMINI:
                    return GeminiClient(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

async def main():
    load_dotenv()
    
    config = LLMConfig(
        provider=Provider.GEMINI,
        model="gemini-3.6-flash",
        gemini_api_key=os.environ.get("GOOGLE_API_KEY"),
    )
    
    client = LLMFactory.create_client(config)
    response = await client.chat("Say hello in one short sentence")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
