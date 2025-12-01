"""
LLM Client for CGM MCP Server

Supports multiple LLM providers (OpenAI, Anthropic, etc.)
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from .config import LLMConfig


class BaseLLMClient(ABC):
    """Base class for LLM clients"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is healthy"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.api_base or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                payload = {
                    "model": self.config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                }

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check OpenAI API health"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/models", headers=self.headers
                )
                return response.status_code == 200
        except:
            return False


class AnthropicClient(BaseLLMClient):
    """Anthropic API client"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.api_base or "https://api.anthropic.com"
        self.headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                payload = {
                    "model": self.config.model,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", self.config.temperature),
                }

                response = await client.post(
                    f"{self.base_url}/v1/messages", headers=self.headers, json=payload
                )
                response.raise_for_status()

                data = response.json()
                return data["content"][0]["text"]

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check Anthropic API health"""
        try:
            # Anthropic doesn't have a dedicated health endpoint
            # We'll try a minimal request
            async with httpx.AsyncClient(timeout=10) as client:
                payload = {
                    "model": self.config.model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "Hi"}],
                }
                response = await client.post(
                    f"{self.base_url}/v1/messages", headers=self.headers, json=payload
                )
                return response.status_code == 200
        except:
            return False


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing"""

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        await asyncio.sleep(0.1)  # Simulate API delay

        if "analysis" in prompt.lower() and "extraction" in prompt.lower():
            return """
[start_of_analysis]
This is a mock analysis of the provided issue. The issue appears to be related to a bug in the authentication system.
[end_of_analysis]
[start_of_related_code_entities]
auth/models.py
auth/views.py
utils/validators.py
[end_of_related_code_entities]
[start_of_related_keywords]
authentication
validation
user_login
[end_of_related_keywords]
"""
        elif "relevant files" in prompt.lower():
            return """
[start_of_analysis]
Based on the issue description, the authentication system files are most relevant for this problem.
[end_of_analysis]
[start_of_relevant_files]
1. auth/models.py
2. auth/views.py
[end_of_relevant_files]
"""
        elif "score" in prompt.lower() and "file" in prompt.lower():
            return """
[start_of_analysis]
This file is highly relevant to the authentication issue and likely needs modification.
[end_of_analysis]
[start_of_score]
Score 4
[end_of_score]
"""
        elif "patch" in prompt.lower() or "code" in prompt.lower():
            return """
[start_of_analysis]
The authentication bug can be fixed by updating the password validation logic.
[end_of_analysis]
[start_of_patches]
PATCH 1:
File: auth/views.py
Description: Fix password validation
Line Range: 10-15
Original Code:
```
def authenticate_user(username, password):
    return validate_password(password)
```
Modified Code:
```
def authenticate_user(username, password):
    if not password:
        return False
    return validate_password(password)
```
Explanation: Added null check for password before validation
[end_of_patches]
[start_of_summary]
Fixed authentication bug by adding proper password validation checks.
[end_of_summary]
"""
        else:
            return "This is a mock response from the LLM client."

    async def health_check(self) -> bool:
        """Mock health check"""
        return True


class OllamaClient(BaseLLMClient):
    """Ollama local model client"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.api_base or "http://localhost:11434"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                payload = {
                    "model": self.config.model or "codellama:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get(
                            "temperature", self.config.temperature
                        ),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    },
                }

                response = await client.post(
                    f"{self.base_url}/api/generate", json=payload
                )
                response.raise_for_status()

                data = response.json()
                return data["response"]

        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check Ollama API health"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except:
            return False


class LMStudioClient(BaseLLMClient):
    """LM Studio local model client"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.api_base or "http://localhost:1234/v1"

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using LM Studio API"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                payload = {
                    "model": self.config.model or "local-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                }

                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
                response.raise_for_status()

                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"LM Studio API error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check LM Studio API health"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/models")
                return response.status_code == 200
        except:
            return False


class OllamaCloudClient(BaseLLMClient):
    """Ollama Cloud client - compatible with Ollama API format but running on cloud"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        # Default to Ollama Cloud API endpoint, but allow override
        self.base_url = config.api_base or "https://ollama.example.com"  # Placeholder - should be replaced with real Ollama Cloud URL
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama Cloud API (Ollama-compatible format)"""
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                payload = {
                    "model": self.config.model or "llama3",  # Default Ollama model
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get(
                            "temperature", self.config.temperature
                        ),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    },
                }

                response = await client.post(
                    f"{self.base_url}/api/generate", 
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()

                data = response.json()
                return data["response"]

        except Exception as e:
            logger.error(f"Ollama Cloud API error: {e}")
            raise

    async def health_check(self) -> bool:
        """Check Ollama Cloud API health"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/api/tags", headers=self.headers
                )
                return response.status_code == 200
        except:
            return False


class LLMClient:
    """Main LLM client that delegates to specific providers"""

    def __init__(self, config: LLMConfig):
        self.config = config

        if config.provider == "openai":
            self.client = OpenAIClient(config)
        elif config.provider == "anthropic":
            self.client = AnthropicClient(config)
        elif config.provider == "ollama":
            self.client = OllamaClient(config)
        elif config.provider == "ollama_cloud":
            self.client = OllamaCloudClient(config)
        elif config.provider == "lmstudio":
            self.client = LMStudioClient(config)
        elif config.provider == "mock":
            self.client = MockLLMClient(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        return await self.client.generate(prompt, **kwargs)

    async def health_check(self) -> bool:
        """Check if the LLM service is healthy"""
        return await self.client.health_check()

    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts concurrently"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
