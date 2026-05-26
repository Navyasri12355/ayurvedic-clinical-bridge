"""
LLM Service for Ayurvedic Clinical Bridge

Provides unified interface for different LLM providers (OpenAI, Anthropic, Ollama, etc.)
for clinical reasoning, response generation, and conversational analysis.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GROQ = "groq"


@dataclass
class LLMConfig:
    """Configuration for LLM service"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # For Ollama or custom endpoints
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 30


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from conversation messages"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai

            self.client = openai.OpenAI(api_key=config.api_key)
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using OpenAI"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from conversation messages"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude LLM Provider"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Anthropic Claude"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            message = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=tokens,
                temperature=temp,
                messages=[{"role": "user", "content": full_prompt}],
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from conversation messages"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=tokens,
                temperature=temp,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OllamaProvider(BaseLLMProvider):
    """Ollama Local LLM Provider (open-source models)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import requests

            self.requests = requests
        except ImportError:
            raise ImportError("requests package required. Install: pip install requests")
        self.base_url = config.api_base or "http://localhost:11434"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Ollama"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": temp,
                    "num_predict": tokens,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

    def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from conversation messages"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        # Convert messages to prompt format
        prompt = ""
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt += f"{role}: {content}\n\n"
        prompt += "ASSISTANT:"

        try:
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": temp,
                    "num_predict": tokens,
                },
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise


class GroqProvider(BaseLLMProvider):
    """Groq Cloud LLM Provider (fast inference)"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from groq import Groq

            self.client = Groq(api_key=config.api_key)
        except ImportError:
            raise ImportError("groq package required. Install: pip install groq")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text using Groq"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from conversation messages"""
        temp = temperature or self.config.temperature
        tokens = max_tokens or self.config.max_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise


class LLMService:
    """Unified LLM Service with support for multiple providers"""

    _instance = None
    _provider: Optional[BaseLLMProvider] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, config: LLMConfig) -> "LLMService":
        """Initialize LLM service with configuration"""
        instance = cls()

        if config.provider == LLMProvider.OPENAI:
            instance._provider = OpenAIProvider(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            instance._provider = AnthropicProvider(config)
        elif config.provider == LLMProvider.OLLAMA:
            instance._provider = OllamaProvider(config)
        elif config.provider == LLMProvider.GROQ:
            instance._provider = GroqProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

        logger.info(
            f"LLM Service initialized with {config.provider.value} - {config.model_name}"
        )
        return instance

    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self._provider is not None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from prompt"""
        if not self.is_available():
            raise RuntimeError("LLM service not initialized")
        return self._provider.generate(prompt, system_prompt, temperature, max_tokens)

    def generate_with_context(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from conversation messages"""
        if not self.is_available():
            raise RuntimeError("LLM service not initialized")
        return self._provider.generate_with_context(messages, temperature, max_tokens)


# Clinical prompts templates
CLINICAL_REASONING_SYSTEM_PROMPT = """You are an expert Ayurvedic clinician with deep knowledge of:
- Classical Ayurvedic texts (Charaka Samhita, Sushruta Samhita)
- Modern clinical evidence for traditional herbs
- Herb-drug interactions and safety
- Dosha theory and constitutional diagnosis
- Treatment protocols and contraindications

Provide comprehensive, evidence-based clinical analysis. Be precise, concise, and clinically relevant.
Always prioritize patient safety and recommend consulting qualified practitioners for serious conditions."""

RESPONSE_GENERATION_SYSTEM_PROMPT = """You are a knowledgeable Ayurvedic health advisor. Your role is to:
- Explain Ayurvedic concepts clearly to general audiences
- Connect symptoms to Ayurvedic principles
- Suggest safe, evidence-supported herbs and lifestyle practices
- Always recommend consulting healthcare providers for serious conditions
- Use friendly, accessible language while maintaining accuracy

Generate clear, well-structured responses that educate users about Ayurvedic approaches."""

SAFETY_ANALYSIS_SYSTEM_PROMPT = """You are a clinical pharmacist specializing in herb-drug interactions.
Analyze interactions between herbs and medications with a focus on:
- Severity level (mild, moderate, severe)
- Mechanism of interaction
- Clinical relevance and patient risk
- Recommendations (avoid, monitor, dose adjust, alternatives)
- Evidence quality (strong, moderate, limited)

Be thorough but concise. Prioritize patient safety."""

EXPLANATION_SYSTEM_PROMPT = """You are a clinical explainer who makes complex medical predictions understandable.
When given model predictions and input data:
- Explain the reasoning in plain language
- Highlight key contributing factors
- Provide context from Ayurvedic principles
- Suggest next steps or follow-up questions
- Use structured formatting for clarity"""
