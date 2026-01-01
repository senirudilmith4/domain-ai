"""
LLM Client for Ollama integration
"""

import os
import requests
from typing import Optional, Generator
from dotenv import load_dotenv
from MVP.backend.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class LLMClient:
    """
    Ollama-based LLM client for local inference.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ):
        """
        Initialize Ollama LLM client.
        
        Args:
            model: Ollama model name (default from .env or "llama3")
            base_url: Ollama API base URL (default: http://localhost:11434)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        """
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "1024"))
        self.temperature = temperature or float(os.getenv("TEMPERATURE", "0.2"))
        
        # API endpoints
        self.generate_url = f"{self.base_url}/api/generate"
        self.chat_url = f"{self.base_url}/api/chat"
        
        logger.info(
            f"Initialized Ollama LLM | model={self.model}, "
            f"max_tokens={self.max_tokens}, temperature={self.temperature}"
        )
        
        # Verify Ollama is running
        self._check_ollama_connection()
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("✓ Ollama server is accessible")
            return True
        except requests.exceptions.ConnectionError:
            logger.error(
                f"✗ Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (run: ollama serve)"
            )
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama connection: {e}")
            return False
    
    def ask(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a grounded answer using retrieved context.
        
        Args:
            question: User's question
            context: Retrieved context from vector store
            system_prompt: Optional custom system prompt
            stream: Whether to stream the response
        
        Returns:
            Generated answer string
        """
        # Default system prompt for RAG
        if system_prompt is None:
            system_prompt = """You are a helpful assistant that answers questions based strictly on the provided context.

RULES:
- Use ONLY the information from the provided context
- Do NOT use your general knowledge or make assumptions
- If the answer is not in the context, say: "I cannot answer this based on the provided context."
- Be concise and direct in your answers
- Cite relevant parts of the context when appropriate"""
        
        # Build the prompt
        prompt = f"""{system_prompt}

Context:
{context}

Question: {question}

Answer:"""
        
        if stream:
            return self._generate_streaming(prompt)
        else:
            return self._generate_complete(prompt)
    
    def _generate_complete(self, prompt: str) -> str:
        """Generate complete response (non-streaming)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": False
        }
        
        try:
            logger.info("Sending request to Ollama...")
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "").strip()
            
            if not answer:
                logger.warning("Ollama returned empty response")
                return "Error: Received empty response from LLM."
            
            logger.info(f"Response generated successfully ({len(answer)} chars)")
            return answer
            
        except requests.exceptions.ConnectionError:
            error_msg = (
                "Cannot connect to Ollama. Please ensure:\n"
                "1. Ollama is installed (https://ollama.ai)\n"
                f"2. Ollama server is running: ollama serve\n"
                f"3. Model '{self.model}' is pulled: ollama pull {self.model}"
            )
            logger.error(error_msg)
            return f"Error: {error_msg}"
            
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out after 120 seconds")
            return "Error: LLM request timed out. Try a shorter context or question."
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ollama HTTP error: {e}")
            if e.response.status_code == 404:
                return (
                    f"Error: Model '{self.model}' not found. "
                    f"Pull it with: ollama pull {self.model}"
                )
            return f"Error: HTTP {e.response.status_code} - {e}"
            
        except Exception as e:
            logger.error(f"Ollama LLM call failed: {e}", exc_info=True)
            return f"Error: LLM generation failed - {str(e)}"
    
    def _generate_streaming(self, prompt: str) -> Generator[str, None, None]:
        """Generate streaming response (for real-time output)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": True
        }
        
        try:
            logger.info("Starting streaming request to Ollama...")
            response = requests.post(
                self.generate_url,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    try:
                        import json
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"Error: {str(e)}"
    
    def chat(
        self,
        messages: list,
        stream: bool = False
    ) -> str:
        """
        Use Ollama's chat API with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello"}]
            stream: Whether to stream the response
        
        Returns:
            Generated response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": stream
        }
        
        try:
            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "").strip()
            
        except Exception as e:
            logger.error(f"Chat API call failed: {e}", exc_info=True)
            return f"Error: {str(e)}"
    
    def get_available_models(self) -> list:
        """Get list of available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []


# Convenience function
def get_llm_client(model: Optional[str] = None) -> LLMClient:
    """
    Get an initialized LLM client.
    
    Args:
        model: Optional model name override
    
    Returns:
        Initialized LLMClient
    """
    return LLMClient(model=model)