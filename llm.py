import os
from typing import Optional

class LLM:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = None
        self._load_model()

    def _load_model(self):
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            print(f"✅ Loaded LLM: {self.model_path}")
        except ImportError:
            print("❌ llama-cpp-python not installed. Please install: pip install llama-cpp-python")
            self.llm = None
        except Exception as e:
            print(f"❌ Failed to load LLM: {e}")
            self.llm = None

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.llm is None:
            return "LLM not available. Please check installation and model path."

        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stop=["</s>", "User:"]
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            print(f"LLM generation error: {e}")
            return "Error generating response."

    def generate_chat(self, messages: list, max_tokens: int = 512) -> str:
        # Simple chat wrapper for instruction-tuned models
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        return self.generate(prompt, max_tokens=max_tokens)