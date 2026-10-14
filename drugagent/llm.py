"""LLM backends behind one small interface.

`ScriptedLLM` is deterministic and used by every test, so the agent graph is tested
without a model. `OllamaLLM` talks to a local Ollama server - the experiments in this
repo ran llama3.1:8b on a laptop CPU, with no API key and no data leaving the machine.
Any other provider is one class with a `complete` method.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Callable, Protocol

Message = dict  # {"role": "system" | "user" | "assistant", "content": str}


class LLM(Protocol):
    name: str

    def complete(self, messages: list[Message], json_mode: bool = False) -> str: ...


class ScriptedLLM:
    """Replays canned answers, or answers with a function of the messages."""

    def __init__(self, script: list[str] | Callable[[list[Message]], str], name: str = "scripted"):
        self.script, self.name, self.calls = script, name, []

    def complete(self, messages: list[Message], json_mode: bool = False) -> str:
        self.calls.append(messages)
        if callable(self.script):
            return self.script(messages)
        return self.script[min(len(self.calls), len(self.script)) - 1]


class OllamaLLM:
    """A local model served by Ollama. Greedy decoding and a fixed seed, for repeatability."""

    def __init__(self, model: str = "llama3.1:8b", host: str | None = None, num_ctx: int = 8192,
                 timeout: float = 900.0, **options):
        self.model, self.name = model, f"ollama/{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.options = {"temperature": 0, "seed": 0, "num_ctx": num_ctx, **options}
        if "OLLAMA_NUM_GPU" in os.environ:  # 0 forces CPU, e.g. when a GPU backend misbehaves
            self.options.setdefault("num_gpu", int(os.environ["OLLAMA_NUM_GPU"]))
        self.timeout, self.seconds = timeout, 0.0

    def complete(self, messages: list[Message], json_mode: bool = False) -> str:
        body = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if json_mode:
            body["format"] = "json"
        request = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            reply = json.load(response)
        self.seconds += time.perf_counter() - start
        if "error" in reply:
            raise RuntimeError(f"ollama: {reply['error']}")
        return reply["message"]["content"]

    @staticmethod
    def available(host: str | None = None) -> bool:
        host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        try:
            with urllib.request.urlopen(f"{host}/api/version", timeout=2):
                return True
        except OSError:
            return False
