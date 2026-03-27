from enum import Enum
from typing import Type

from openai import OpenAI


class InferenceClient():
    """Abstract base client for LLM inference servers.

    Provides a uniform interface for communicating with different types of LLM
    inference backends. Used by LLMManager to maintain a pool of stateful
    clients that can serve requests and track conversation context versions.

    Subclasses must implement the `run()` method to handle actual inference
    calls to their specific backend (e.g., vLLM, TGI, etc.).

    Attributes:
        model_name: Name of the LLM model running on the inference server.
        url: Base URL of the inference server endpoint (e.g.,
            'http://localhost:8000').
    """

    def __init__(self, name: str, model_name: str, url: str, **kwargs):
        self.name = name
        self.model_name = model_name
        self.url = url

    def run(self, prompt: str, max_tokens: int = 1024, **kwargs):
        raise NotImplementedError

    def __call__(self, prompt: str, max_tokens: int = 1024, **kwargs):
        return self.run(prompt, max_tokens, **kwargs)


class VLLMInferenceClient(InferenceClient):
    """vLLM inference client using OpenAI-compatible API.

    Concrete implementation of InferenceClient for vLLM inference servers.
    Wraps the standard, synchronous OpenAI Python client to communicate with 
    vLLM's OpenAI-compatible endpoint (/v1). Supports both streaming and 
    non-streaming inference.

    Attributes:
        model_name: Name of the LLM model loaded in vLLM.
        url: Base URL of the vLLM server (e.g., 'http://localhost:8000').
        client: Synchronous OpenAI client instance configured for vLLM endpoint.

    Example:
        >>> client = VLLMInferenceClient('MyClient', 'Qwen/Qwen2.5-7B-Instruct', 'http://localhost:8000')
        >>> response = client("Hello, how are you?", max_tokens=512)
        >>> print(response.choices[0].message.content)
    """  # noqa

    def __init__(self, name: str, model_name: str, url: str, **kwargs):
        super().__init__(name, model_name, url)

        self.client = OpenAI(
            base_url=f'{self.url}/v1',
            api_key='dummy-key'  # vLLM doesn't require a real API key
        )

    def run(self,
            prompt: str,
            max_tokens: int = 1024,
            temp: float = 0.7,
            seed: int = None,
            stream: bool = False,
            **kwargs):
        """Execute inference request on the vLLM server.

        Sends a chat completion request to the vLLM backend using
        OpenAI-compatible format. Supports both streaming and non-streaming
        responses in a blocking, synchronous manner.
        """
        chat_args = {
            'model': self.model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                },
            ],
            'stream': stream,
            'max_tokens': max_tokens,
            'temperature': temp,
            'extra_body': {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        }

        if stream:
            chat_args['stream_options'] = {"include_usage": True}

        if seed is not None:
            chat_args['seed'] = seed

        output = self.client.chat.completions.create(**chat_args)

        return output


class ClientType(Enum):
    VLLM = 'vllm'


CLIENT_MAPPINGS: dict[ClientType, Type[InferenceClient]] = {
    ClientType.VLLM: VLLMInferenceClient,
}


def create_inference_client(
    client_type: ClientType,
    name: str,
    model_name: str,
    url: str,
    **kwargs,
):
    """Factory function returning an inference client instance.

    How to use:
        vllm_client = create_inference_client(ClientType.VLLM, 'Client1', 'your_model', 'your_url')
    """  # noqa
    # Get inference client class
    client_class = CLIENT_MAPPINGS.get(client_type)
    if not client_class:
        raise ValueError(f'Invalid client type: {client_type}')

    # Initialize inference client with parameters
    client = client_class(name, model_name, url, **kwargs)

    return client
