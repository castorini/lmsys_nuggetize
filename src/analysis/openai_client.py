import os
import time

import openai
import tiktoken
from openai import AzureOpenAI

## As of March 12, 2025
OPENAI_PRICING = {
    "gpt-4o-mini": {"input": 0.15 / 1000000, "output": 0.6 / 1000000},
    "gpt-4o-mini-batch": {"input": 0.075 / 1000000, "output": 0.30 / 1000000},
    "gpt-4o": {"input": 2.50 / 1000000, "output": 10.00 / 1000000},
    "gpt-4o-batch": {"input": 1.25 / 1000000, "output": 5.00 / 1000000},
    "o3-mini": {"input": 1.10 / 1000000, "output": 4.40 / 1000000},
    "o1": {"input": 15.00 / 1000000, "output": 60.00 / 1000000},
    "gpt-4.5": {"input": 75.00 / 1000000, "output": 150.00 / 1000000},
    "gpt-4.1": {"input": 2.0 / 1000000, "output": 8.0 / 1000000},
    "gpt-4.1-mini": {"input": 0.4 / 1000000, "output": 1.6 / 1000000},
    "gpt-4.1-nano": {"input": 0.1 / 1000000, "output": 0.4 / 1000000},
}

# https://community.openai.com/t/whats-the-tokenization-algorithm-gpt-4-1-uses/1245758
# gpt-4.1 uses the same tokenizer as gpt-4o
TOKENIZER_OPENAI = {
    "gpt-4o-mini": "o200k_base",
    "gpt-4o-mini-batch": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-batch": "o200k_base",
    "o3-mini": "o200k_base",
    "o1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4.1-nano": "o200k_base",
}


class OpenAIClient:
    def __init__(
        self,
        model_name_or_path: str,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = None,
        wait: int = 10,
    ):
        self.deployment_name = model_name_or_path
        self.wait = wait
        print(f"Initializing OpenAI API Client: {model_name_or_path}")
        self.client = AzureOpenAI(
            azure_endpoint=(
                os.getenv("AZURE_OPENAI_ENDPOINT") if endpoint is None else endpoint
            ),
            api_key=os.getenv("AZURE_OPENAI_API_KEY") if api_key is None else api_key,
            api_version=(
                os.getenv("AZURE_OPENAI_API_VERSION")
                if api_version is None
                else api_version
            ),
        )
        self.price = OPENAI_PRICING[model_name_or_path]
        self.tokenizer = tiktoken.get_encoding(TOKENIZER_OPENAI[model_name_or_path])

    def count_tokens(self, text):
        # Encode the text (convert the text to tokens)
        tokens = self.tokenizer.encode(text, disallowed_special=())

        # Return the number of tokens
        return len(tokens)

    def cost(self, prompt: str, max_tokens: int = 512, n: int = 1):
        # Get the number of tokens for the prompt
        cost = 0.0
        input_tokens = self.count_tokens(prompt)

        # Calculate the cost for the input tokens for running the input prompt
        cost += input_tokens * self.price["input"]

        # Calculate the cost for the output tokens (assuming max_tokens)
        cost += max_tokens * self.price["output"] * n

        return cost

    def response(
        self,
        prompt: str,
        temperature: None,
        max_tokens: int = 1024,
        n: int = 1,
        disable_logging: bool = False,
        **kwargs,
    ):
        try:
            if not disable_logging:
                generation_config = {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "n": n,
                }
                print(f"OpenAI generation config: {generation_config}")

            if self.deployment_name in ["o3-mini", "o1"]:
                # No temperature parameter for this model.
                response = self.client.chat.completions.create(
                    model=self.deployment_name,  # model = "deployment_name".
                    messages=[{"role": "user", "content": f"{prompt}"}],
                    max_completion_tokens=max_tokens,
                )
                return response

            else:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,  # model = "deployment_name".
                    messages=[{"role": "user", "content": f"{prompt}"}],
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    n=n,
                    **kwargs,
                )
                return response

        except openai.RateLimitError as e:
            retry_time = int(str(e).split("retry after")[1].split("second")[0].strip())
            print(
                f"Rate limit exceeded. Retrying after waiting for {retry_time + 2} seconds..."
            )
            time.sleep(retry_time + 2)
            return self.response(
                prompt, temperature, max_tokens, n, disable_logging, **kwargs
            )

        except openai.InternalServerError:
            print(
                f"Internal server error. Retrying after waiting for {self.wait} seconds..."
            )
            time.sleep(self.wait)
            return self.response(
                prompt, temperature, max_tokens, n, disable_logging, **kwargs
            )

    def __call__(
        self,
        prompts: list[str],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        n: int = 1,
        disable_logging: bool = False,
        **kwargs,
    ):
        responses = []
        for prompt in prompts:
            output = self.response(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                disable_logging=disable_logging,
                **kwargs,
            )
            responses.append(output)
        return responses
