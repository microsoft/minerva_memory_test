import openai
from openai import AzureOpenAI
import time
import yaml

from azure.identity import (
    AzureCliCredential,
    get_bearer_token_provider,
)

import logging
logging.basicConfig(level=logging.INFO)


class Azure_LLM_API:
    def __init__(self, model_name, endpoint, api_version, client_id, scope):
        self.model_name = model_name

        self.azure_endpoint = endpoint
        self.api_version = api_version
        self.client_id = client_id
        self.scope = scope

        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            self.scope,
        )

        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_version=self.api_version,
            azure_ad_token_provider=token_provider,
        )

        self.max_tries = 3

        self.system_message = "You are a helpful AI assistant."
        self.max_new_tokens = 4096
        self.temperature = 0.0
        self.top_p = 1.0

    def generate(
        self,
        prompt,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        chat_history=None,
    ):

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        if temperature is None:
            temperature = self.temperature

        if top_p is None:
            top_p = self.top_p

        messages = [
            {"role": "user", "content": prompt},
        ]

        if chat_history:
            messages = chat_history + messages

        else:
            messages = [{"role": "system", "content": self.system_message}] + messages

        response = None

        start_time = time.time()

        for _ in range(self.max_tries):
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    stop=None,
                )

                if response.choices[0]:
                    break

            except openai.RateLimitError as e:
                logging.info("Rate limit exceeded. Waiting for 60 seconds.")
                time.sleep(60)
                continue

            except (
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
                openai.InternalServerError,
                openai.APIConnectionError,
                openai.APIStatusError,
            ) as e:
                
                logging.error(e)

                break

        if not response:
            return None
        
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

        return response.choices[0].message.content
    

if __name__ == "__main__":

    config_path = "src/azure_api_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    llm = Azure_LLM_API(model_name=config["model_name"],
                        endpoint=config["endpoint"],
                        api_version=config["api_version"],
                        client_id=config["client_id"],
                        scope=config["scope"])
    
    prompt = "What is the capital of France?"
    response = llm.generate(prompt)

    if response:
        print(f"Response: {response}")
    else:
        print("Failed to generate a response.")