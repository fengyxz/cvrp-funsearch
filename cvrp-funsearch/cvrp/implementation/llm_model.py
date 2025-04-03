import os
import time
import logging
import openai
from typing import Collection

class BaseLLMModel:
    def __init__(self):
        pass

    def call(self, prompt):
        raise NotImplementedError("need to complete!")
    

# 定义调用 GPT 的子类
class GPTModel(BaseLLMModel):
    def __init__(self, api_key, model="gpt-4o"):
        super().__init__()
        self.api_key = api_key
        self.system_prompt= """
        You are a state-of-the-art Python code completion system.
        Your primary task is to solve the Capacitated Vehicle Routing Problem (CVRP) and provide the optimal solution as much as possible.
        You will be provided with a list of functions, and your job is to improve the code. 
        1. Observe the patterns in the dataset and adjust the code according to the data characteristics.
        2. Keep the code concise and comments brief.
        3. You may use the numpy and itertools libraries.
        The code you generate will be appended to the user prompt and run as a Python program.
        """
        BASE_URL = 'https://api.bltcy.ai/v1'
        self.client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)
        self.model = model

    def call(self, prompt):
        content = f"${self.system_prompt} ${prompt}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": content}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return None
        
class DsModel(BaseLLMModel):
    def __init__(self, api_key, model="deepseek-chat"):
        super().__init__()
        self.api_key = api_key
        self.system_prompt= """
        You are a state-of-the-art Python code completion system.
        Your primary task is to solve the Capacitated Vehicle Routing Problem (CVRP) and provide the optimal solution as much as possible.
        You will be provided with a list of functions, and your job is to improve the code. 
        1. Observe the patterns in the dataset and adjust the code according to the data characteristics.
        2. Keep the code concise and comments brief.
        3. You may use the numpy and itertools libraries.
        The code you generate will be appended to the user prompt and run as a Python program.
        """
        BASE_URL = "https://api.deepseek.com"
        self.client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)
        self.model = model

    def call(self, prompt):
        content = f"${self.system_prompt} ${prompt}"
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": content}
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling Deepseek: {e}")
            return None
        
class LLM:
  """Language model that predicts continuation of provided source code."""

  def __init__(self, samples_per_prompt: int, model:BaseLLMModel) -> None:
    # define the prompt pool
    self._model = model
    self._samples_per_prompt = samples_per_prompt

  def _draw_sample(self, prompt: str) -> str:
    """Returns a predicted continuation of `prompt`."""
    return self._model.call(prompt)
    
  def draw_samples(self, prompt: str) -> Collection[str]:
    """Returns multiple predicted continuations of `prompt`."""
    return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]
