import yaml
import os

class PromptBuilder:
    def __init__(self, prompt_path: str = "prompts.yaml"):
        self.prompt_path = prompt_path
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"Prompts file not found at {self.prompt_path}")
        with open(self.prompt_path, "r") as f:
            return yaml.safe_load(f)

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found in {self.prompt_path}")
        
        prompt_template = self.prompts[prompt_name]
        return prompt_template.format(**kwargs)
