from openai import OpenAI
import torch

class OpenRouterModel:
    """
    Class to handle interactions with OpenRouter API
    """
    def __init__(self, api_key, site_url, site_name, model_name="google/gemini-2.0-flash-thinking-exp:free"):
        """
        Initialize OpenRouter client
        
        Args:
            api_key (str): OpenRouter API key
            site_url (str): Site URL for rankings on openrouter.ai
            site_name (str): Site name for rankings on openrouter.ai
            model_name (str): Model name to use
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.extra_headers = {
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        }
        self.model_name = model_name
        # For compatibility with the original pipeline
        self.device = torch.device("cpu")  # This is just for compatibility
        self.__class__.__name__ = "OpenRouterModel"  # For output model name consistency
        
    def generate_response(self, prompt, max_tokens=512):
        """
        Generate a response using the OpenRouter API
        
        Args:
            prompt (str): The input prompt
            max_tokens (int): Maximum number of tokens to generate
            
        Returns:
            str: The generated response
        """
        completion = self.client.chat.completions.create(
            extra_headers=self.extra_headers,
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens
        )
        
        return completion.choices[0].message.content


def load_openrouter_model(api_key, site_url, site_name, model_name="google/gemini-2.0-flash-thinking-exp:free"):
    """
    Load the OpenRouter model
    
    Args:
        api_key (str): OpenRouter API key
        site_url (str): Site URL for rankings on openrouter.ai
        site_name (str): Site name for rankings on openrouter.ai
        model_name (str): Model name to use
        
    Returns:
        tuple: (model, None) - The model and a placeholder for tokenizer
    """
    model = OpenRouterModel(api_key, site_url, site_name, model_name)
    # Return None for tokenizer since we don't need it with OpenRouter
    return model, None


def generate_openrouter_outputs(model, _, prompt, num_outputs=5):
    """
    Generate outputs from OpenRouter model
    
    Args:
        model: The OpenRouter model
        _: Placeholder for tokenizer (not used)
        prompt (str): The input prompt
        num_outputs (int): Number of outputs to generate
        
    Returns:
        list: List of dictionaries containing model outputs
    """
    model_outputs = []
    
    # Generate multiple outputs
    for i in range(num_outputs):
        output = model.generate_response(prompt)
        output_entry = {
            "model": model.__class__.__name__,
            "beam": i + 1,  # For compatibility with the original pipeline
            "output": output
        }
        model_outputs.append(output_entry)
        print(f"Output {i+1}: {output}\n")
    
    return model_outputs 