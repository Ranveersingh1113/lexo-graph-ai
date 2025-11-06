"""
Figure Processing Module

Supports multiple Visual Language Models (VLM) for figure captioning.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional
from PIL import Image


class FigureProcessor:
    """Base class for figure processors."""
    
    def process(self, image: np.ndarray, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process figure image and return caption."""
        raise NotImplementedError


class BLIP2Processor(FigureProcessor):
    """BLIP-2 model for figure captioning."""
    
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        """
        Initialize BLIP-2.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
            
            self.model_name = model_name
            self.processor = Blip2Processor.from_pretrained(model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch accelerate")
        except Exception as e:
            raise Exception(f"Failed to load BLIP-2: {e}")
    
    def process(self, image: np.ndarray, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process figure image using BLIP-2."""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Create prompt
        if prompt is None:
            prompt = "Describe this image in detail, focusing on what the chart, graph, or figure shows."
        
        # Process
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_length=200)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            "model": "blip2",
            "caption": caption,
            "confidence": 0.95  # Placeholder - BLIP-2 doesn't provide confidence scores
        }


class LLaVAProcessor(FigureProcessor):
    """LLaVA model for figure captioning."""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initialize LLaVA.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            import torch
            
            self.model_name = model_name
            self.processor = LlavaProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch accelerate")
        except Exception as e:
            raise Exception(f"Failed to load LLaVA: {e}")
    
    def process(self, image: np.ndarray, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process figure image using LLaVA."""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Create prompt
        if prompt is None:
            prompt = "USER: <image>\nDescribe this image in detail, focusing on what the chart, graph, or figure shows.\nASSISTANT:"
        
        # Process
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
        generate_ids = self.model.generate(**inputs, max_length=200)
        caption = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Extract caption (remove prompt part)
        if "ASSISTANT:" in caption:
            caption = caption.split("ASSISTANT:")[-1].strip()
        
        return {
            "model": "llava",
            "caption": caption,
            "confidence": 0.95  # Placeholder
        }


class BLIPProcessor(FigureProcessor):
    """Original BLIP model (lighter alternative)."""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize BLIP.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
            
            self.model_name = model_name
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")
        except Exception as e:
            raise Exception(f"Failed to load BLIP: {e}")
    
    def process(self, image: np.ndarray, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process figure image using BLIP."""
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process
        if prompt:
            inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        
        out = self.model.generate(**inputs, max_length=200)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        return {
            "model": "blip",
            "caption": caption,
            "confidence": 0.95
        }


class GPT4VProcessor(FigureProcessor):
    """GPT-4V API for figure captioning (cloud-based)."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT-4V API client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        try:
            import os
            import openai
            import base64
            import io
            
            self.os = os
            self.openai = openai
            self.base64 = base64
            self.io = io
            
            if api_key:
                openai.api_key = api_key
            elif not self.os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key")
            
            self.loaded = True
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        except Exception as e:
            raise Exception(f"Failed to initialize GPT-4V: {e}")
    
    def process(self, image: np.ndarray, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process figure image using GPT-4V API."""
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = self.base64.b64encode(buffer).decode('utf-8')
        
        # Create prompt
        if prompt is None:
            prompt = "Describe this image in detail, focusing on what the chart, graph, or figure shows."
        
        # Call API
        response = self.openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        
        caption = response.choices[0].message.content
        
        return {
            "model": "gpt4v",
            "caption": caption,
            "confidence": 1.0  # API doesn't provide confidence
        }


# Model registry
FIGURE_MODELS = {
    "blip2-opt-2.7b": lambda: BLIP2Processor("Salesforce/blip2-opt-2.7b"),
    "blip2-opt-6.7b": lambda: BLIP2Processor("Salesforce/blip2-opt-6.7b"),
    "blip2-flan-t5-xl": lambda: BLIP2Processor("Salesforce/blip2-flan-t5-xl"),
    "llava-1.5-7b": lambda: LLaVAProcessor("llava-hf/llava-1.5-7b-hf"),
    "llava-1.5-13b": lambda: LLaVAProcessor("llava-hf/llava-1.5-13b-hf"),
    "blip-base": lambda: BLIPProcessor("Salesforce/blip-image-captioning-base"),
    "blip-large": lambda: BLIPProcessor("Salesforce/blip-image-captioning-large"),
    "gpt4v": lambda: GPT4VProcessor(),
}


def create_figure_processor(model_name: str = "blip2-opt-2.7b", **kwargs) -> FigureProcessor:
    """Factory function to create figure processor."""
    if model_name in FIGURE_MODELS:
        return FIGURE_MODELS[model_name](**kwargs)
    else:
        raise ValueError(f"Unknown figure model: {model_name}. Available: {list(FIGURE_MODELS.keys())}")

