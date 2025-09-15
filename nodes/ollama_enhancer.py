"""
ComfyUI custom node: OllamaEnhancer

This node enriches user prompts using the Ollama API and encodes them
into CLIP-compatible CONDITIONING tensors for use with Stable Diffusion / Flux.
"""

import argparse
import json
import logging
import random
from json import JSONDecodeError
from pathlib import Path
from string import Template
from typing import Final, List

from ollama import Client, ResponseError
from .models import DummyClip  # Local stub for testing without ComfyUI


class OllamaEnhancer:
    """
    A ComfyUI node that generates positive and negative prompts with Ollama
    and encodes them into CONDITIONING for Stable Diffusion pipelines.
    """

    @staticmethod
    def seed():
        """Return a random seed for reproducibility."""
        return random.randint(0, 2**32 - 1)

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable=invalid-name
        """
        Define input types for ComfyUI node interface.
        """
        return {
            "required": {
                "clip": ("CLIP",),
                "user_prompt": ("STRING", {"multiline": True}),
                "ollama_url": ("STRING", {"default": "http://ollama:11434"}),
                "template_path": ("STRING", {"default": "prompt.jinja"}),
                "enhance_positive": (["true", "false"], {"default": "true"}),
                "reuse_running_model": (["true", "false"], {"default": "true"}),
                "fallback_model": ("STRING", {"default": "llama3.2:3b"}),
                "fallback_force_cpu": (["true", "false"], {"default": "true"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "generate"
    CATEGORY = "Ollama"

    ALWAYS_NEGATIVE: Final[List[str]] = [
        "duplicate", "cloned", "extra heads", "extra limbs", "deformed",
        "distorted", "warped", "surreal", "cgi", "watermark", "text",
        "logo", "lowres", "blurry", "low resolution", "blurry details",
        "pixelation", "distortion", "grainy textures", "overexposure",
        "underexposure", "washed-out colors", "dull colors", "artifacts",
        "noise", "poor lighting", "flat composition", "lack of depth",
        "unnatural shadows", "oversaturation", "unbalanced contrast",
        "unrealistic details", "amateurish quality", "unprofessional finish",
    ]

    @classmethod
    def _encode_text(cls, clip_model, text: str):
        """
        Encode text via CLIP and wrap into (tensor, dict) format for ComfyUI.
        """
        cond_dict = clip_model.encode_from_tokens(
            clip_model.tokenize(text),
            return_pooled=True,
            return_dict=True,
        )
        return [[cond_dict.pop("cond"), cond_dict]]

    @staticmethod
    def load_instruction_template(path="prompt.jinja", **kwargs) -> str:
        """
        Load and render a Jinja-like template file with substitutions.
        """
        base_dir = Path(__file__).parent
        full_path = (base_dir / path).resolve()
        text = full_path.read_text(encoding="utf-8")
        tpl = Template(text)
        return tpl.safe_substitute(**kwargs).strip()

    @staticmethod
    def retrieve_running_models(client: Client, fallback_model: str) -> str:
        """
        Return the first running Ollama model or a fallback model.
        """
        response = client.ps()
        if len(response.models) > 0:
            return response.models[0].model
        return fallback_model

    @classmethod
    def generate(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
        cls,
        clip,
        user_prompt: str,
        ollama_url: str,
        template_path: str,
        enhance_positive: str,
        reuse_running_model: str,
        fallback_model: str,
        fallback_force_cpu: str,
        seed: int,
    ):
        """
        Main ComfyUI entrypoint.  
        Generates positive/negative prompts with Ollama, encodes them into
        CLIP CONDITIONING, and returns them.
        """
        logging.info("Running OllamaEnhancer (seed=%s)", seed)
        client = Client(host=ollama_url)

        force_cpu_bool = str(fallback_force_cpu).lower() in ("true", "1", "yes")
        enhance_positive_bool = str(enhance_positive).lower() in ("true", "1", "yes")

        # Build Ollama instruction
        instruction = OllamaEnhancer.load_instruction_template(
            template_path, user_prompt=user_prompt
        )
        logging.info("OllamaEnhancer generated instruction: %s", instruction)

        selected_model = (
            OllamaEnhancer.retrieve_running_models(client, fallback_model=fallback_model)
            if reuse_running_model
            else fallback_model
        )
        fallback_model_options = {"num_gpu": 0} if force_cpu_bool else {}

        try:
            resp = client.generate(
                model=selected_model,
                prompt=instruction,
                options=fallback_model_options if selected_model == fallback_model else {},
                format="json" if selected_model == fallback_model else None,
            )
        except ResponseError as e:
            logging.error("OllamaEnhancer generation error: %s", e)
            return (
                cls._encode_text(clip, user_prompt),
                cls._encode_text(clip, ", ".join(cls.ALWAYS_NEGATIVE)),
            )

        raw = resp.get("response", "").strip()
        logging.info("raw response: %s", raw)

        try:
            result = json.loads(raw) if isinstance(raw, str) else raw

            # Positive text
            positive_text = " ".join(result.get("positive_text", [user_prompt])).strip()
            if not enhance_positive_bool:
                positive_text = user_prompt

            # Negative text
            negative_list = result.get("negative_text", [])
            negative_text = ", ".join(set(negative_list + cls.ALWAYS_NEGATIVE)).strip()

            logging.info("Positive text: %s", positive_text)
            logging.info("Negative text: %s", negative_text)

            return (
                cls._encode_text(clip, positive_text),
                cls._encode_text(clip, negative_text),
            )

        except JSONDecodeError as e:
            logging.error("Failed to decode Ollama response: %s", e)
            return (
                cls._encode_text(clip, user_prompt),
                cls._encode_text(clip, ", ".join(cls.ALWAYS_NEGATIVE)),
            )


# Register node for ComfyUI
NODE_CLASS_MAPPINGS = {"OllamaEnhancerNode": OllamaEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"OllamaEnhancerNode": "OllamaEnhancerNode"}



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate positive and negative prompts with Ollama")

    parser.add_argument("--user_prompt", type=str, required=True, help="User prompt text")
    parser.add_argument("--template_path", type=str, default="prompt.jinja", help="Path to template file")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model to use")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--enhance_positive", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True)
    parser.add_argument("--force_cpu", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=False)

    args = parser.parse_args()
    dummyClip = DummyClip()  # stub for local dev

    node = OllamaEnhancer()
    pos, neg = node.generate(
        clip=dummyClip,
        user_prompt=args.user_prompt,
        template_path=args.template_path,
        ollama_url=args.ollama_url,
        enhance_positive=args.enhance_positive,
        reuse_running_model="true",
        fallback_model=args.model,
        fallback_force_cpu=args.force_cpu,
        seed=OllamaEnhancer.seed(),
    )

    print("Positive:", pos)
    print("Negative:", neg)
