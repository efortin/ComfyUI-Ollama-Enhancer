import argparse
import json
import logging
import random
from pathlib import Path
from string import Template
from typing import Final, List

from ollama import Client, ResponseError

from .models import DummyClip  # pour le test local




class OllamaEnhancer:
    """
    ComfyUI node: génère des prompts positifs/négatifs avec Ollama,
    puis encode en CONDITIONING compatible avec KSampler.
    """
    @staticmethod
    def seed():
        return random.randint(0, 2 ** 32 - 1)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "user_prompt": ("STRING", {"multiline": True}),
                "model": ("STRING", {"default": "llama3.2:3b"}),
                "ollama_url": ("STRING", {"default": "http://ollama:11434"}),
                "template_path": ("STRING", {"default": "prompt.jinja"}),
                "enhance_positive": (["true", "false"], {"default": "true"}),
                "force_cpu": (["true", "false"], {"default": "false"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1})
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "generate"
    CATEGORY = "Ollama"

    ALWAYS_NEGATIVE: Final[List[str]] = [
        "duplicate", "cloned", "extra heads", "extra limbs", "deformed",
        "distorted", "warped", "surreal", "cgi", "watermark", "text",
        "logo", "lowres", "blurry",  "low resolution", "blurry details", "pixelation", "distortion", "grainy textures",
        "overexposure", "underexposure", "washed-out colors", "dull colors",
        "artifacts", "noise", "poor lighting", "flat composition", "lack of depth",
        "unnatural shadows", "oversaturation", "unbalanced contrast",
        "unrealistic details", "amateurish quality", "unprofessional finish",
    ]

    @classmethod
    def _encode_text(cls, clip, text: str):
        """
        Encode un texte via CLIP et le met en format (tensor, {pooled_output}).
        """
        cond_dict = clip.encode_from_tokens(
            clip.tokenize(text),
            return_pooled=True,
            return_dict=True,
        )
        return [[cond_dict.pop("cond"), cond_dict]]

    @staticmethod
    def load_instruction_template(path="prompt.jinja", **kwargs) -> str:
        """Charge et rend un template Jinja simple depuis un fichier texte."""
        base_dir = Path(__file__).parent
        full_path = (base_dir / path).resolve()
        text = full_path.read_text(encoding="utf-8")
        tpl = Template(text)
        return tpl.safe_substitute(**kwargs).strip()

    @classmethod
    def generate(
            cls,
            clip,
            user_prompt: str,
            model: str,
            ollama_url,
            template_path,
            enhance_positive: str,
            force_cpu: str,
            seed: int,
    ):
        logging.info(f"Running OllamaEnhancer (seed={seed})")
        client = Client(host=ollama_url)

        force_cpu_bool = str(force_cpu).lower() in ("true", "1", "yes")
        enhance_positive_bool = str(enhance_positive).lower() in ("true", "1", "yes")

        # Instruction pour Ollama
        instruction = OllamaEnhancer.load_instruction_template(template_path, user_prompt=user_prompt)
        logging.info(f"OllamaEnhancer generated instruction: {instruction}")

        options = {"num_gpu": 0} if force_cpu_bool else {"num_gpu": 1}

        try:
            resp = client.generate(
                model=model,
                prompt=instruction,
                options=options,
                format="json",
            )
        except ResponseError as e:
            logging.error(f"OllamaEnhancer generation error: {e}")
            # Fallback : encode directement user_prompt et ALWAYS_NEGATIVE
            return (
                cls._encode_text(clip, user_prompt),
                cls._encode_text(clip, ", ".join(cls.ALWAYS_NEGATIVE)),
            )

        raw = resp.get("response", "").strip()
        logging.info(f"raw response: {raw}")

        try:
            result = json.loads(raw) if isinstance(raw, str) else raw

            # Texte positif
            positive_text = " ".join(result.get("positive_text", [user_prompt])).strip()
            if not enhance_positive_bool:
                positive_text = user_prompt

            # Texte négatif
            negative_list = result.get("negative_text", [])
            negative_text = ", ".join(set(negative_list + cls.ALWAYS_NEGATIVE)).strip()

            logging.info(f"Positive text: {positive_text}")
            logging.info(f"Negative text: {negative_text}")

            # Encode et renvoie au format attendu
            return (
                cls._encode_text(clip, positive_text),
                cls._encode_text(clip, negative_text),
            )

        except Exception as e:
            logging.error("Failed to parse Ollama response: %s", e)
            return (
                cls._encode_text(clip, user_prompt),
                cls._encode_text(clip, ", ".join(cls.ALWAYS_NEGATIVE)),
            )


# Enregistrement ComfyUI
NODE_CLASS_MAPPINGS = {"OllamaPosNegNode": OllamaEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"OllamaPosNegNode": "Ollama Pos+Neg (LLM)"}
# Empêcher ComfyUI de mettre ce nœud en cache
OllamaEnhancer.ALWAYS_RUN = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate positive and negative prompts with Ollama")

    parser.add_argument("--user_prompt", type=str, required=True, help="User prompt text")
    parser.add_argument("--template_path", type=str, default="prompt.jinja", help="Path to template file")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model to use")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--enhance_positive", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=True,
                        help="Enhance positive prompt (true/false)")
    parser.add_argument("--force_cpu", type=lambda x: str(x).lower() in ["true", "1", "yes"], default=False,
                        help="Force CPU usage (true/false)")

    args = parser.parse_args()

    clip = DummyClip()  # stub local

    node = OllamaEnhancer()
    pos, neg = node.generate(
        clip=clip,
        user_prompt=args.user_prompt,
        template_path=args.template_path,
        model=args.model,
        ollama_url=args.ollama_url,
        enhance_positive=args.enhance_positive,
        force_cpu=args.force_cpu,
        seed=OllamaEnhancer.seed(),
    )

    print("Positive:", pos)
    print("Negative:", neg)
