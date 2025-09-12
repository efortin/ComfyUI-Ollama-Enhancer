import torch

class DummyClip:
    """Stub CLIP class for local dev/CLI usage without ComfyUI."""

    @staticmethod
    def encode(text: str):
        # Simule un embedding tensor (batch=1, dim=768 comme CLIP-L)
        return torch.ones(1, 768) * hash(text) % 1000

    @staticmethod
    def tokenize(text: str):
        # En vrai, ça retourne des IDs de tokens ; ici on garde une string
        return f"[TOKENS {text}]"

    @staticmethod
    def encode_from_tokens(tokens, return_pooled=True, return_dict=True):
        # Simule une sortie ComfyUI: cond tensor + pooled_output
        cond = torch.ones(1, 768) * hash(tokens) % 1000
        pooled = cond.mean(dim=-1, keepdim=True)  # simul "pooled_output"

        result = {"cond": cond}
        if return_pooled:
            result["pooled_output"] = pooled
        if return_dict:
            return result
        return cond