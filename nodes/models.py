"""
Stub and helper models for ComfyUI Ollama Enhancer.
Includes lightweight CLIP replacements used for local testing.
"""


class DummyClip:
    """
    Minimal CLIP stub for tests.

    Provides the same interface as a real CLIP model but returns strings or
    simple dicts instead of tensors. Used to run tests and/or debug locally without comfy.
    """

    @staticmethod
    def encode(text: str) -> str:
        """Return the input text unchanged (real CLIP would return an embedding)."""
        return text

    @staticmethod
    def tokenize(text: str) -> str:
        """Return the input text unchanged (real CLIP would return token IDs)."""
        return text

    @staticmethod
    def encode_from_tokens(tokens: str, return_pooled=True, return_dict=True):
        """
        Mimic CLIP.encode_from_tokens but return strings instead of tensors.
        """
        cond = f"{tokens}"
        pooled = f"{tokens}" if return_pooled else None
        hidden = f"{tokens}"

        if return_dict:
            result = {"cond": cond, "last_hidden_state": hidden}
            if pooled is not None:
                result["pooled_output"] = pooled
            return result

        # If not returning dict, just return the "cond"
        return cond
