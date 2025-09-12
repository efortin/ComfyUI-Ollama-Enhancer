class DummyClip:
    """Stub CLIP class for tests, not used in string-return mode."""

    @staticmethod
    def encode(text: str):
        return text

    @staticmethod
    def tokenize(text: str):
        return text

    @staticmethod
    def encode_from_tokens(tokens, return_pooled=True, return_dict=True):
        return {"cond": tokens}
