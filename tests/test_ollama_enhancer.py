import http
import json
import logging
from unittest.mock import MagicMock, patch, ANY

import pytest
from ollama import ResponseError

from nodes.models import DummyClip
from nodes.ollama_enhancer import OllamaEnhancer


class FakeModel:
    def __init__(self, name):
        self.model = name


logging.basicConfig(level=logging.INFO)


@pytest.fixture
def fake_client():
    return MagicMock()


def test_generate_with_mock(fake_client):
    """
    GIVEN a model already exists in Ollama
    WHEN generate() is called
    THEN it should return enriched positive and negative prompts
    AND not attempt to pull the model again.
    """

    fake_client = fake_client()
    fake_client.list.return_value = type("ListResponse", (), {"models": [FakeModel("llama3.2:3b")]})()
    fake_client.generate.return_value = {
        "response": json.dumps({
            "positive_text": [
                "A lone, ethereal girl with wildflower-crowned hair dances under the warm, golden light of a setting sun in a mystical forest, surrounded by towering trees and vibrant autumn hues."],
            "negative_text": ["foo bar"] + OllamaEnhancer.ALWAYS_NEGATIVE,
        })
    }

    user_prompt = "a girl watching the mountain at sunset"

    with patch("nodes.ollama_enhancer.Client", return_value=fake_client):
        node = OllamaEnhancer()
        pos, neg = node.generate(
            clip=DummyClip(),
            user_prompt=user_prompt,
            model="llama3.2:3b",
            ollama_url="http://fake",
            template_path="prompt.jinja",
            enhance_positive=True,
            force_cpu=False
        )

    assert user_prompt != pos
    assert all(word in neg for word in OllamaEnhancer.ALWAYS_NEGATIVE)


def test_generate_with_ollama_pull_response_error():
    """
    GIVEN Ollama returns a ResponseError for a not found model
    WHEN generate() is called
    THEN it should fall back to user prompt and ALWAYS_NEGATIVE.
    """
    fake_client = MagicMock()
    fake_client.generate.side_effect = ResponseError("fake failure", http.HTTPStatus.NOT_FOUND)

    with patch("nodes.ollama_enhancer.Client", return_value=fake_client):
        node = OllamaEnhancer()
        pos, neg = node.generate(
            clip=DummyClip(),
            user_prompt="a fallback prompt",
            model="llama3.2:3b",
            ollama_url="http://fake",
            template_path="prompt.jinja",
            enhance_positive=True,
            force_cpu=False
        )

    assert pos == "a fallback prompt"
    assert neg == ", ".join(OllamaEnhancer.ALWAYS_NEGATIVE)

def test_generate_retrun_user_prompt_if_enhance_positive_is_false():
    """
    GIVEN Ollama returns a ResponseError for a not found model
    WHEN generate() is called
    THEN it should fall back to user prompt and ALWAYS_NEGATIVE.
    """
    fake_client = MagicMock()
    fake_client.generate.return_value = {
        "response": json.dumps({
            "positive_text": [
                "A lone, ethereal girl with wildflower-crowned hair dances under the warm, golden light of a setting sun in a mystical forest, surrounded by towering trees and vibrant autumn hues."],
            "negative_text": ["foo bar"] + OllamaEnhancer.ALWAYS_NEGATIVE,
        })
    }

    user_prompt = "a fallback prompt"
    with patch("nodes.ollama_enhancer.Client", return_value=fake_client):
        node = OllamaEnhancer()
        pos, neg = node.generate(
            clip=DummyClip(),
            user_prompt=user_prompt,
            model="llama3.2:3b",
            ollama_url="http://fake",
            template_path="prompt.jinja",
            enhance_positive=False,
            force_cpu=False
        )

    assert pos == user_prompt
    assert all(word in neg + ", ".join(OllamaEnhancer.ALWAYS_NEGATIVE) for word in OllamaEnhancer.ALWAYS_NEGATIVE)

def test_generate_return_user_prompt_if_force_cpu_is_true():
    """
    GIVEN Ollama returns a response with positive/negative text
    WHEN generate() is called with force_cpu=True
    THEN it should still call client.generate,
    AND ensure options are set to CPU mode (num_gpu=0).
    """
    fake_client = MagicMock()
    fake_client.generate.return_value = {
        "response": json.dumps({
            "positive_text": ["ignored text"],
            "negative_text": ["foo bar"] + OllamaEnhancer.ALWAYS_NEGATIVE,
        })
    }

    user_prompt = "a fallback prompt"
    with patch("nodes.ollama_enhancer.Client", return_value=fake_client):
        node = OllamaEnhancer()
        pos, neg = node.generate(
            clip=DummyClip(),
            user_prompt=user_prompt,
            model="llama3.2:3b",
            ollama_url="http://fake",
            template_path="prompt.jinja",
            enhance_positive=True,
            force_cpu=True  # 👈 force CPU execution
        )

    # ✅ Ensure client.generate was called with CPU option
    fake_client.generate.assert_called_once_with(
        model="llama3.2:3b",
        prompt=ANY,
        options={"num_gpu": 0},  # 👈 CPU mode
        format="json",
    )