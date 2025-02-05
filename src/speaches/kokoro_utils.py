from collections.abc import AsyncGenerator
import logging
import time
from typing import Literal
import os

# from kokoro_onnx import Kokoro
import numpy as np
from huggingface_hub import list_repo_files

from speaches.audio import resample_audio

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # the default sample rate for Kokoro

Language = Literal["en-us", "en-gb", "es", "fr-fr", "hi", "it", "pt-br", "ja", "zh"]
LANGUAGES: list[Language] = [
    "en-us",
    "en-gb",
    "es",
    "fr-fr",
    "hi",
    "it",
    "pt-br",
    "ja",
    "zh",
]

LanguageToText = {
    "en-us": "English (US)",
    "en-gb": "English (UK)",
    "es": "Spanish",
    "fr-fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "pt-br": "Portuguese (Brazil)",
    "ja": "Japanese",
    "zh": "Chinese",
}


def get_voice_names(repo_id):
    """Fetches and returns a list of voice names (without extensions) from the given Hugging Face repository."""
    return [
        os.path.splitext(file.replace("voices/", ""))[0]
        for file in list_repo_files(repo_id)
        if file.startswith("voices/")
    ]


VOICE_IDS = get_voice_names("hexgrad/Kokoro-82M")


# async def generate_audio(
#     kokoro_tts: Kokoro,
#     text: str,
#     voice: str,
#     *,
#     language: Language = "en-us",
#     speed: float = 1.0,
#     sample_rate: int | None = None,
# ) -> AsyncGenerator[bytes, None]:
#     if sample_rate is None:
#         sample_rate = SAMPLE_RATE
#     start = time.perf_counter()
#     async for audio_data, _ in kokoro_tts.create_stream(text, voice, lang=language, speed=speed):
#         assert isinstance(audio_data, np.ndarray) and audio_data.dtype == np.float32 and isinstance(sample_rate, int)
#         normalized_audio_data = (audio_data * np.iinfo(np.int16).max).astype(np.int16)
#         audio_bytes = normalized_audio_data.tobytes()
#         if sample_rate != SAMPLE_RATE:
#             audio_bytes = resample_audio(audio_bytes, SAMPLE_RATE, sample_rate)
#         yield audio_bytes
#     logger.info(f"Generated audio for {len(text)} characters in {time.perf_counter() - start}s")
