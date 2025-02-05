# Initalize a pipeline
from kokoro import KPipeline

# from IPython.display import display, Audio
# import soundfile as sf
import os
import uuid
import re
from typing import Literal
from speaches import kokoro_utils

# import soundfile as sf
import numpy as np
import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence


def create_audio_dir():
    """Creates the 'kokoro_audio' directory in the root folder if it doesn't exist."""
    root_dir = os.getcwd()  # Use current working directory instead of __file__
    audio_dir = os.path.join(root_dir, "kokoro_audio")

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        print(f"Created directory: {audio_dir}")
    else:
        print(f"Directory already exists: {audio_dir}")
    return audio_dir


last_used_language = "a"
pipeline = KPipeline(lang_code=last_used_language)
temp_folder = create_audio_dir()


def update_pipeline(new_lang: kokoro_utils.Language):
    """Updates the pipeline only if the language has changed."""
    global pipeline, last_used_language

    # Get language code, default to 'a' if not found
    # Only update if the language is different
    if new_lang != last_used_language:
        try:
            pipeline = KPipeline(lang_code=new_lang)
            last_used_language = new_lang  # Update last used language
            print(f"Pipeline updated to {new_lang}")
        except Exception as e:
            print(
                f"Error initializing KPipeline: {e}\nRetrying with default language..."
            )
            pipeline = KPipeline(lang_code="a")  # Fallback to English
            last_used_language = "a"


import re


def clean_text(text):
    # Define replacement rules
    replacements = {
        "–": " ",  # Replace en-dash with space
        "-": " ",  # Replace hyphen with space
        "**": " ",  # Replace double asterisks with space
        "*": " ",  # Replace single asterisk with space
        "#": " ",  # Replace hash with space
    }

    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis using regex (covering wide range of Unicode characters)
    emoji_pattern = re.compile(
        r"[\U0001F600-\U0001F64F]|"  # Emoticons
        r"[\U0001F300-\U0001F5FF]|"  # Miscellaneous symbols and pictographs
        r"[\U0001F680-\U0001F6FF]|"  # Transport and map symbols
        r"[\U0001F700-\U0001F77F]|"  # Alchemical symbols
        r"[\U0001F780-\U0001F7FF]|"  # Geometric shapes extended
        r"[\U0001F800-\U0001F8FF]|"  # Supplemental arrows-C
        r"[\U0001F900-\U0001F9FF]|"  # Supplemental symbols and pictographs
        r"[\U0001FA00-\U0001FA6F]|"  # Chess symbols
        r"[\U0001FA70-\U0001FAFF]|"  # Symbols and pictographs extended-A
        r"[\U00002702-\U000027B0]|"  # Dingbats
        r"[\U0001F1E0-\U0001F1FF]"  # Flags (iOS)
        r"",
        flags=re.UNICODE,
    )

    text = emoji_pattern.sub(r"", text)

    # Remove multiple spaces and extra line breaks
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tts_file_name(text):
    global temp_folder
    # Remove all non-alphabetic characters and convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Retain only alphabets and spaces
    text = (
        text.lower().strip()
    )  # Convert to lowercase and strip leading/trailing spaces
    text = text.replace(" ", "_")  # Replace spaces with underscores

    # Truncate or handle empty text
    truncated_text = text[:20] if len(text) > 20 else text if len(text) > 0 else "empty"

    # Generate a random string for uniqueness
    random_string = uuid.uuid4().hex[:8].upper()

    # Construct the file name
    file_name = f"{temp_folder}/{truncated_text}_{random_string}.wav"
    return file_name


def remove_silence_function(file_path, minimum_silence=50):
    # Extract file name and format from the provided path
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(
        sound, min_silence_len=100, silence_thresh=-45, keep_silence=minimum_silence
    )
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format=audio_format)
    return output_path


def generate_and_save_audio(
    text,
    lang=kokoro_utils.Language,
    voice="af_bella",
    speed=1,
    keep_silence_up_to=0.05,
):
    text = clean_text(text)
    update_pipeline(lang)
    generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")

    return generator

    save_path = tts_file_name(text)
    # Open the WAV file for writing
    with wave.open(save_path, "wb") as wav_file:
        # Set the WAV file parameters
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        wav_file.setframerate(24000)  # Sample rate

        # Process each audio chunk
        for i, (gs, ps, audio) in enumerate(generator):
            # print(f"{i}. {gs}")
            # print(f"Phonetic Transcription: {ps}")
            # display(Audio(data=audio, rate=24000, autoplay=i==0))
            print("\n")
            # Convert the Tensor to a NumPy array
            audio_np = audio.numpy()  # Convert Tensor to NumPy array
            audio_int16 = (audio_np * 32767).astype(np.int16)  # Scale to 16-bit range
            audio_bytes = audio_int16.tobytes()  # Convert to bytes

            # Write the audio chunk to the WAV file
            wav_file.writeframes(audio_bytes)

    return save_path
