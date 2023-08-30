"""
Utility functions
"""
import os
from typing import Optional
import asyncio

from elevenlabs import generate, set_api_key
import openai
from deepgram import Deepgram


class Chat:
    def __init__(self, system: Optional[str] = None):
        self.system = system
        self.messages = []

        if system is not None:
            self.messages.append({"role": "system", "content": system})

    def prompt(self, content: str) -> str:
        self.messages.append({"role": "user", "content": content})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.messages
        )
        response_content = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": response_content})
        return response_content


def text_to_speech(text):
    """
    Convert text to speech
    """
    set_api_key(os.getenv("ELEVENLABS_API_KEY"))
    audio = generate(
        text=text,
        voice="Bella",
        model="eleven_monolingual_v1",
    )
    return audio


async def deepgram_transcribe(audio_filepath):
    """
    Transcribe audio using Deepgram API

    Args:
        audio_filepath (str): Path to audio file

    Returns:
        str: Transcription
    """
    # create deepgram client
    dg_client = Deepgram(os.environ.get("DEEPGRAM_API_KEY"))
    options = {"punctuate": True, "model": "general", "tier": "enhanced"}
    with open(audio_filepath, "rb") as audio:
        source = {"buffer": audio, "mimetype": "wav"}
        response = await dg_client.transcription.prerecorded(source, options)
    transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]
    return transcription


def speech_to_text(audio_filepath):
    """
    Convert speech to text
    """
    transcription = asyncio.run(deepgram_transcribe(audio_filepath))
    return transcription


def get_llm_response(chat, input_text):
    """
    Get response from LLM
    """
    response_output = chat.prompt(input_text)
    return response_output
