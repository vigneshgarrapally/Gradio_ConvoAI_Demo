"""
A module that contains utility functions for the chatbot.

Usage:
    Import this module into another module to use the utility functions.

Dependencies:
    - os
    - typing
    - asyncio
    - elevenlabs
    - openai
    - deepgram

Author:
    Vignesh Garrapally (garrapallyvignesh8055@gmail.com)
"""
import os
from typing import Optional
import asyncio
from elevenlabs import generate, set_api_key
import openai
from deepgram import Deepgram


class Chat:
    """
    A class that represents a conversation between a user and an AI assistant.

    Attributes:
    system (Optional[str]): The initial message from the system.
    messages (List[Dict[str, str]]): A list of messages in the conversation.

    Methods:
    prompt(content: str) -> str: Adds the user's message to the conversation, sends it to the AI assistant, and returns the assistant's response.
    """
    def __init__(self, system: Optional[str] = None):
        self.system = system
        self.messages = []

        if system is not None:
            self.messages.append({"role": "system", "content": system})

    def prompt(self, content: str) -> str:
        """
        Adds the user's message to the conversation, sends it to the AI assistant, and returns the assistant's response.

        Args:
        content (str): The user's message.

        Returns:
        str: The assistant's response.
        """
        self.messages.append({"role": "user", "content": content})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=self.messages
        )
        response_content = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": response_content})
        return response_content


def text_to_speech(text):
    """
    Convert text to speech using Eleven Labs API.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        bytes: The audio file generated from the text.
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

    Raises:
        Exception: If there is an error with the Deepgram API request
    """
    # create deepgram client
    dg_client = Deepgram(os.environ.get("DEEPGRAM_API_KEY"))
    options = {"punctuate": True, "model": "general", "tier": "enhanced"}
    with open(audio_filepath, "rb") as audio:
        source = {"buffer": audio, "mimetype": "wav"}
        response = await dg_client.transcription.prerecorded(source, options)
    try:
        transcription = response["results"]["channels"][0]["alternatives"][0]["transcript"]
    except KeyError:
        raise Exception("Error with Deepgram API request")
    return transcription


def speech_to_text(audio_filepath):
    """
    Convert speech to text using Deepgram's transcription API.

    Args:
        audio_filepath (str): The file path of the audio file to be transcribed.

    Returns:
        str: The transcription of the audio file.
    """
    transcription = asyncio.run(deepgram_transcribe(audio_filepath))
    return transcription


def get_llm_response(chat, input_text):
    """
    Get response from LLM.

    Args:
    chat (Object): The chat object to use for generating the response.
    input_text (str): The input text to use as the prompt for generating the response.

    Returns:
    str: The generated response from the chatbot.
    """
    response_output = chat.prompt(input_text)
    return response_output