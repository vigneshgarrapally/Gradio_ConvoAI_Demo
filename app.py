"""
This module contains the code for a conversational AI bot that uses Gradio for input and output.

The bot uses the OpenAI Language Model (LLM) to generate responses to user inputs. 
The user can input audio through a microphone, and the bot will transcribe the audio to text using the DeepGram. 
The bot will then generate a response using the LLM model and convert the response to audio using the Eleven Labs API. 
The conversation history is stored in a list of tuples containing the user's input and the corresponding response.

The bot is launched using Gradio, which provides a user interface for the bot. 
The user can input audio through the microphone and click the "Send Audio" button to generate a response. 
The response audio is played automatically and the conversation history is displayed in the chatbot window.

Usage:
    Run this module to launch the bot using Gradio.

Dependencies:
    - gradio
    - openai
    - dotenv
    - utils (custom module)

Author:
    Vignesh Garrapally
"""

import os
import gradio as gr
import openai
from dotenv import load_dotenv
from utils import text_to_speech, speech_to_text, get_llm_response, Chat

load_dotenv()
chat = Chat(system="You are a helpful assistant.")
openai.api_key = os.getenv("OPENAI_API_KEY")


def process_inputs(audio, chat_history):
    """
    Process the user's audio input and generate a response using the LLM model.

    Args:
        audio (bytes): The user's audio input.
        chat_history (list): A list of tuples containing the user's input and the corresponding response.

    Returns:
        tuple: A tuple containing None (since we don't use the transcribed text), the response audio, and the updated chat history.
    """
    input_text = speech_to_text(audio)
    response = get_llm_response(chat, input_text)
    audio = text_to_speech(response)
    chat_history.append((input_text, response))
    return None, audio, chat_history


if __name__ == "__main__":
    function = process_inputs
    with gr.Blocks() as demo:
        with gr.Row():
            input_audio = gr.Audio(
                source="microphone",
                type="filepath",
                label="Source Audio",
                interactive=True,
                show_label=True,
                scale=2,
            )
            send_audio_button = gr.Button("Send Audio", interactive=True)
        with gr.Row():
            output_audio = gr.Audio(
                type="numpy",
                interactive=False,
                autoplay=True,
                label="Response Audio",
                show_label=True,
            )
        chatbot = gr.Chatbot()
        send_audio_button.click(
            function, [input_audio, chatbot], [input_audio, output_audio, chatbot]
        )
        with gr.Row():
            gr.ClearButton([input_audio, output_audio, chatbot])

    demo.launch()
