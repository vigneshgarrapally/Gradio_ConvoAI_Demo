"""
App file for conversational AI bot
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
    Process inputs
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
