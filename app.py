import gradio as gr

# Define the predict function
def predict(input):
    return input

# Set up the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.inputs.Textbox(label="Input"),
    outputs=gr.outputs.Label(label="Prediction")
)

# Run the interface
if __name__ == "__main__":
    iface.launch()