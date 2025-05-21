from datasets import load_dataset
import gradio as gr
from PIL import Image

dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="test")

def get_image_and_label(index):
    # Load the image and label from the dataset
    image = dataset[index]['image']
    label = dataset[index]['label']
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    return image, f"Label: {label}"

# Define the Gradio interface
iface = gr.Interface(
    fn=get_image_and_label,
    inputs=gr.Number(label="Image Index"),
    outputs=[gr.Image(label="Image"), gr.Textbox(label="Label")],
    title="Skin Cancer Dataset Viewer",
    description="Enter an index to view an image with its corresponding label."
)

# Launch the Gradio app
iface.launch(share=True)
