from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
from nltk.translate.bleu_score import sentence_bleu

# Load the model and processor from your saved directory
processor = AutoProcessor.from_pretrained("./BLIP_finetuned_dora_r16/checkpoint_26")
model = Blip2ForConditionalGeneration.from_pretrained("./BLIP_finetuned_dora_r16/checkpoint_26")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
from datasets import load_dataset 

dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="test")
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont

# Function to process and generate caption
def generate_caption(index):
    # Load the image from the dataset (ensure compatibility with PIL.Image)
    image = dataset[index]['image']
    print(dataset[index]['label'])
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to("cuda", torch.float16)
    pixel_values = inputs.pixel_values

    # Generate caption
    generated_ids = model.generate(pixel_values=pixel_values, max_length=25)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Draw the caption on the image
    img = image.convert("RGB")  # Ensure RGB format
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Wrap text to fit within the image width
    max_width = img.width - 20  # Allow for padding
    words = generated_caption.split()
    lines = []
    line = ""

    for word in words:
        test_line = f"{line} {word}" if line else word
        test_width = draw.textbbox((0, 0), test_line, font=font)[2] - draw.textbbox((0, 0), test_line, font=font)[0]
        if test_width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    # Calculate text height and position
    line_height = draw.textbbox((0, 0), "Test", font=font)[3] - draw.textbbox((0, 0), "Test", font=font)[1]
    text_height = line_height * len(lines)
    text_position = (10, img.height - text_height - 10)  # Bottom padding

    # Draw each line of text
    for i, line in enumerate(lines):
        draw.text((10, text_position[1] + i * line_height), line, fill="white", font=font)

    return img

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Number(label="Image Index"),
    outputs=gr.Image(label="Image with Caption"),
    title="Skin Cancer Diagnosis",
    description="Enter an index to view an image with its generated caption."
)

# Launch the Gradio app
iface.launch(share=True)