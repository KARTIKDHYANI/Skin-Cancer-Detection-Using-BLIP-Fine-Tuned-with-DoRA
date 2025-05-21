from datasets import load_dataset
from collections import defaultdict
import random
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

# Load dataset
dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="train")

# Load additional medical descriptions for variation (replace with actual dataset)
medical_descriptions = {
    0: ["AK is a scaly, rough patch on the skin caused by sun exposure.", 
        "A precancerous lesion that needs monitoring.", 
        "Early-stage AK may feel like sandpaper on the skin."],
    1: ["BCC is the most common skin cancer, appearing as pearly bumps.", 
        "A slow-growing cancer that rarely spreads.", 
        "Often caused by long-term UV exposure."],
    2: ["BKL is a non-cancerous skin lesion that resembles a mole.", 
        "These lesions are benign but may change in appearance.", 
        "Harmless, but sometimes mistaken for melanoma."],
    3: ["Melanoma is an aggressive cancer that develops from pigment cells.", 
        "It often appears as a dark, irregularly shaped mole.", 
        "Early detection is crucial for survival."],
    4: ["Nevus, or mole, is a common skin growth.", 
        "Most nevi are harmless but require observation.", 
        "Moles can be congenital or develop over time."],
    5: ["SCC often presents as a red, scaly patch or ulcer.", 
        "It can metastasize if left untreated.", 
        "Common in sun-exposed areas."],
    6: ["DF is a firm, brownish growth often found on the legs.", 
        "It is harmless but can be mistaken for other skin conditions.", 
        "Caused by excess fibrous tissue growth."],
    7: ["Vascular lesions are abnormal clusters of blood vessels.", 
        "They can appear as red or purple spots on the skin.", 
        "Mostly benign, but some may require treatment."]
}

def extract_images_per_class(dataset, num_images_per_class=6):
    """Extracts more images per class for better diversity."""
    class_to_images = defaultdict(list)
    for item in dataset:
        class_to_images[item["label"]].append(item)
    
    new_dataset = []
    for label in range(8):
        images_of_class = class_to_images[label][:num_images_per_class]
        new_dataset.extend(images_of_class)
    
    return new_dataset

# Augment dataset with diverse captions
new_dataset = extract_images_per_class(dataset, num_images_per_class=6)

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Assign a random caption from the medical descriptions to add variability
        encoding["text"] = random.choice(medical_descriptions.get(item["label"], ["Unknown condition."]))

        return encoding

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], 
                padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

# Load BLIP-2 processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True
)

# LoRA configuration with higher rank (r=32) for richer representations
config = LoraConfig(
    r=32,  
    lora_alpha=64,  
    lora_dropout=0.1,  
    bias="none",  
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

# Prepare dataset and dataloader
train_dataset = ImageCaptioningDataset(new_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=collate_fn)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Lower LR for better generalization
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.train()

output_dir = "./BLIP_finetuned_dora_r32_vocab"

# Training loop with mixed precision for efficiency
for epoch in range(3):
    print(f"Epoch {epoch+1}/{3}")
    total_loss = 0
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss

        total_loss += loss.item()
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        if idx % 10 == 0:
            print(f"Step {idx}, Loss: {loss.item()}")

    # Save checkpoint per epoch
    model.save_pretrained(f"{output_dir}/checkpoint_{epoch+1}")
    processor.save_pretrained(f"{output_dir}/checkpoint_{epoch+1}")
    print(f"Checkpoint saved at {output_dir}/checkpoint_{epoch+1}")

# Save final model
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Final model saved at {output_dir}")
