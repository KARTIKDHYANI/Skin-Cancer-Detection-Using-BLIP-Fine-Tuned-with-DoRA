from datasets import load_dataset
from collections import defaultdict
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

# Load dataset
dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="train")

# Medical descriptions for caption augmentation
medical_descriptions = {
    0: ["AK is a scaly, rough patch on the skin caused by sun exposure.", "A precancerous lesion that needs monitoring."],
    1: ["BCC is the most common skin cancer, appearing as pearly bumps.", "A slow-growing cancer that rarely spreads."],
    2: ["BKL is a non-cancerous skin lesion that resembles a mole.", "These lesions are benign but may change in appearance."],
    3: ["Melanoma is an aggressive cancer that develops from pigment cells.", "It often appears as a dark, irregularly shaped mole."],
    4: ["Nevus, or mole, is a common skin growth.", "Most nevi are harmless but require observation."],
    5: ["SCC often presents as a red, scaly patch or ulcer.", "It can metastasize if left untreated."],
    6: ["DF is a firm, brownish growth often found on the legs.", "It is harmless but can be mistaken for other skin conditions."],
    7: ["Vascular lesions are abnormal clusters of blood vessels.", "They can appear as red or purple spots on the skin."]
}

# Dataset class
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["text"] = random.choice(medical_descriptions.get(item["label"], ["Unknown condition."]))
        return encoding

# Collate function
def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        if key != "text":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        else:
            text_inputs = processor.tokenizer(
                [example["text"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch

# Load BLIP-2 processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True
)

# LoRA fine-tuning configuration
config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, bias="none", target_modules=["q_proj", "k_proj"])
model = get_peft_model(model, config)
model.print_trainable_parameters()

# Split dataset (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Prepare data loaders
train_dataset = ImageCaptioningDataset(train_dataset, processor)
val_dataset = ImageCaptioningDataset(val_dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=3, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=3, collate_fn=collate_fn)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Training setup with early stopping
patience = 3
e = 10
best_val_loss = float("inf")
no_improve_epochs = 0
output_dir = "./BLIP_finetuned_dora_r32_final"
print("Starting........................")
for epoch in range(e):
    print(f"Epoch {epoch + 1}/{e}")
    model.train()
    total_train_loss = 0
    
    for batch in train_dataloader:
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device, torch.float16)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"  Training Loss: {avg_train_loss:.4f}")
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device, torch.float16)
            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            total_val_loss += outputs.loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    print(f"  Validation Loss: {avg_val_loss:.4f}")

    # Save checkpoints
    checkpoint_dir = f"{output_dir}/checkpoint_{epoch+1}"
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    print(f"  Checkpoint saved at {checkpoint_dir}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
        print("  Validation loss improved. Saving best model...")
        model.save_pretrained(f"{output_dir}/best_model")
        processor.save_pretrained(f"{output_dir}/best_model")
    else:
        no_improve_epochs += 1
        print(f"  No improvement for {no_improve_epochs} epoch(s).")
    
    if no_improve_epochs >= patience:
        print("  Early stopping triggered. Training stopped.")
        break

# Final model save
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Final model and processor saved at {output_dir}")
