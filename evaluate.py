import os
import json
from nltk.translate.bleu_score import sentence_bleu
from transformers import Blip2ForConditionalGeneration, AutoProcessor
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import torch
from datasets import load_dataset 
validation_dataset = load_dataset("Pranavkpba2000/skin_cancer_small_dataset", split="test")
# Define label to prompt mapping for evaluation
label_to_prompt = {
    0: "The detected disease is Actinic Keratosis (AK), a precancerous condition characterized by scaly, crusty patches of skin. It can potentially develop into skin cancer if not treated.",
    1: "The detected disease is Basal Cell Carcinoma (BCC), a common form of skin cancer that typically appears as a pearly or waxy bump. It is slow-growing and usually doesn't spread.",
    2: "The detected disease is Benign Keratosis (BKL), a non-cancerous skin growth that often appears as a wart or mole. These are usually harmless but may need monitoring.",
    3: "The detected disease is Melanoma (MEL), a serious and aggressive form of skin cancer that often presents as a mole with irregular edges or different colors. It requires immediate attention.",
    4: "The detected disease is Nevus (NV), commonly known as a mole, which can vary in color and size. Most are harmless, but changes in size, shape, or color may require evaluation.",
    5: "The detected disease is Squamous Cell Carcinoma (SCC), a type of skin cancer that often appears as a firm, red nodule or scaly patch. It can spread if not treated early.",
    6: "The detected disease is Dermatofibroma (DF), a benign growth on the skin that is typically brown or tan. These are harmless and generally do not require treatment.",
    7: "The detected disease is Vascular Lesions (VASC), abnormal blood vessel growths that can appear as red or purple spots on the skin. They are generally harmless but may require treatment for cosmetic reasons."
}

# Initialize evaluators
rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
cider_evaluator = Cider()
meteor_evaluator = Meteor()

# Function to evaluate BLEU, ROUGE, CIDEr, and METEOR scores on validation set
def evaluate_metrics(model, processor, validation_dataset):
    print("Evaluating")
    model.eval()
    total_bleu_score = 0
    total_rouge_score = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    total_cider_score = 0
    total_meteor_score = 0
    count = 0

    for item in validation_dataset:
        image = item["image"]
        label = item["label"]
        print("Generating caption")
        # Generate caption
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_length=80)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(generated_text)
        # Reference prompt
        reference_text = label_to_prompt[label]
        print(reference_text)
        
        print("_________________________________________________")

        # Calculate BLEU score
        reference_tokens = reference_text.split()
        generated_tokens = generated_text.split()
        bleu_score = sentence_bleu([reference_tokens], generated_tokens)
        total_bleu_score += bleu_score

        # Calculate ROUGE scores
        rouge_scores = rouge_scorer_instance.score(reference_text, generated_text)
        for key in total_rouge_score:
            total_rouge_score[key] += rouge_scores[key].fmeasure

        # Calculate CIDEr score (CIDEr expects a list of dicts)
        cider_score, _ = cider_evaluator.compute_score(
            {0: [reference_text]}, {0: [generated_text]}
        )
        total_cider_score += cider_score

        # Calculate METEOR score
        meteor_score, _ = meteor_evaluator.compute_score(
            {0: [reference_text]}, {0: [generated_text]}
        )
        total_meteor_score += meteor_score

        count += 1

    # Calculate average scores
    average_bleu_score = total_bleu_score / count
    average_rouge_score = {k: v / count for k, v in total_rouge_score.items()}
    average_cider_score = total_cider_score / count
    average_meteor_score = total_meteor_score / count

    return {
        "average_bleu_score": average_bleu_score,
        "average_rouge_score": average_rouge_score,
        "average_cider_score": average_cider_score,
        "average_meteor_score": average_meteor_score
    }

# Path to the JSON file for storing results
result_file_path = "best_scores.json"

# Load existing results if the JSON file exists
if os.path.exists(result_file_path):
    with open(result_file_path, "r") as f:
        best_scores = json.load(f)
else:
    # Initialize the best scores dictionary
    best_scores = {
        "bleu": {"score": 0, "checkpoint": None},
        "rouge1": {"score": 0, "checkpoint": None},
        "rouge2": {"score": 0, "checkpoint": None},
        "rougeL": {"score": 0, "checkpoint": None},
        "cider": {"score": 0, "checkpoint": None},
        "meteor": {"score": 0, "checkpoint": None}
    }

# Loop through checkpoints and find the best one for each metric
for checkpoint in range(1, 71):
    print("Starting checkpoint ",checkpoint)
    # Load model and processor for the checkpoint
    processor = AutoProcessor.from_pretrained(f"./BLIP_finetuned_dora_r16/checkpoint_{checkpoint}")
    model = Blip2ForConditionalGeneration.from_pretrained(f"./BLIP_finetuned_dora_r16/checkpoint_{checkpoint}")

    # Evaluate metrics
    metrics = evaluate_metrics(model, processor, validation_dataset)

    # Update best scores for each metric if the current checkpoint's scores are higher
    if metrics["average_bleu_score"] > best_scores["bleu"]["score"]:
        best_scores["bleu"] = {"score": metrics["average_bleu_score"], "checkpoint": checkpoint}
    
    for rouge_key in ["rouge1", "rouge2", "rougeL"]:
        if metrics["average_rouge_score"][rouge_key] > best_scores[rouge_key]["score"]:
            best_scores[rouge_key] = {"score": metrics["average_rouge_score"][rouge_key], "checkpoint": checkpoint}

    if metrics["average_cider_score"] > best_scores["cider"]["score"]:
        best_scores["cider"] = {"score": metrics["average_cider_score"], "checkpoint": checkpoint}

    if metrics["average_meteor_score"] > best_scores["meteor"]["score"]:
        best_scores["meteor"] = {"score": metrics["average_meteor_score"], "checkpoint": checkpoint}

    # Write the updated best scores to the JSON file after each checkpoint
    with open(result_file_path, "w") as f:
        json.dump(best_scores, f, indent=4)

# Print best scores for each metric
for metric, data in best_scores.items():
    print(f"Best {metric.upper()} Score: {data['score']:.4f} at checkpoint {data['checkpoint']}")
