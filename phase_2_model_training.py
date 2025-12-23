# Phase 2: Model Selection, Training, and Fine-Tuning
# YouTube Summarizer - NLP Project

"""
This notebook covers Phase 2 of the NLP project:
- Model Selection: DistilBART and T5-small
- Data Preparation: YouTube transcripts + CNN/DailyMail
- Fine-tuning both models
- Training visualization and checkpointing

Environment: Google Colab (Free GPU) or Kaggle
Author: Team YouTube Summarizer
Date: December 2024
"""

# ============================================================
# SECTION 1: Environment Setup
# ============================================================

# Install required packages (uncomment for Colab/Kaggle)
# !pip install -q transformers datasets accelerate evaluate rouge_score sentencepiece
# !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

# Hugging Face libraries
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict, load_dataset
import evaluate

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================
# SECTION 2: Configuration
# ============================================================

class Config:
    """Training configuration for both models."""
    
    # Data settings
    MAX_INPUT_LENGTH = 1024
    MAX_TARGET_LENGTH = 150
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    TEST_SIZE = 0.1
    
    # DistilBART settings
    DISTILBART_MODEL = "sshleifer/distilbart-cnn-12-6"
    DISTILBART_EPOCHS = 3
    DISTILBART_BATCH_SIZE = 4
    DISTILBART_GRAD_ACCUM = 4  # Effective batch = 16
    DISTILBART_LR = 2e-5
    
    # T5-small settings
    T5_MODEL = "t5-small"
    T5_EPOCHS = 5
    T5_BATCH_SIZE = 8
    T5_GRAD_ACCUM = 2  # Effective batch = 16
    T5_LR = 3e-4
    T5_PREFIX = "summarize: "
    
    # Training settings
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    LOGGING_STEPS = 100
    
    # Paths
    OUTPUT_DIR = "./models"
    CHECKPOINT_DIR = "./checkpoints"
    
config = Config()

# ============================================================
# SECTION 3: Data Loading and Preparation
# ============================================================

def load_cnn_dailymail(num_samples: int = 5000) -> DatasetDict:
    """
    Load CNN/DailyMail dataset for summarization.
    
    Args:
        num_samples: Number of samples to use
        
    Returns:
        DatasetDict with train/validation/test splits
    """
    print("Loading CNN/DailyMail dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    
    # Sample subset for efficiency
    train_size = int(num_samples * config.TRAIN_SIZE)
    val_size = int(num_samples * config.VAL_SIZE)
    test_size = int(num_samples * config.TEST_SIZE)
    
    train_data = dataset["train"].shuffle(seed=42).select(range(train_size))
    val_data = dataset["validation"].shuffle(seed=42).select(range(val_size))
    test_data = dataset["test"].shuffle(seed=42).select(range(test_size))
    
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
    
    return DatasetDict({
        "train": train_data,
        "validation": val_data,
        "test": test_data
    })


def load_youtube_transcripts(data_path: str = None) -> Dataset:
    """
    Load custom YouTube transcript data if available.
    
    Format expected: JSON with 'transcript' and 'summary' fields
    """
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return Dataset.from_dict({
            "article": [d["transcript"] for d in data],
            "highlights": [d["summary"] for d in data]
        })
    
    print("No YouTube transcript data found. Using CNN/DailyMail only.")
    return None


def preprocess_for_distilbart(examples: Dict, tokenizer) -> Dict:
    """Preprocess data for DistilBART fine-tuning."""
    
    inputs = examples["article"]
    targets = examples["highlights"]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=config.MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=config.MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_for_t5(examples: Dict, tokenizer) -> Dict:
    """Preprocess data for T5 fine-tuning with prefix."""
    
    # Add T5 prefix
    inputs = [config.T5_PREFIX + doc for doc in examples["article"]]
    targets = examples["highlights"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=config.MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=config.MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Load data
dataset = load_cnn_dailymail(num_samples=5000)

# Display sample
print("\n--- Sample Article ---")
print(dataset["train"][0]["article"][:500] + "...")
print("\n--- Sample Summary ---")
print(dataset["train"][0]["highlights"])

# ============================================================
# SECTION 4: Model 1 - DistilBART Fine-Tuning
# ============================================================

print("\n" + "="*60)
print("MODEL 1: DistilBART Fine-Tuning")
print("="*60)

# Load tokenizer and model
distilbart_tokenizer = AutoTokenizer.from_pretrained(config.DISTILBART_MODEL)
distilbart_model = AutoModelForSeq2SeqLM.from_pretrained(config.DISTILBART_MODEL)

print(f"Model parameters: {distilbart_model.num_parameters():,}")

# Preprocess dataset
distilbart_dataset = dataset.map(
    lambda x: preprocess_for_distilbart(x, distilbart_tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing for DistilBART"
)

# Data collator
distilbart_collator = DataCollatorForSeq2Seq(
    tokenizer=distilbart_tokenizer,
    model=distilbart_model,
    padding=True
)

# Evaluation metric
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """Compute ROUGE scores for evaluation."""
    predictions, labels = eval_pred
    
    # Decode predictions
    decoded_preds = distilbart_tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, distilbart_tokenizer.pad_token_id)
    decoded_labels = distilbart_tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )
    
    # Compute ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {k: round(v * 100, 2) for k, v in result.items()}

# Training arguments
distilbart_args = Seq2SeqTrainingArguments(
    output_dir=f"{config.CHECKPOINT_DIR}/distilbart",
    num_train_epochs=config.DISTILBART_EPOCHS,
    per_device_train_batch_size=config.DISTILBART_BATCH_SIZE,
    per_device_eval_batch_size=config.DISTILBART_BATCH_SIZE,
    gradient_accumulation_steps=config.DISTILBART_GRAD_ACCUM,
    learning_rate=config.DISTILBART_LR,
    warmup_steps=config.WARMUP_STEPS,
    weight_decay=config.WEIGHT_DECAY,
    evaluation_strategy="steps",
    eval_steps=config.EVAL_STEPS,
    save_strategy="steps",
    save_steps=config.SAVE_STEPS,
    logging_steps=config.LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=config.MAX_TARGET_LENGTH,
    fp16=torch.cuda.is_available(),  # Mixed precision on GPU
    report_to="none",  # Disable wandb
)

# Create trainer
distilbart_trainer = Seq2SeqTrainer(
    model=distilbart_model,
    args=distilbart_args,
    train_dataset=distilbart_dataset["train"],
    eval_dataset=distilbart_dataset["validation"],
    tokenizer=distilbart_tokenizer,
    data_collator=distilbart_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train model
print("\nStarting DistilBART training...")
distilbart_results = distilbart_trainer.train()

# Save model
distilbart_model.save_pretrained(f"{config.OUTPUT_DIR}/distilbart-youtube-summarizer")
distilbart_tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/distilbart-youtube-summarizer")

print(f"\nDistilBART training complete!")
print(f"Training time: {distilbart_results.metrics['train_runtime']:.2f}s")

# ============================================================
# SECTION 5: Model 2 - T5-small Fine-Tuning
# ============================================================

print("\n" + "="*60)
print("MODEL 2: T5-small Fine-Tuning")
print("="*60)

# Load tokenizer and model
t5_tokenizer = AutoTokenizer.from_pretrained(config.T5_MODEL)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(config.T5_MODEL)

print(f"Model parameters: {t5_model.num_parameters():,}")

# Preprocess dataset
t5_dataset = dataset.map(
    lambda x: preprocess_for_t5(x, t5_tokenizer),
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing for T5"
)

# Data collator
t5_collator = DataCollatorForSeq2Seq(
    tokenizer=t5_tokenizer,
    model=t5_model,
    padding=True
)

# Training arguments
t5_args = Seq2SeqTrainingArguments(
    output_dir=f"{config.CHECKPOINT_DIR}/t5-small",
    num_train_epochs=config.T5_EPOCHS,
    per_device_train_batch_size=config.T5_BATCH_SIZE,
    per_device_eval_batch_size=config.T5_BATCH_SIZE,
    gradient_accumulation_steps=config.T5_GRAD_ACCUM,
    learning_rate=config.T5_LR,
    warmup_steps=config.WARMUP_STEPS,
    weight_decay=config.WEIGHT_DECAY,
    evaluation_strategy="steps",
    eval_steps=config.EVAL_STEPS,
    save_strategy="steps",
    save_steps=config.SAVE_STEPS,
    logging_steps=config.LOGGING_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=config.MAX_TARGET_LENGTH,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# Compute metrics function for T5
def compute_metrics_t5(eval_pred):
    predictions, labels = eval_pred
    
    decoded_preds = t5_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, t5_tokenizer.pad_token_id)
    decoded_labels = t5_tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# Create trainer
t5_trainer = Seq2SeqTrainer(
    model=t5_model,
    args=t5_args,
    train_dataset=t5_dataset["train"],
    eval_dataset=t5_dataset["validation"],
    tokenizer=t5_tokenizer,
    data_collator=t5_collator,
    compute_metrics=compute_metrics_t5,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train model
print("\nStarting T5-small training...")
t5_results = t5_trainer.train()

# Save model
t5_model.save_pretrained(f"{config.OUTPUT_DIR}/t5-youtube-summarizer")
t5_tokenizer.save_pretrained(f"{config.OUTPUT_DIR}/t5-youtube-summarizer")

print(f"\nT5-small training complete!")
print(f"Training time: {t5_results.metrics['train_runtime']:.2f}s")

# ============================================================
# SECTION 6: Training Visualization
# ============================================================

def plot_training_curves(trainer, model_name: str):
    """Plot training loss and evaluation metrics."""
    
    history = trainer.state.log_history
    
    # Extract metrics
    train_loss = [x["loss"] for x in history if "loss" in x]
    eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]
    eval_rouge1 = [x["eval_rouge1"] for x in history if "eval_rouge1" in x]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Training loss
    axes[0].plot(train_loss, label="Train Loss", color="blue")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Evaluation loss
    axes[1].plot(eval_loss, label="Eval Loss", color="orange")
    axes[1].set_xlabel("Evaluation Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{model_name} - Evaluation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # ROUGE-1 score
    axes[2].plot(eval_rouge1, label="ROUGE-1", color="green")
    axes[2].set_xlabel("Evaluation Step")
    axes[2].set_ylabel("ROUGE-1 Score")
    axes[2].set_title(f"{model_name} - ROUGE-1 Progress")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/{model_name.lower()}_training_curves.png", dpi=150)
    plt.show()
    
    return fig

# Plot training curves
print("\n" + "="*60)
print("TRAINING VISUALIZATION")
print("="*60)

plot_training_curves(distilbart_trainer, "DistilBART")
plot_training_curves(t5_trainer, "T5-small")

# ============================================================
# SECTION 7: Sample Inference Comparison
# ============================================================

print("\n" + "="*60)
print("SAMPLE INFERENCE COMPARISON")
print("="*60)

def generate_summary(model, tokenizer, text: str, prefix: str = "") -> str:
    """Generate summary using fine-tuned model."""
    
    input_text = prefix + text if prefix else text
    
    inputs = tokenizer(
        input_text,
        max_length=config.MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=config.MAX_TARGET_LENGTH,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test on sample
sample_text = dataset["test"][0]["article"]
reference_summary = dataset["test"][0]["highlights"]

print("--- Original Text (truncated) ---")
print(sample_text[:800] + "...")
print("\n--- Reference Summary ---")
print(reference_summary)

print("\n--- DistilBART Summary ---")
distilbart_summary = generate_summary(distilbart_model, distilbart_tokenizer, sample_text)
print(distilbart_summary)

print("\n--- T5-small Summary ---")
t5_summary = generate_summary(t5_model, t5_tokenizer, sample_text, prefix=config.T5_PREFIX)
print(t5_summary)

# ============================================================
# SECTION 8: Model Summary and Export
# ============================================================

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)

summary_data = {
    "Model": ["DistilBART", "T5-small"],
    "Parameters": [
        f"{distilbart_model.num_parameters():,}",
        f"{t5_model.num_parameters():,}"
    ],
    "Epochs": [config.DISTILBART_EPOCHS, config.T5_EPOCHS],
    "Training Time (s)": [
        round(distilbart_results.metrics["train_runtime"], 2),
        round(t5_results.metrics["train_runtime"], 2)
    ],
    "Final Train Loss": [
        round(distilbart_results.metrics.get("train_loss", 0), 4),
        round(t5_results.metrics.get("train_loss", 0), 4)
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary
summary_df.to_csv(f"{config.OUTPUT_DIR}/training_summary.csv", index=False)

print(f"\n✅ Models saved to: {config.OUTPUT_DIR}/")
print("  - distilbart-youtube-summarizer/")
print("  - t5-youtube-summarizer/")
print("\n✅ Training curves saved")
print("✅ Training summary exported to CSV")

# ============================================================
# SECTION 9: Upload to Hugging Face Hub (Optional)
# ============================================================

"""
# Uncomment to push models to Hugging Face Hub

from huggingface_hub import login
login()  # Enter your HF token

# Push DistilBART
distilbart_model.push_to_hub("your-username/distilbart-youtube-summarizer")
distilbart_tokenizer.push_to_hub("your-username/distilbart-youtube-summarizer")

# Push T5
t5_model.push_to_hub("your-username/t5-youtube-summarizer")
t5_tokenizer.push_to_hub("your-username/t5-youtube-summarizer")
"""

print("\n" + "="*60)
print("PHASE 2 COMPLETE ✅")
print("="*60)
print("""
Next Steps:
1. Run Phase 3 evaluation notebook
2. Compare models on full test set
3. Integrate best model into backend
""")
