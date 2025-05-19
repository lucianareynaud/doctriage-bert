# src/train.py
import argparse
import os
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig
import torch
import evaluate
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on report vs. regulation classification with LoRA + 4-bit quantization")
    parser.add_argument("--model", default="distilbert-base-uncased", help="Hugging Face model ID")
    parser.add_argument("--output_dir", default="outputs", help="Where to save the final model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_data", default="data/train_valid", help="Path to training data")
    parser.add_argument("--test_data", default="data/test", help="Path to test data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available (default is CPU mode)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Use gradient checkpointing to save memory")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation")
    return parser.parse_args()


def preprocess(examples, tokenizer):
    """Tokenize text and map domain strings to numeric labels."""
    enc = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    enc["labels"] = [0 if d == "reports" else 1 for d in examples["domain"]]
    return enc


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
        "precision": precision.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    }


def main():
    args = parse_args()

    # 1) Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Check if MPS (Apple Silicon GPU) is available
    mps_available = hasattr(torch, 'mps') and torch.mps.is_available()
    
    # Determine device map strategy
    if args.gpu and torch.cuda.is_available():
        print("Using GPU with 4-bit quantization")
        # Configure 4-bit quantization for CUDA GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model, 
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto"
        )
    elif args.gpu and mps_available:
        print("Using Mac GPU (MPS) - Note: Limited acceleration without quantization")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model, 
            num_labels=2
        ).to("mps")
    else:
        print("Using CPU-only mode")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            args.model, 
            num_labels=2
        )
    
    # Enable gradient checkpointing to save memory
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    # 2) Apply LoRA
    print("Applying LoRA adapter")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        # For DistilBERT, use these target modules
        target_modules=["q_lin", "k_lin", "v_lin", "out_lin"] if "distilbert" in args.model.lower() else ["q_proj", "k_proj", "v_proj", "out_proj"]
    )
    model = get_peft_model(base_model, peft_config)

    # 3) Load train/validation and test splits
    # Prepare train and validation datasets (train_valid directory)
    train_valid_files = [os.path.join(args.train_data, f) for f in os.listdir(args.train_data) 
                         if f.endswith('.parquet')]
    raw_train_ds = load_dataset("parquet", data_files=train_valid_files)["train"]
    
    # Split into train and validation
    train_val_split = raw_train_ds.train_test_split(test_size=0.2, seed=args.seed)
    
    # Prepare test dataset (test directory)
    test_files = [os.path.join(args.test_data, f) for f in os.listdir(args.test_data) 
                 if f.endswith('.parquet')]
    if test_files:
        test_ds = load_dataset("parquet", data_files=test_files)["train"]
    else:
        # If no test files found, use a portion of the validation set
        test_ds = train_val_split["test"].select(range(len(train_val_split["test"]) // 2))
        train_val_split["test"] = train_val_split["test"].select(
            range(len(train_val_split["test"]) // 2, len(train_val_split["test"]))
        )
    
    # Create a unified dataset dictionary
    ds = DatasetDict({
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": test_ds
    })
    
    # Apply tokenization
    tokenized_ds = {
        split: ds[split].map(
            lambda x: preprocess(x, tokenizer),
            batched=True,
            remove_columns=ds[split].column_names
        )
        for split in ds
    }

    # 4) Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        remove_unused_columns=True,
        # Optimize memory usage
        fp16=torch.cuda.is_available() and args.gpu,
        bf16=False,
        optim="adamw_torch",  # Memory-efficient optimizer
        seed=args.seed,
        report_to="none",
        # Only use GPU if explicitly requested
        no_cuda=not args.gpu,
        # Enable gradient checkpointing (save memory during backprop)
        gradient_checkpointing=args.gradient_checkpointing,
        # Efficient caching
        dataloader_num_workers=0,  # Better for macOS
        dataloader_pin_memory=False  # Better for limited RAM
    )

    # 5) Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6) Train & save
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 7) Final evaluation on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(tokenized_ds["test"])
    print(metrics)


if __name__ == "__main__":
    main()
