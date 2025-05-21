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
import logging
import time
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from sklearn.model_selection import KFold
import random


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
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    return parser.parse_args()


def setup_augmenters():
    """Initialize text augmentation models."""
    # Synonym augmentation
    syn_aug = naw.SynonymAug(aug_src='wordnet')
    
    # Back translation augmentation
    back_trans_aug = nas.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de',
        to_model_name='facebook/wmt19-de-en'
    )
    
    return syn_aug, back_trans_aug


def augment_text(text, syn_aug, back_trans_aug, num_augmentations=2):
    """Apply multiple augmentation techniques to a text."""
    augmented_texts = []
    
    # Original text
    augmented_texts.append(text)
    
    # Synonym replacement
    for _ in range(num_augmentations):
        try:
            aug_text = syn_aug.augment(text)[0]
            augmented_texts.append(aug_text)
        except Exception as e:
            logging.warning(f"Synonym augmentation failed: {e}")
    
    # Back translation
    try:
        aug_text = back_trans_aug.augment(text)[0]
        augmented_texts.append(aug_text)
    except Exception as e:
        logging.warning(f"Back translation augmentation failed: {e}")
    
    return augmented_texts


def preprocess(examples, tokenizer, augment=False, syn_aug=None, back_trans_aug=None):
    """Tokenize text and map domain strings to numeric labels."""
    if augment and syn_aug and back_trans_aug:
        # Augment texts for small dataset
        augmented_texts = []
        augmented_labels = []
        
        for text, domain in zip(examples["text"], examples["domain"]):
            aug_texts = augment_text(text, syn_aug, back_trans_aug)
            augmented_texts.extend(aug_texts)
            augmented_labels.extend([domain] * len(aug_texts))
        
        texts = augmented_texts
        labels = [0 if d == "reports" else 1 for d in augmented_labels]
    else:
        texts = examples["text"]
        labels = [0 if d == "reports" else 1 for d in examples["domain"]]
    
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
    )
    enc["labels"] = labels
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

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize augmenters if needed
    syn_aug, back_trans_aug = None, None
    if args.augment:
        logger.info("Initializing text augmenters...")
        syn_aug, back_trans_aug = setup_augmenters()
    
    # 1) Load tokenizer & base model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Check if MPS (Apple Silicon GPU) is available
    mps_available = hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available()
    
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
    train_valid_files = [os.path.join(args.train_data, f) for f in os.listdir(args.train_data) 
                         if f.endswith('.parquet')]
    raw_train_ds = load_dataset("parquet", data_files=train_valid_files)["train"]
    
    # For small datasets, use k-fold cross validation
    if len(raw_train_ds) < 100:  # Small dataset threshold
        logger.info("Using k-fold cross validation for small dataset")
        
        # Determine appropriate number of folds based on dataset size
        # Ensure we have at least 1 sample per fold
        n_samples = len(raw_train_ds)
        max_folds = min(5, n_samples)
        
        if n_samples <= 1:
            logger.warning(f"Dataset too small for cross-validation (only {n_samples} samples). Using simple train/test split.")
            # For extremely small datasets, just use the data directly
            tokenized_ds = raw_train_ds.map(
                lambda x: preprocess(x, tokenizer, augment=args.augment, syn_aug=syn_aug, back_trans_aug=back_trans_aug),
                batched=True,
                remove_columns=raw_train_ds.column_names
            )
            
            # Training arguments for tiny dataset
            training_args = TrainingArguments(
                output_dir=args.output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=1,  # Tiny batch size
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                learning_rate=args.lr,
                logging_steps=1,
                evaluation_strategy="no",  # No evaluation with just 1 sample
                save_strategy="epoch",
                push_to_hub=False,
                remove_unused_columns=True,
                fp16=torch.cuda.is_available() and args.gpu,
                bf16=False,
                optim="adamw_torch",
                seed=args.seed,
                report_to="none",
                no_cuda=not args.gpu,
                gradient_checkpointing=args.gradient_checkpointing,
                dataloader_num_workers=0,
                dataloader_pin_memory=False
            )
            
            # Trainer setup for tiny dataset
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_ds,
                tokenizer=tokenizer,
            )
            
            # Train
            trainer.train()
            trainer.save_model(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            logger.warning("Training completed with extremely small dataset. Model may not generalize well.")
            
        else:
            # Use k-fold cross validation with appropriate number of folds
            logger.info(f"Using {max_folds}-fold cross validation for dataset with {n_samples} samples")
            kf = KFold(n_splits=max_folds, shuffle=True, random_state=args.seed)
            
            # Store metrics for each fold
            fold_metrics = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(raw_train_ds)):
                logger.info(f"Training fold {fold + 1}/{max_folds}")
                
                # Split data for this fold
                train_data = raw_train_ds.select(train_idx)
                val_data = raw_train_ds.select(val_idx)
                
                # Apply tokenization with augmentation
                tokenized_train = train_data.map(
                    lambda x: preprocess(x, tokenizer, augment=True, syn_aug=syn_aug, back_trans_aug=back_trans_aug),
                    batched=True,
                    remove_columns=train_data.column_names
                )
                
                tokenized_val = val_data.map(
                    lambda x: preprocess(x, tokenizer, augment=False),
                    batched=True,
                    remove_columns=val_data.column_names
                )
                
                # Training arguments for this fold
                training_args = TrainingArguments(
                    output_dir=os.path.join(args.output_dir, f"fold_{fold}"),
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
                    fp16=torch.cuda.is_available() and args.gpu,
                    bf16=False,
                    optim="adamw_torch",
                    seed=args.seed,
                    report_to="none",
                    no_cuda=not args.gpu,
                    gradient_checkpointing=args.gradient_checkpointing,
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False
                )
                
                # Trainer setup for this fold
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_val,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics,
                )
                
                # Train and evaluate
                trainer.train()
                metrics = trainer.evaluate()
                fold_metrics.append(metrics)
                
                logger.info(f"Fold {fold + 1} metrics: {metrics}")
            
            # Calculate and log average metrics
            avg_metrics = {
                metric: sum(fold[metric] for fold in fold_metrics) / len(fold_metrics)
                for metric in fold_metrics[0].keys()
            }
            logger.info(f"Average metrics across folds: {avg_metrics}")
        
    else:
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
                lambda x: preprocess(x, tokenizer, augment=False),
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

        # 6.1) Garantir artefatos essenciais com até 3 tentativas
        logger = logging.getLogger("doctriage-artifacts")
        essential_files = [
            (os.path.join(args.output_dir, "config.json"), lambda: model.config.save_pretrained(args.output_dir)),
            (os.path.join(args.output_dir, "pytorch_model.bin"), lambda: model.save_pretrained(args.output_dir)),
            (os.path.join(args.output_dir, "tokenizer.json"), lambda: tokenizer.save_pretrained(args.output_dir)),
        ]
        failed = []
        for fpath, save_fn in essential_files:
            if not os.path.isfile(fpath):
                for attempt in range(3):
                    try:
                        save_fn()
                        if os.path.isfile(fpath):
                            logger.info(f"Generated missing artifact: {fpath} (attempt {attempt+1})")
                            break
                        else:
                            logger.warning(f"Tried to generate {fpath}, but file still missing (attempt {attempt+1}).")
                    except Exception as e:
                        logger.warning(f"Could not generate {fpath} (attempt {attempt+1}): {e}")
                    time.sleep(1)
                if not os.path.isfile(fpath):
                    failed.append(fpath)
        if failed:
            raise RuntimeError(f"Failed to generate essential model files after 3 attempts: {failed}. "
                               f"Check permissões, espaço em disco e integridade do modelo/tokenizer.")

    # 7) Final evaluation on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(tokenized_ds["test"])
    print(metrics)


if __name__ == "__main__":
    main()
