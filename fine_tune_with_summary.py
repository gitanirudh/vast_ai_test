import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"

import time
import datetime
from pathlib import Path
import random
import argparse
import contextlib

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, f1_score

# ----------------------------
# CLI
# ----------------------------
def build_args():
    p = argparse.ArgumentParser("Token classification fine-tuning from an MLM checkpoint")
    p.add_argument("--model-path", type=str, default="runs/microsoft__deberta-v3-large/best",
                   help="Directory with pretrained weights/tokenizer")
    p.add_argument("--data-file", type=str, default="CR_ECSS_dataset.json",
                   help="JSON dataset file with columns: sentence_id, words, labels")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--runs", type=int, default=1, help="How many fine-tuning runs (repeats)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-number", type=int, default=1, help="Run number for tracking")
    p.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    return p.parse_args()

# ----------------------------
# Config
# ----------------------------
args = build_args()

epochs = args.epochs
fine_tuning_runs = args.runs
batch_num = args.batch_size
seed_val = args.seed
model_path = args.model_path
data_file = args.data_file
use_amp = True

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# ----------------------------
# Data
# ----------------------------
dataset = pd.read_json(data_file)

# ----------------------------
# Helpers
# ----------------------------
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round(elapsed))))

def build_optimizer(model, lr=3e-5, wd=0.01):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": wd},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)

def tokenize_and_align_labels(examples, labels, tokenizer, max_len=512, stride=0):
    enc = tokenizer(
        examples,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=max_len,
        stride=stride if stride else 0,
        return_overflowing_tokens=False,
        return_attention_mask=True,
    )

    word_piece_labels = []
    label_all_tokens = True
    for i, label in enumerate(labels):
        word_ids = enc.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        word_piece_labels.append(label_ids)

    enc["labels"] = word_piece_labels
    return enc

def save_run_summary(output_dir, model_name, run_number, seed, training_stats, total_time):
    """Save summary of this run to CSV"""
    # Get best epoch (highest F1 score)
    best_epoch = max(training_stats, key=lambda x: x["F1 score"])
    
    # Create summary row
    summary_row = {
        "model": model_name,
        "run": run_number,
        "seed": seed,
        "epochs_trained": len(training_stats),
        "best_epoch": best_epoch["epoch"],
        "best_f1_score": best_epoch["F1 score"],
        "best_train_loss": best_epoch["Training Loss"],
        "best_val_loss": best_epoch["Valid. Loss"],
        "final_f1_score": training_stats[-1]["F1 score"],
        "final_train_loss": training_stats[-1]["Training Loss"],
        "final_val_loss": training_stats[-1]["Valid. Loss"],
        "total_training_time_seconds": total_time,
        "total_training_time_formatted": format_time(total_time),
    }
    
    # Path for summary CSV
    summary_file = Path(output_dir) / f"{model_name}_run_summary.csv"
    
    # Append to CSV (create if doesn't exist)
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        df = pd.DataFrame([summary_row])
    
    df.to_csv(summary_file, index=False)
    print(f"\nRun summary appended to: {summary_file}")
    
    return summary_file

# ----------------------------
# Reproducibility
# ----------------------------
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_val)
set_seed(seed_val)

torch.backends.cudnn.benchmark = True

# ----------------------------
# Device & AMP
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(f"Device: {device} | GPUs: {n_gpu}")

amp_dtype = torch.float16
if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
    model_name_lower = model_path.lower()
    if "roberta-base" in model_name_lower and "modernbert" not in model_name_lower:
        amp_dtype = torch.float16
        print("Using float16 for RoBERTa-base")
    else:
        amp_dtype = torch.bfloat16
        print(f"Using bfloat16 for {model_path}")

def autocast_ctx(enabled: bool):
    if not (enabled and torch.cuda.is_available()):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)

try:
    from torch.amp import GradScaler as _GradScaler
    scaler = _GradScaler("cuda", enabled=(use_amp and torch.cuda.is_available() and amp_dtype == torch.float16))
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available() and amp_dtype == torch.float16))

# ----------------------------
# Fine-tuning
# ----------------------------
for run_idx in range(fine_tuning_runs):
    print(f"\n==================== Fine Tuning Round {run_idx + 1} ====================")

    cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_type = getattr(cfg, "model_type", "").lower()
    roberta_like = model_type in {"roberta", "xlm-roberta"} or ("modernbert" in model_path.lower())
    print(f"Loaded model_type: {model_type} | roberta_like={roberta_like}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
        add_prefix_space=roberta_like,
    )

    MAX_LEN = 128  # Fixed for stability

    sentence_ids = dataset["sentence_id"].unique()
    sentences = [[w for w in dataset[dataset["sentence_id"] == sid]["words"].values] for sid in sentence_ids]
    labels_list = [[dataset[dataset["sentence_id"] == sid]["labels"].map(lambda x: x).tolist()[i] for i in range(len(dataset[dataset["sentence_id"] == sid]["labels"]))] for sid in sentence_ids]
    
    tag_vals = dataset["labels"].unique()
    tag2idx = {tag: i for i, tag in enumerate(tag_vals)}
    tag2name = {v: k for k, v in tag2idx.items()}
    tag2name[-100] = "None"
    labels_list = [[tag2idx[lbl] for lbl in dataset[dataset["sentence_id"] == sid]["labels"].values] for sid in sentence_ids]

    encoded = tokenize_and_align_labels(sentences, labels_list, tokenizer, max_len=MAX_LEN, stride=0)
    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]
    labels = encoded["labels"]

    tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = train_test_split(
        input_ids, labels, attention_masks, random_state=seed_val, test_size=0.213, shuffle=True
    )

    tr_inputs = torch.as_tensor(tr_inputs)
    val_inputs = torch.as_tensor(val_inputs)
    tr_tags = torch.as_tensor(tr_tags)
    val_tags = torch.as_tensor(val_tags)
    tr_masks = torch.as_tensor(tr_masks)
    val_masks = torch.as_tensor(val_masks)

    num_workers = max(1, os.cpu_count() // 4)
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)

    train_dataloader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=batch_num,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_dataloader = DataLoader(
        valid_data,
        sampler=SequentialSampler(valid_data),
        batch_size=batch_num,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    id2label = {i: t for t, i in tag2idx.items()}
    label2id = {t: i for t, i in tag2idx.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=len(tag2idx),
        id2label=id2label,
        label2id=label2id,
    ).to(device)
    
    # Re-initialize classifier
    if hasattr(model, 'classifier'):
        torch.nn.init.normal_(model.classifier.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(model.classifier.bias)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer = build_optimizer(model, lr=3e-5, wd=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    training_stats = []
    train_loss_hist = []
    total_t0 = time.time()

    for epoch_i in range(epochs):
        print(f"\n======== Epoch {epoch_i + 1} / {epochs} ========")
        print("Training...")
        t0 = time.time()
        total_train_loss = 0.0

        model.train()
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device, non_blocking=True)
            b_input_mask = batch[1].to(device, non_blocking=True)
            b_labels = batch[2].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(use_amp):
                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    b_labels.view(-1),
                    ignore_index=-100,
                )

            if hasattr(scaler, "is_enabled") and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_train_loss += float(loss.detach())
            scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_dataloader))
        training_time = format_time(time.time() - t0)
        print(f"\n  Average training loss: {avg_train_loss:.4f}")
        print(f"  Training epoch took: {training_time}")
        train_loss_hist.append(avg_train_loss)

        # Validation
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()

        total_eval_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in valid_dataloader:
                b_input_ids = batch[0].to(device, non_blocking=True)
                b_input_mask = batch[1].to(device, non_blocking=True)
                b_labels = batch[2].to(device, non_blocking=True)

                outputs = model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs.logits
                
                vloss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    b_labels.view(-1),
                    ignore_index=-100,
                )

                total_eval_loss += float(vloss.detach())

                preds = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                preds = preds.detach().cpu().numpy()
                label_ids = b_labels.detach().cpu().numpy()
                input_mask = b_input_mask.detach().cpu().numpy()

                for i_m, m in enumerate(input_mask):
                    t1, t2 = [], []
                    for j, flag in enumerate(m):
                        if flag:
                            if tag2name.get(label_ids[i_m][j], "None") != "None":
                                t1.append(tag2name[label_ids[i_m][j]])
                                t2.append(tag2name[preds[i_m][j]])
                        else:
                            break
                    y_true.append(t1)
                    y_pred.append(t2)

        y_true_words = [w for sent in y_true for w in sent]
        y_pred_words = [w for sent in y_pred for w in sent]

        labels_for_scores = [lab for lab in set(y_true_words) if lab != "O"]
        if labels_for_scores:
            pr, rc, f1s, support = precision_recall_fscore_support(
                y_true_words, y_pred_words, labels=labels_for_scores, zero_division=0
            )
            f1_scores = {lab: f1s[i] for i, lab in enumerate(labels_for_scores)}
            examples = {lab: support[i] for i, lab in enumerate(labels_for_scores)}
            f1_scores["weighted"] = f1_score(
                y_true_words, y_pred_words, average="weighted", labels=labels_for_scores, zero_division=0
            )
            examples["sum"] = int(np.sum([examples[k] for k in examples.keys()]))
        else:
            f1_scores = {"weighted": 0.0}
            examples = {"sum": 0}

        avg_val_loss = total_eval_loss / max(1, len(valid_dataloader))
        print(f"  F1_score (weighted): {f1_scores['weighted']:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        validation_time = format_time(time.time() - t0)
        print(f"  Validation took: {validation_time}")

        training_stats.append({
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "F1 score": f1_scores["weighted"],
            "examples_sum": examples["sum"],
            "Label_F1_scores": f1_scores,
            "examples": examples,
            "Training Time": training_time,
            "Validation Time": validation_time,
        })
        
        with open("Train_results.json", "w+", encoding="utf-8") as file:
            pd.DataFrame(training_stats).to_json(file, orient="records", force_ascii=False)

    print("\nTraining complete!")
    total_training_time = time.time() - total_t0
    print("Total training took {:} (h:mm:ss)".format(format_time(total_training_time)))
    
    # NEW: Save run summary
    # Extract model name from path (e.g., "runs/roberta-base/best" -> "roberta-base")
    model_name = Path(model_path).parent.name
    save_run_summary(
        output_dir=args.output_dir,
        model_name=model_name,
        run_number=args.run_number,
        seed=seed_val,
        training_stats=training_stats,
        total_time=total_training_time
    )

    # Existing: Save classification report
    y_true_words = [w for sent in y_true for w in sent]
    y_pred_words = [w for sent in y_pred for w in sent]
    report_dict = classification_report(
        y_true_words,
        y_pred_words,
        digits=3,
        labels=[label for label in set(y_true_words) if label != "O"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    base = Path("report.csv")
    filename = base
    idx = 1
    while filename.exists():
        filename = base.with_name(f"{base.stem}_{idx}{base.suffix}")
        idx += 1
    report_df.to_csv(filename, encoding="utf-8")
    print(f"Classification report saved to: {filename}")
