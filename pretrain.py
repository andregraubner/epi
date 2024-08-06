import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers import AutoConfig, TrainingArguments, Trainer
from data import MethylomeDataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import einops
from scipy.ndimage import gaussian_filter1d
from torch.cuda.amp import autocast
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import confusion_matrix, accuracy_score

from transformers import AutoModelForMaskedLM, AutoModel, AutoTokenizer
import wandb

from model import CaduceusForMaskedLM, CaduceusConfig

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

device = "cuda"

wandb.init(
    project="methylome",
)

config = CaduceusConfig(
    d_model = 512,
    n_layer = 8,
    vocab_size = 8,
    pad_token_id = 0,

    # Caduceus-specific params
    bidirectional = True,
    bidirectional_strategy = "add",
    bidirectional_weight_tie = True,
    rcps = False
)

model = CaduceusForMaskedLM(config)
model = model.to(device)

#model.gradient_checkpointing_enable()  # Enable gradient checkpointing
#model.compile()

dataset = MethylomeDataset(seq_len=16000, split="train")
train, test = torch.utils.data.random_split(
    dataset,
    (len(dataset)-1000, 1000)
)

train_loader = DataLoader(train, batch_size=1)
test_loader = DataLoader(test, batch_size=1)

scaler = GradScaler()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

grad_acc = 32
n_epochs = 5
scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=len(train_loader)*n_epochs//grad_acc, num_warmup_steps=32)

def evaluate():
    model.eval()
    losses = []

    all_preds, all_targets, all_preds_m = [], [], []
    for step, (data, target) in enumerate(test_loader):
        model.eval()

        data = data.to(device)
        target = target.to(device)

        #with torch.autocast(device_type="cuda"):
        with torch.no_grad():
            output = model(input_ids=data, labels=target)
        loss = output.loss
        logits = output.logits

        preds = torch.argmax(logits, dim=-1).flatten().cpu().numpy()
        preds_m = torch.argmax(logits[..., [2, 5]], dim=-1).flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()
        mask = (target != 0)

        all_preds.append(preds[mask])
        all_targets.append(target[mask])
        all_preds_m.append(preds_m[mask])

    preds = np.concatenate(all_preds)
    preds_m = np.concatenate(all_preds_m)
    targets = np.concatenate(all_targets)

    wandb.log({
        "conf_mat" : wandb.plot.confusion_matrix(
            probs=None,
            y_true=targets,
            preds=preds,
            class_names=['N', 'A', 'C', 'G', 'T', 'M', 'mask', 'unused']),
        "accuracy": accuracy_score(preds, targets)
        },
        step=total_steps
    )

    preds_m = preds_m[(targets==2) | (targets==5)]
    true_m = targets[(targets==2) | (targets==5)]
    true_m = (true_m == 5)
    wandb.log({
        "conf_mat_methylated" : wandb.plot.confusion_matrix(
            probs=None,
            y_true=true_m,
            preds=preds_m,
            class_names=['C', 'M']),
        },
        step=total_steps
    )

    model.train()


# Training loop
# Warmup
with autocast():
    output = model(input_ids=torch.ones(1, 16000).long().cuda(), labels=torch.ones(1, 16000).long().cuda())

total_steps = 0

losses = []
evaluate()
for epoch in range(n_epochs):

    model.train()

    for step, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        with autocast():
            output = model(input_ids=data, labels=target)
            loss = output.loss / grad_acc

        scaler.scale(loss).backward()
        losses.append(loss.item() * grad_acc)

        if (total_steps + 1) % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            wandb.log({
                "loss": np.mean(losses),
                "lr": optimizer.param_groups[0]['lr']
            }, step=total_steps)

            losses = []

        if (total_steps + 1) % 20000 == 0:
            evaluate()

        total_steps += 1

    torch.save(model.state_dict(), f"weights/mamba2/weights_bigdata_{epoch}.pth")
