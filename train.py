import gzip
import random
import tqdm
import numpy as np
from pathlib import Path

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import wandb

from accelerate import Accelerator
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 1e-3
VALIDATE_EVERY = 100
PRIME_LENGTH = 64
GENERATE_EVERY = 250
GENERATE_LENGTH = 2048
SEQ_LEN = 2048

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# accelerator

accelerator = Accelerator()

device = accelerator.device
acc_print = accelerator.print

# instantiate palm

model = BlockRecurrentTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8,
    max_seq_len = 1024,
    block_width = 512,
    num_state_vectors = 512,
    recurrent_layers = (4,),
    use_flash_attn = True
)

train_wrapper = RecurrentTrainerWrapper(
    model,
    xl_memories_dropout = 0.1,
    state_dropout = 0.1,
)

model.to(device)

# wandb
wandb.init(project='block-recurrent')


train_data = Path("./data/librispeech-lm-norm.txt.gz")
with gzip.open(train_data) as file:
    np_train = np.frombuffer(file.read(int(train_data.stat().st_size)), dtype=np.uint8).copy()
    data_train = torch.from_numpy(np_train)

valid_data = Path("./data/dev-clean.txt")
with open(valid_data, "rb") as file:
    np_valid = np.frombuffer(file.read(int(valid_data.stat().st_size)), dtype=np.uint8).copy()
    data_val = torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# optimizer

optim = Adam(model.parameters())

model, optim, train_loader, val_loader = accelerator.prepare(
    model, optim, train_loader, val_loader
)


def compute_lr(step, warmup=NUM_BATCHES//10, lr=LEARNING_RATE):
    # linear warmup followed by cosine decay to 10% of original
    lrmin = lr * 0.01
    if step < warmup:
        t = step / warmup
        return lrmin + (lr-lrmin) * t
    else:
        t = (step - warmup) / (NUM_BATCHES - warmup)
        return lrmin + (lr-lrmin) * 0.5 * (1 + np.cos(np.pi * t))


# training

for step in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    optim.param_groups[0]["lr"] = compute_lr(step)
    model.train()

    loss = 0.
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch_loss = train_wrapper(next(train_loader)) / GRADIENT_ACCUMULATE_EVERY
        accelerator.backward(batch_loss)
        loss += batch_loss

    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 0.5)
    wandb.log({
        'train/loss': loss.item(),
        'train/ppl': loss.exp().item(),
        'train/lr': compute_lr(step),
        'train/step': step,
        'train/grad_norm': grad_norm.item()
    })

    optim.step()
    optim.zero_grad()

    if step % VALIDATE_EVERY == 0:
        acc_print(f"step {step} training loss: {loss.item()} grad_norm: {grad_norm.item()}")

        model.eval()
        with torch.no_grad():
            loss = train_wrapper(next(val_loader))
            acc_print(f"validation loss: {loss.item()}")
            wandb.log({
                'valid/loss': loss.item(),
                'valid/ppl': loss.exp().item(),
            })


    if step % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)

        sample = train_wrapper.generate(inp[None, ...], length = GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        acc_print(prime, "|", output_str, "\n")
        accelerator.save(model.state_dict(), "model.pt")