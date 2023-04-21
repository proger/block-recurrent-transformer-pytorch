import math
import torch

from block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 1e-4
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

device = 'cuda:0'
model.to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

bpc, count = 0., 0
with torch.inference_mode():
    with open('data/dev-clean.txt') as file:
        for line in file:
            line = '\n' + line.strip()
            inp = torch.tensor(list(map(ord, line))).to(device)

            loss = train_wrapper.forward(inp[None, ...])
            print(loss.item(), line[1:])
            bpc += loss.item() / math.log(2)
            count += 1
print('average bpc:', bpc / count)