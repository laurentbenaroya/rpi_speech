"""
distil-gpt2 next sentence generation (run on raspberrypi 3B+, 1Go RAM)
https://dejanbatanjac.github.io/gpt2-example/
https://medium.com/analytics-vidhya/ai-writer-text-generation-using-gpt-2-transformers-4c33d8c52d5a
Copyright E.L. Benaroya - 10/2022 - License GNU GLP v3
"""


from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

import time
import random
import numpy as np
import os
import sys


def gensuite():
    sentence=input("Enter your sentence : ")
    print(f' sentence type : {type(sentence)}')
    mode = input("Enter gen mode : [d/s]")
    if len(mode) == 0:
        mode = 'd'
    else:
        mode = mode.lower()
    # sentence = "What is love?"
    print(sentence)
    print(mode)

    input_ids = tokenizer.encode(sentence)
    print(f'input ids {input_ids}')
    # print(tokenizer.decode(input_ids[0]))

    tic = time.time()
    print('generate text')
    model.eval()
    if mode == 'd':
        print('deterministic')
        outputs = model.generate(torch.tensor(input_ids).unsqueeze(0),
            max_length = 10000,
            num_beams = 5,
            no_repeat_ngram_size  = 2,
            early_stopping = True)
    else:
        print('sampling')
        outputs = model.generate(
            torch.tensor(input_ids).unsqueeze(0), 
            max_length=500,
            do_sample=True,
            top_k=20,
            temperature=0.7
            )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f'outputs shape : {outputs.shape}')  # (torch.Size([1, 500]), torch.Size([500]))
    print(f'Elapsed time = {(time.time()-tic):.1f}')


print('load tokenizer')
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

print('load model')
tic = time.time()
model = GPT2LMHeadModel.from_pretrained("distilgpt2", pad_token_id=tokenizer.eos_token_id)
print(f'Elapsed time : {(time.time()-tic):.1f}')

print(f'num parameters in model = {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

print('set random seed')
seed = 0  # random.randint(0, 13)
np.random.seed(seed)
torch.random.manual_seed(seed)
# torch.cuda.manual_seed(seed)
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('display memory state')
os.system('sudo free -h')

# run inference
while True:
    try:
        gensuite()
    except KeyboardInterrupt:
        sys.exit()
