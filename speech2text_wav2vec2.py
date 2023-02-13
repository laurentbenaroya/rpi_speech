import os
import time
import soundfile as snd
import torch
import sys

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

"""
Copyright E. Laurent Benaroya, License Gnu GPL v3
"""

if len(sys.argv) == 1:
    print('Please provide an audio file as argument')
    sys.exit()

print(torch.__version__)
torch.random.manual_seed(0)
device = 'cpu'
usecuda = True
if torch.cuda.is_available()and usecuda:
    device = "cuda:0"

SPEECH_FILE = sys.argv[1]
waveform, sample_rate = snd.read(SPEECH_FILE)

# from hugging face hub
remote_model_path = 'facebook/wav2vec2-base-960h'
print(remote_model_path)

# load processor
print('load processor from hub')
processor = Wav2Vec2Processor.from_pretrained(remote_model_path)

# load model from hub
print('load model from hub')
tic = time.time()
model = Wav2Vec2ForCTC.from_pretrained(remote_model_path)
print(f'Elapsed time {(time.time()-tic):0f}')

print('tokenize')
tic = time.time()
input_values = processor(waveform, return_tensors="pt", padding="longest", sampling_rate=sample_rate).input_values  # Batch size 1
input_values = input_values.to(device)
model = model.to(device)
print(input_values.shape)
print(f'Elapsed time {(time.time()-tic):0f}')

# retrieve logits
for _ in range(4):
    print('retrieve logits')
    tic = time.time()
    logits = model(input_values).logits
    print(f'Elapsed time {(time.time()-tic):0f}')

# take argmax and decode
print('take argmax and decode')
tic = time.time()
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(f'Elapsed time {(time.time()-tic):0f}')

print(transcription)
