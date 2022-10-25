# rpi_speech
This repo goes with my post [reconnaissance de la parole sur Raspberry Pi](https://astus-geekus.com/2022/10/23/607/), not yet published.
Audio comes from : VCTK

usage :
python speech2text_wav2vec2.py audio/p232_228.wav


distilgpt2.py raises this error atm
RuntimeError: Failed to import transformers.models.gpt2.modeling_gpt2 because of the following error (look up to see its traceback):
cannot import name 'autocast' from 'torch.cuda.amp' (/home/benaroya/.virtualenvs/main/lib/python3.7/site-packages/torch/cuda/amp/__init__.py)
