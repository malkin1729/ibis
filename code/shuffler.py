import numpy as np
import torch as T
from transformers import AutoTokenizer, AutoModelForCausalLM
import ibis
T.set_grad_enabled(False)

model_name = "gpt2"

b = 128
B = 512
max_steps = 1024
patience = 128

device = T.device('cuda:0')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

vocab = tokenizer.get_vocab()
vocab = {vocab[i]:i for i in vocab}
V = len(vocab)

unbreakable = np.zeros((V,))
for v in range(V):
    unbreakable[v] = vocab[v][0].lower() in 'abcdefghijklmnopqrstuvwxyz'

print(f'Loaded model {model_name}')

def shuffle(s):

    sentence = T.LongTensor(tokenizer.encode(s))
    before = T.LongTensor(tokenizer.encode('<|endoftext|>'))#],return_tensors='pt').input_ids[0]
    after = T.LongTensor(tokenizer.encode('<|endoftext|>'))#],return_tensors='pt').input_ids[0]
    
    mask = (1-unbreakable[sentence])
    mask[0] = 1
                                    
    for nch, k in enumerate(ibis.ibis(model, device, before, sentence, after, b, B, max_steps, patience, False, mask)):
        if nch==0: 
            starting = k.item()
            print('Original order NLL = ', starting)
        else:
            print(k[0], k[1], k[2], tokenizer.decode(k[3][1:-1], clean_up_tokenization_spaces=False))

while True:
    print('Enter a sentence (>5 words)...')
    shuffle(input())
    print('done')
