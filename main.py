# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import torch
import random
import matplotlib as plt

# NEURAL NETWORK CODE ---------------------------------------------------------------

names = open('names.txt', 'r', encoding='utf-8').read().splitlines()
old_names = open('old_names.txt', 'r', encoding='utf-8').read().splitlines()

characters = sorted(list(set(''.join(names))))
string_to_int = {s:i+1 for i,s in enumerate(characters)}
string_to_int['.'] = 0
int_to_string = {i:s for s,i in string_to_int.items()}
print(int_to_string) # 30 characters

# dataset
block_size = 2  # Adjusted block size
embedding_dim = 10  # Fixed embedding dimension
hidden_size = 200  # Hidden layer size
num_characters = 30  # Total unique characters, update if needed

def build_dataset(names):
  X, Y = [], []
  for name in names:

    #print(w)
    context = [0] * block_size
    for character in name + '.':
      ix = string_to_int[character]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

random.seed(14)
random.shuffle(names)
random.shuffle(old_names)

n1 = int(0.7*len(names))
n2 = int(0.8*len(names))

Xtr, Ytr = build_dataset(names[:n1]) # train here
Xdev, Ydev = build_dataset(names[n1:n2]) # validation, to change hyperparameters
Xte, Yte = build_dataset(names[n2:]) # final set


generator = torch.Generator().manual_seed(12345)  # For reproducibility

C = torch.randn((num_characters, embedding_dim), generator=generator)  
W1 = torch.randn((block_size * embedding_dim, hidden_size), generator=generator)  
b1 = torch.randn(hidden_size, generator=generator)
W2 = torch.randn((hidden_size, num_characters), generator=generator)  
b2 = torch.randn(num_characters, generator=generator)

parameters = [C, W1, b1, W2, b2]

for p in parameters:
  p.requires_grad = True # gradient

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

# TRAINING
for i in range(50000):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (42,))

  # forward pass
  emb = C[Xtr[ix]] # (32, 3, 2)
  h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
  logits = h @ W2 + b2 # (32, 27)
  loss = torch.nn.functional.cross_entropy(logits, Ytr[ix])
  #print(loss.item())

  # backward pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  #lr = lrs[i]
  lr = 0.1 if i < 100000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # track stats
  #lri.append(lre[i])
  stepi.append(i)
  lossi.append(loss.log10().item())

print(loss.item())

# training loss
emb = C[Xtr] # (32, 3, 2)
h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
logits = h @ W2 + b2 # (32, 27)
loss = torch.nn.functional.cross_entropy(logits, Ytr)
print(f'Training loss: {loss}')

# validation loss
emb = C[Xdev] # (32, 3, 2)
h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
logits = h @ W2 + b2 # (32, 27)
loss = torch.nn.functional.cross_entropy(logits, Ydev)
print(f'Validation loss: {loss}')

# test loss
emb = C[Xte] # (32, 3, 2)
h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
logits = h @ W2 + b2 # (32, 27)
loss = torch.nn.functional.cross_entropy(logits, Yte)
print(f'Test loss: {loss}')


# sample from the model
generator = torch.Generator().manual_seed(33)

for _ in range(50):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
      logits = h @ W2 + b2
      probs = torch.nn.functional.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(int_to_string[i] for i in out))


torch.save(parameters, 'trained_model.pth')

# OLD NAMES CODE

# OLD NAMES DATASET
characters = sorted(list(set(''.join(old_names))))
string_to_int = {s:i+1 for i,s in enumerate(characters)}
string_to_int['.'] = 0
int_to_string = {i:s for s,i in string_to_int.items()}
print(int_to_string) # 32 characters

# OLD NAMES DATASET
random.seed(14)
random.shuffle(old_names)

n1 = int(0.7 * len(names))
n2 = int(0.8 * len(names))

Xtr, Ytr = build_dataset(names[:n1])  # Training set
Xdev, Ydev = build_dataset(names[n1:n2])  # Validation set
Xte, Yte = build_dataset(names[n2:])  # Test set

parameters = torch.load('trained_model.pth', weights_only=True)
for p in parameters:
    p.requires_grad = False  # Disable training

W2 = torch.randn((hidden_size, len(characters)), generator=generator)  # New output size
b2 = torch.randn(len(characters), generator=generator)

# sample from the model
generator = torch.Generator().manual_seed(12345)

for _ in range(50):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
      logits = h @ W2 + b2
      probs = torch.nn.functional.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    print(''.join(int_to_string[i] for i in out))