# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import torch
import random
import matplotlib as plt

# NEURAL NETWORK CODE ---------------------------------------------------------------

# file of names
names = open('names.txt', 'r', encoding='utf-8').read().splitlines()

# defined set of characters to include even those that are missing in the dataset 
characters = sorted(list(set('aąbcćdeęfghijklłmnńoóprsśtuwyzźż')))
# creates a mapping and reverse mapping of characters and assigns 0 to . representing the end of the name
string_to_int = {s:i+1 for i,s in enumerate(characters)}
string_to_int['.'] = 0
int_to_string = {i:s for s,i in string_to_int.items()}
# print(int_to_string) # 33 characters

# model hyperparameters
block_size = 2  # context length
embedding_dim = 10  # size of the character embedding vectors
hidden_size = 200  # layer size
num_characters = 33  # total number of unique characters including .

# function to the build the dataset
def build_dataset(names):
  X, Y = [], [] # empty lists to store input X and target data Y
  for name in names:
    context = [0] * block_size # initialize context with zeros
    for character in name + '.': # iterate through each character in the name
      ix = string_to_int[character] # converts character to its corresponding integer
      X.append(context) # adds the current context to the input
      Y.append(ix) # adds the current character's int to the target list
      context = context[1:] + [ix] # updates the context by removing the first character and appending the current one

  X = torch.tensor(X) # converts input list to a pytorch tensor
  Y = torch.tensor(Y) # converts target list to a pytorch tensor
#   print(X.shape, Y.shape) # prints the shapes
  return X, Y

random.seed(77) # random seed for a deterministic result
random.shuffle(names) # shuffle the list of names to ensure randomness

# splits dataset in training (70%), validation (10%) and test (20%)
n1 = int(0.7*len(names))
n2 = int(0.8*len(names))

Xtr, Ytr = build_dataset(names[:n1]) # train set
Xdev, Ydev = build_dataset(names[n1:n2]) # validation set, to change hyperparameters
Xte, Yte = build_dataset(names[n2:]) # test set

generator = torch.Generator().manual_seed(77)
# model parameters
C = torch.randn((num_characters, embedding_dim), generator=generator) # character embedding matrix
W1 = torch.randn((block_size * embedding_dim, hidden_size), generator=generator) # first layer weights
b1 = torch.randn(hidden_size, generator=generator) # first layer bias
W2 = torch.randn((hidden_size, num_characters), generator=generator) # second layer weights
b2 = torch.randn(num_characters, generator=generator) # second layer bias

parameters = [C, W1, b1, W2, b2] # collect all parameters into a list for easy access during training

# enable gradient computation for all parameters, this is needed to train the nn
for parameter in parameters:
  parameter.requires_grad = True # gradient

# TRAINING LOOP
for i in range(50000):

  # random minibatch construct of 42 examples
  ix = torch.randint(0, Xtr.shape[0], (42,))

  # forward pass
  embedding = C[Xtr[ix]] # (33, 3, 2) # looks up character embeddings for the minibatch
  h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)  # computes hidden layer activations
  logits = h @ W2 + b2 # (33, 27) computes output logits
  loss = torch.nn.functional.cross_entropy(logits, Ytr[ix]) # calculates cross entropy loss

  # backward pass
  for parameter in parameters:
    parameter.grad = None # resets gradients to zero
  loss.backward() # computes gradients using backpropagation

  # updates parameters
  learning_rate = 0.1 if i < 100000 else 0.01
  for parameter in parameters:
    parameter.data += -learning_rate * parameter.grad # updates parameters using gradients

print(loss.item())

# training loss
embedding = C[Xtr] # (32, 3, 2)
h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
logits = h @ W2 + b2 # (32, 27)
loss = torch.nn.functional.cross_entropy(logits, Ytr)
print(f'Training loss: {loss}')

# validation loss
embedding = C[Xdev] # (32, 3, 2)
h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
logits = h @ W2 + b2 # (32, 27)
loss = torch.nn.functional.cross_entropy(logits, Ydev)
print(f'Validation loss: {loss}')

# test loss
embedding = C[Xte] # (32, 3, 2)
h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)  # FIXED SHAPE
logits = h @ W2 + b2 # (32, 27)
loss = torch.nn.functional.cross_entropy(logits, Yte)
print(f'Test loss: {loss}')

# sample names from the trained model
generator = torch.Generator().manual_seed(77)
print('Polskie imiona') # polish names
for _ in range(50): # generates 50 names

    out = [] # list to store the generated characters
    context = [0] * block_size # initialize with all zeros
    while True:
      embedding = C[torch.tensor([context])] # looks uo the embeddings for the current context
      h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)  # calculates hidden layer activations
      logits = h @ W2 + b2 # computes output logits
      probs = torch.nn.functional.softmax(logits, dim=1) # converts logits to probabilities
      ix = torch.multinomial(probs, num_samples=1, generator=generator).item() # samples a character
      context = context[1:] + [ix] # updates the context
      out.append(ix) # appends the sampled character to the output
      if ix == 0: # if token is . stop generating
        break

    print(''.join(int_to_string[i] for i in out)) # prints the name


# OLD NAMES CODE TO FINE TUNE THE PREVIOUS MODEL SO NOT AS EXTENSIVELY COMMENTED

# this code is the same as above, just with a new dataset. It fine tunes the last hidden layer on old polish names

old_names = open('old_names.txt', 'r', encoding='utf-8').read().splitlines()

random.shuffle(old_names)

# split into train, val and test sets
n1 = int(0.7 * len(old_names))
n2 = int(0.8 * len(old_names))

Xtr_new, Ytr_new = build_dataset(old_names[:n1])
Xdev_new, Ydev_new = build_dataset(old_names[n1:n2])
Xte_new, Yte_new = build_dataset(old_names[n2:])

print('Fine tuning...')
# fine-tuning loop
fine_tune_epochs = 6001
fine_tune_lr = 0.01  # learning rate

for i in range(fine_tune_epochs):

    # minibatch
    ix = torch.randint(0, Xtr_new.shape[0], (42,))
    
    # forward pass
    embedding = C[Xtr_new[ix]]  # embedding lookup
    h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)  # Hidden layer
    logits = h @ W2 + b2  # output layer
    loss = torch.nn.functional.cross_entropy(logits, Ytr_new[ix])  # loss
    
    # backwards pass
    for parameter in parameters:
        parameter.grad = None
    loss.backward()
    
    # updates parameters with a lower learning rate
    for parameter in parameters:
        parameter.data += -fine_tune_lr * parameter.grad
    
    # prints the loss
    if i % 1000 == 0:
        print(f"Fine-tuning step {i}, loss: {loss.item()}")

# validation loss
embedding = C[Xdev_new]  # Embedding lookup
h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)
logits = h @ W2 + b2
loss = torch.nn.functional.cross_entropy(logits, Ydev_new)
print(f'Validation loss after fine-tuning: {loss.item()}')

# test loss
embedding = C[Xte_new] 
h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)
logits = h @ W2 + b2 
loss = torch.nn.functional.cross_entropy(logits, Yte_new)
print(f'Test loss after fine-tuning: {loss.item()}')

# sample from the fine-tuned model
generator = torch.Generator().manual_seed(77)

print('Staropolskie imiona') # old polish names
for _ in range(50):
    out = []
    context = [0] * block_size
    while True:
        embedding = C[torch.tensor([context])] 
        h = torch.tanh(embedding.view(-1, block_size * embedding_dim) @ W1 + b1)
        logits = h @ W2 + b2
        probs = torch.nn.functional.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(int_to_string[i] for i in out))