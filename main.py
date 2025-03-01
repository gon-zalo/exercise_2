# Makes the current directory the path of the .py file
import os
import sys
os.chdir(sys.path[0])

import requests
from bs4 import BeautifulSoup
import torch
import random
import matplotlib as plt

# SCRAPING CODE ---------------------------------------------------------------------

# names_list = [] # empty list to append the names to
# for page in range(1, 14): # there are 13 pages in the website

#     url = f'https://imiennik.net/imiona-meskie.html?cp_page={page}'
#     print(f'Fetching names... Current page: {page}')
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser') # gets the html of the url
#     table = soup.find('table', class_='tabb') # gets the table where the names are stored, this is needed so that the code does not fetch other names in the website

#     names = table.find_all('a', attrs={'boy'}) # getting the names out of the table, they are in an a tag 
#     for name in names: # names is a list, for loop to iterate over the list and append the names to the list above
#         names_list.append(name.get_text())

# for name in names_list: # 'imiona męskie' means 'male names', it is under an a tag too in the website and it gets scraped, this is a simple quick fix
#     if name == 'Imiona męskie': 
#         names_list.remove(name)

# # saving names_list in a .txt file
# with open ('names.txt', 'a', encoding='utf-8') as f:
#     for name in names_list:
#         print(name.lower(), file=f)


# NEURAL NETWORK CODE ---------------------------------------------------------------

# names = open('names.txt', 'r', encoding='utf-8').read().splitlines()

# characters = sorted(list(set(''.join(names))))
# string_to_int = {s:i+1 for i,s in enumerate(characters)}
# string_to_int['.'] = 0
# int_to_string = {i:s for s,i in string_to_int.items()}
# print(int_to_string)

# block_size = 3 # context length: how many characters do we take to predict the next one?

# def build_dataset(names):
#   X, Y = [], []
#   for name in names:

#     #print(w)
#     context = [0] * block_size
#     for character in name + '.':
#       ix = string_to_int[character]
#       X.append(context)
#       Y.append(ix)
#       #print(''.join(itos[i] for i in context), '--->', itos[ix])
#       context = context[1:] + [ix] # crop and append

#   X = torch.tensor(X)
#   Y = torch.tensor(Y)
#   print(X.shape, Y.shape)
#   return X, Y

# random.seed(14)
# random.shuffle(names)

# n1 = int(0.8*len(names))
# n2 = int(0.9*len(names))

# Xtr, Ytr = build_dataset(names[:n1]) # train here
# Xdev, Ydev = build_dataset(names[n1:n2]) # validation, to change hyperparameters
# Xte, Yte = build_dataset(names[n2:]) # final set


# generator = torch.Generator().manual_seed(14) # for reproducibility
# C = torch.randn((29, 10), generator=generator) # generates a matrix 27x10 and adding random numbers to each cell, how we are representing each of the different letters (that's why there are 27), number 10 is the embedding dimension
# W1 = torch.randn((30, 200), generator=generator) # 30 is 3 tokens with 10 dimensions each one
# b1 = torch.randn(200, generator=generator)
# W2 = torch.randn((200, 29), generator=generator) # kind of like a softmax (probability), with one it would work the same way
# b2 = torch.randn(29, generator=generator)
# parameters = [C, W1, b1, W2, b2]

# for p in parameters:
#   p.requires_grad = True # gradient

# lre = torch.linspace(-3, 0, 1000)
# lrs = 10**lre

# lri = []
# lossi = []
# stepi = []

# for i in range(200000):

#   # minibatch construct
#   ix = torch.randint(0, Xtr.shape[0], (42,))

#   # forward pass
#   emb = C[Xtr[ix]] # (32, 3, 2)
#   h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
#   logits = h @ W2 + b2 # (32, 27)
#   loss = torch.nn.functional.cross_entropy(logits, Ytr[ix])
#   #print(loss.item())

#   # backward pass
#   for p in parameters:
#     p.grad = None
#   loss.backward()

#   # update
#   #lr = lrs[i]
#   lr = 0.1 if i < 100000 else 0.01
#   for p in parameters:
#     p.data += -lr * p.grad

#   # track stats
#   #lri.append(lre[i])
#   stepi.append(i)
#   lossi.append(loss.log10().item())

# print(loss.item())

# plt.plot(stepi, lossi)
# plt.show()

# # training loss
# emb = C[Xtr] # (32, 3, 2)
# h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
# logits = h @ W2 + b2 # (32, 27)
# loss = torch.nn.functional.cross_entropy(logits, Ytr)
# print(loss)

# # validation loss
# emb = C[Xdev] # (32, 3, 2)
# h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
# logits = h @ W2 + b2 # (32, 27)
# loss = torch.nn.functional.cross_entropy(logits, Ydev)
# print(loss)

# # test loss
# emb = C[Xte] # (32, 3, 2)
# h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
# logits = h @ W2 + b2 # (32, 27)
# loss = torch.nn.functional.cross_entropy(logits, Yte)
# print(loss)

# # sample from the model
# generator = torch.Generator().manual_seed(2147483647 + 63)

# for _ in range(50):

#     out = []
#     context = [0] * block_size # initialize with all ...
#     while True:
#       emb = C[torch.tensor([context])] # (1,block_size,d)
#       h = torch.tanh(emb.view(1, -1) @ W1 + b1)
#       logits = h @ W2 + b2
#       probs = torch.nn.functional.softmax(logits, dim=1)
#       ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
#       context = context[1:] + [ix]
#       out.append(ix)
#       if ix == 0:
#         break

#     print(''.join(int_to_string[i] for i in out))