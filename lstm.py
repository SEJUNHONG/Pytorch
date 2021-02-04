import torch
import torch.nn as nn
import unidecode
import string
import random
import re
import time, math

num_epochs=2000
print_every=100
plot_every=10

chunk_len=200

hidden_size=100
batch_size=1
num_layers=1
embedding_size=70
lr=0.002

all_characters=string.printable
n_characters=len(all_characters)
print(all_characters)
print('num_char=', n_characters)

file=unidecode.unidecode(open('C:/Python/pytorch/data/input.txt').read())
file_len=len(file)
print('file_len=', file_len)

def random_chunk():
    start_index=random.randint(0, file_len-chunk_len)
    end_index=start_index+chunk_len+1
    return file[start_index:end_index]

def char_tensor(string):
    tensor=torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c]=all_characters.index(string[c])
    return tensor

print(char_tensor('ABCdef'))

def random_training_set():
    chunk=random_chunk()
    inp=char_tensor(chunk[:-1])
    target=char_tensor(chunk[1:])
    return inp, target

class RNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.rnn = nn.LSTM(self.embedding_size,self.hidden_size,self.num_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        
    
    def forward(self, input, hidden, cell):
        out = self.encoder(input.view(1,-1))
        out,(hidden,cell) = self.rnn(out,(hidden,cell))
        out = self.decoder(out.view(batch_size,-1))
        return out,hidden,cell

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        cell = torch.zeros(self.num_layers,batch_size,self.hidden_size)
        return hidden,cell
    

model = RNN(n_characters, embedding_size, hidden_size, n_characters, num_layers)

inp = char_tensor("A")
print(inp)
hidden,cell = model.init_hidden()
print(hidden.size())

out,hidden,cell = model(inp,hidden,cell)
print(out.size())

optimizer=torch.optim.Adam(model.parameters(), lr=lr)
loss_func=nn.CrossEntropyLoss()

def test():
    start_str = "b"
    inp = char_tensor(start_str)
    hidden,cell = model.init_hidden()
    x = inp

    print(start_str,end="")
    for i in range(200):
        output,hidden,cell = model(x,hidden,cell)

        output_dist = output.data.view(-1).div(0.8).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        predicted_char = all_characters[top_i]

        print(predicted_char,end="")

        x = char_tensor(predicted_char)

for i in range(num_epochs):
    inp,label = random_training_set()
    hidden,cell = model.init_hidden()

    loss = torch.tensor([0]).type(torch.FloatTensor)
    optimizer.zero_grad()
    for j in range(chunk_len-1):
        x  = inp[j]
        y_ = label[j].unsqueeze(0).type(torch.LongTensor)
        y,hidden,cell = model(x,hidden,cell)
        loss += loss_func(y,y_)

    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print("\n",loss/chunk_len,"\n")
        test()
        print("\n\n")