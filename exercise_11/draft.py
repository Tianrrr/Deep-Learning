import re
import torch
import torch.nn as nn
from exercise_code.rnn.text_classifiers import RNNClassifier
# def tokenize(text):
#     return [s.lower() for s in re.split(r'\W+', text) if len(s) > 0]


# line = 'aaa,SDDADAD,ccc ; ddd   Eee,fGf'
# print(tokenize(line))


# a= torch.normal(0,1,size=(2, 3))
# print(a)


# weight = torch.normal(0,1,size=(5,3))
# inputs = torch.tensor([[1,0,4,2],[1,2,3,0]])
# print(weight)
# print(inputs)
# print(inputs.shape)
# print(weight[inputs,:].shape)

# print(weight[inputs,:])


# seq_len=10
# batch_size=3

# # Create a random sequence
# x = torch.randint(0, 5-1, (seq_len, batch_size))
# print(x)
seq_len=5
batch_size=3

# Create a random sequence

 
x = torch.randint(1, 5-1, (seq_len, batch_size))
print(x)
lengths = torch.tensor([seq_len-i for i in range(batch_size)]).long()
print(lengths)
a = [x[:lengths[i], i].unsqueeze(1) for i in range(lengths.numel())]
print(a[0],'\t',a[1],'\t',a[2])
model = RNNClassifier(num_embeddings=5002, embedding_dim=4, hidden_size=2)
print(model(x, lengths))

print(model(a[0]),model(a[1]),model(a[2]))
# print(model(a[1]))
