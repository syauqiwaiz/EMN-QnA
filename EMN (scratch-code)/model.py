import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#define corpusA for sentences and corpusB for query
corpusA = ["Ali go to the kitchen",
          "Mary is in the bathroom",
          "James were in the bedroom",
          "Ali dropped the milk",
          "Ali go to the studyroom"
          ]

corpusB = ["Where is the milk?"]

#create function for pre-processing sentence
def prep_sentence(sentence):
    return sentence.lower().split

train_sentences = [sent.lower().split() for sent in corpusA]
#print(train_sentences)

train_sentences2 = [sent.lower().split() for sent in corpusB]
#print(train_sentences2)

#create list for locations mentioned in corpusA and perform one-hot encoding
locations = set(["kitchen", "bathroom", "bedroom", "studyroom"])
train_labels = [[1 if word in locations else 0 for word in sent] for sent in train_sentences]
#print("train labels: ", train_labels)

#create vocab dict and add <unk> and <pad>
vocab = set(w for s in train_sentences for w in s)
#print("vocab: ", vocab)
vocab.add("<unk>")
vocab.add("<pad>")

#print(len(vocab))

#create pad window
vector_size = 7
def pad_window(sentence, vector_size, pad_token="<pad>"):
  window = [pad_token] * vector_size
  #return window + sentence + window
  for i in range(len(sentence)):
      window[i+1] = sentence[i]
  #window[1:-1] = sentence
  return window

#window_size = 2
# print("Padded sentence: ", pad_window(train_sentences[0], vector_size=vector_size))
# print("pad_window: ", pad_window(train_sentences[3], vector_size=vector_size))

#sort the vocab in alphabetical order
ix_to_word = sorted(list(vocab))

#convert word to indices
word_to_ix = {word: ind for ind, word in enumerate(ix_to_word)}
#print("Word with indices: ", word_to_ix)

#convert sentences to indices
def convert_token_to_indices(sentence, word_to_ix):
  indices = []
  for token in sentence:
    if token in word_to_ix:
      index = word_to_ix[token]
    else:
      index = word_to_ix["<unk>"]
    indices.append(index)
  return indices

#Example of converting sentence to indices
example_sentence = ["Ahmad", "is", "in", "the", "courtroom"]
example_indices = convert_token_to_indices(example_sentence, word_to_ix)
restored_example = [ix_to_word[ind] for ind in example_indices]

# print(f"Original sentence is: {example_sentence}")
# print(f"Going from words to indices: {example_indices}")
# print(f"Going from indices to words: {restored_example}")

#convert all sentences to indices
example_padded_indices = [convert_token_to_indices(s, word_to_ix) for s in train_sentences]
#print("Example:", example_padded_indices)

#create embedding
embedding_dim = 2
embedA = nn.Embedding(len(vocab), embedding_dim)
embedB = nn.Embedding(len(vocab),embedding_dim)
embedC = nn.Embedding(len(vocab),embedding_dim)
embedW = nn.Embedding(len(vocab),embedding_dim)

#example of usage of created embedding
#print((list(embedA.parameters())))

#index = word_to_ix["ali"]
#print("index: ", index)
#index_tensor = torch.tensor(index, dtype=torch.long)
#ali_embed = embedA(index_tensor)
#print(ali_embed)

#Manually convert each sentences and perform embedding
x = ["ali","go", "to", "the","kitchen"]
sentence1 = convert_token_to_indices(x, word_to_ix)
sentence_tensor1 = torch.tensor(sentence1, dtype=torch.long)
sentence_embed1 = embedA(sentence_tensor1)
#print(sentence_embed1)

x = ["mary","is", "in", "the","bathroom"]
sentence2 = convert_token_to_indices(x, word_to_ix)
sentence_tensor2 = torch.tensor(sentence2, dtype=torch.long)
sentence_embed2 = embedA(sentence_tensor2)
#print(sentence_embed2)

x = ["james","were", "in", "the","bedroom"]
sentence3 = convert_token_to_indices(x, word_to_ix)
sentence_tensor3 = torch.tensor(sentence3, dtype=torch.long)
sentence_embed3 = embedA(sentence_tensor3)
#print(sentence_embed3)

x = ["ali","dropped", "the", "milk"]
sentence4 = convert_token_to_indices(x, word_to_ix)
sentence_tensor4 = torch.tensor(sentence4, dtype=torch.long)
sentence_embed4 = embedA(sentence_tensor4)
#print(sentence_embed4.size())

x = ["ali","go", "to", "the","studyroom"]
sentence5 = convert_token_to_indices(x, word_to_ix)
sentence_tensor5 = torch.tensor(sentence5, dtype=torch.long)
sentence_embed5 = embedA(sentence_tensor5)
#print(sentence_embed5.size())

#EmbeddingA = sentence_embed1 + sentence_embed2 + sentence_embed3 + sentence_embed4 + sentence_embed5

######## testing for loop #######
EmbeddingA = []
for i in train_sentences:
    sent = pad_window(i, vector_size=vector_size)
    sentence = convert_token_to_indices(sent, word_to_ix)
    sentence_tensor = torch.tensor(sentence, dtype=torch.long)
    sentence_embed = embedA(sentence_tensor)
    Embed1 = torch.sum(sentence_embed, dim=0)
    EmbeddingA.append(Embed1)

#print("Embedding A: ", EmbeddingA)
#print(EmbeddingA.size())
EmbeddingB = []
# for i in train_sentences2:
#     sentence = convert_token_to_indices(i, word_to_ix)
#     sentence_tensor = torch.tensor(sentence, dtype=torch.long)
#     sentence_embed = embedB(sentence_tensor)
#     EmbeddingB = sentence_embed
#     EmbeddingB = torch.sum(sentence_embed, dim=0)
#print("Embedding B: ", EmbeddingB)

sentence = convert_token_to_indices(corpusB, word_to_ix)
sentence_tensor = torch.tensor(sentence, dtype=torch.long)
sentence_embed = embedB(sentence_tensor)
EmbeddingB = torch.sum(sentence_embed, dim=0)
#print("Embedding B:", EmbeddingB)

EmbeddingC = EmbeddingA
#print("Embedding C: ", EmbeddingC)

x = vocab
vocabulary = convert_token_to_indices(x, word_to_ix)
vocab_tensor = torch.tensor(vocabulary, dtype=torch.long)
# vocab_embed = embedW(vocab_tensor)
print("Vocab: ", vocab_tensor)

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters

mi = torch.stack(EmbeddingA)
u = EmbeddingB.reshape([1,2])
c = torch.stack(EmbeddingC)
w = vocab_tensor
#print("Tensor 1: ", mi)
#print("Tensor 2: ", tensor2)
#print("Tensor 3: ", tensor3)
#print("vocab: ", w)

# Operation
#print(mi[0].size())
print(u.size())
t = torch.matmul(u, torch.t(mi))
#print(torch.t(mi))
print(t)

p = torch.softmax(t.float(), dim=-1)
print(p)

ci = torch.matmul(p, c)
print("CI:", ci)

o = torch.sum(ci, 0)
print("Output:", o)

#o2 = torch.reshape(o, (4, -1))

ten3 = o + u
print("tensor3: ", ten3)

W = nn.Linear(embedding_dim, len(vocab))

ten4 = W(ten3)

pred_answer = torch.softmax(ten4, dim=-1)
print("Predicted Answer: ", pred_answer)

##next step
#cross entropy loss for corpusA, corpusB and pred_answer
# machine learning: loss, optim
# use BABI datasets
