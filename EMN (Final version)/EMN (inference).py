# Implemented in PyCharm IDE using Python 3.9.1
# Machine Learning framework used is PyTorch 1.8.0
# PyTorch "torchtext" version 0.9.0

#import necessary libraries
import torch
import torch.nn as nn
from nltk import tokenize

punctuations = '''!()-[]{};:'"”“\,<>./?@#$%^&*_~'''

#create pre-processing functions
def remove_punct(sentence):
    for element in sentence:
        if element in punctuations:
            sentence = sentence.replace(element, "")
    return sentence

def split_sentence(sentence):
    new_sentence = tokenize.sent_tokenize(sentence)
    return new_sentence

def prep_sentence(sentence):
    prepped_sentence = sentence.lower().split()
    return prepped_sentence

def create_vocab(story, query):
    vocab = set(w for s in story for w in s)
    vocab.update(query)
    vocab.add("<unk>")
    vocab.add("<pad>")
    #vocab.add(answer)
    sorted_vocab = sorted(vocab)
    vocab = {word: ind for ind, word in enumerate(sorted_vocab)}
    return vocab

#vector size is different for every task and modified manually
vector_size_story =20
def pad_window_story(sentence, vector_size_story, pad_token="<pad>"):
  window = [pad_token] * vector_size_story
  for i in range(len(sentence)):
      window[i+1] = sentence[i]
  return window

vector_size_query =10
def pad_window_query(sentence, vector_size_query, pad_token="<pad>"):
  window = [pad_token] * vector_size_query
  for i in range(len(sentence)):
      window[i+1] = sentence[i]
  return window

def ToIndices(sentence, vocab):
    indices = []
    for token in sentence:
      if token in sentence:
        index = vocab[token]
      else:
        index = vocab["<unk>"]
      indices.append(index)
    return indices

def toTensor_story(sentence):
    sent = pad_window_story(sentence, vector_size_story=vector_size_story)
    ind = ToIndices(sent, vocab)
    sentence_tensor = torch.tensor(ind, dtype=torch.long)
    return sentence_tensor

def toTensor_query(sentence):
    sent = pad_window_query(sentence, vector_size_query=vector_size_query)
    ind = ToIndices(sent, vocab)
    sentence_tensor = torch.tensor(ind, dtype=torch.long)
    return sentence_tensor

def toAnswer(answer):
    for key, value in vocab.items():
         if answer[value] == value:
             return key

#take user input

raw_supporting_sentences = input("Input any supporting sentence(s): ")
raw_query = input("Input the question: ")

#If the user input does not have the answer in it, add answer manually to vocab
#answer = "example_answer"


### Preprocess text ###

#Split supporting sentences
split_supporting_sentences = split_sentence(raw_supporting_sentences)

#Remove punctuations
presentences = []
for i in split_supporting_sentences:
    sentence = remove_punct(i)
    presentences.append(sentence)

prequery = remove_punct(raw_query)

#Lower and split into words
npstory = [prep_sentence(sent) for sent in presentences]
npquery = prep_sentence(prequery)


### Implementation ###

#Create vocabulary
vocab = create_vocab(npstory, npquery)

#Convert to tensors
sentences = []
for sentence in npstory:
    sent = toTensor_story(sentence)
    sentences.append(sent)

query = toTensor_query(npquery)

#input sentences need to be of the same size of bAbI dataset [batch_size x memory_size x *sentence length*]
#query: [batch_size x *sentence length*]

#create memory size for input
memory_size =50

for _ in range(memory_size-len(npstory)):
    zero_tensor = torch.zeros([vector_size_story], dtype=int)
    sentences.append(zero_tensor)
story = torch.stack(sentences)

#create batch size for input
batch_size = 32

list_story = []
for _ in range(batch_size):
    list_story.append(story)
batch_story = torch.stack(list_story)

list_query = []
for _ in range(batch_size):
    list_query.append(query)
batch_query = torch.stack(list_query)

#Manually modify embedding shape according to task
V = 40
d = 20
num_hops = 3

class EMN (nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

        #Create Embeddings
        self.embedA = nn.Embedding(V, d, padding_idx=0)
        self.embedB = nn.Embedding(V, d, padding_idx=0)
        self.embedC = nn.Embedding(V, d, padding_idx=0)
        self.LinW = nn.Linear(d, V)

        self.embedA.weight.data.normal_(0, 0.1)
        self.embedB.weight.data.normal_(0, 0.1)
        self.embedC.weight.data.normal_(0, 0.1)
        self.LinW.weight.data.normal_(-0.25, 0.25)


    def forward(self, story, query):
        self.story = story
        self.query = query

        u = self.embedB(self.query)
        u = torch.sum(u, dim=1)

        for k in range(num_hops):
            a_embed = self.embedA(self.story)
            a_embed = torch.sum(a_embed, dim=2)

            c_embed = self.embedC(self.story)
            c = torch.sum(c_embed, dim=2)

            ip = torch.bmm(a_embed, u.unsqueeze(2)).squeeze()

            p = torch.softmax(ip, -1).unsqueeze(1)

            o = torch.bmm(p, c).squeeze(1)
            u = o + u

        a = self.LinW(u)

        return a

if __name__ == '__main__':

    #Load saved model
    FILE = 'EMN.pth'

    model = EMN()
    model.load_state_dict(torch.load(FILE))
    model.eval()

    #pass pre-processed inputs to loaded model
    output = model(batch_story,batch_query)

    a = torch.argmax(output, dim=-1, keepdim=True)

    a = toAnswer(a)

    #get predicted answer
    print("Predicted answer: ", a)

























