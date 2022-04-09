import torch
import torch.nn as nn
from nltk import tokenize

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#create functions

punctuations = '''!()-[]{};:'"”“\,<>./?@#$%^&*_~'''
stopwords = ["the", "a", "is"]

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
    sorted_vocab = sorted(vocab)
    vocab = {word: ind for ind, word in enumerate(sorted_vocab)}
    return vocab

vector_size =20
def pad_window(sentence, vector_size, pad_token="<pad>"):
  window = [pad_token] * vector_size
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

def toTensor(sentence):
    sent = pad_window(sentence, vector_size=vector_size)
    ind = ToIndices(sent, vocab)
    sentence_tensor = torch.tensor(ind, dtype=torch.long)
    return sentence_tensor

def toAnswer(answer):
    for key, value in vocab.items():
         if answer.item() == value:
             return key

#Prompt

#raw_supporting_sentences = input("Please provide a few supporting sentences: ")
#raw_query = input("Please input the question: ")

raw_supporting_sentences = "Ali go to the kitchen. Mary is in the bathroom. Ali dropped the milk."

raw_query = "Where is the milk?"

answer = "kitchen"


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

#Remove stopwords
wswstory = []
for sentence in npstory:
    removed_sw =  [word for word in sentence if not word in stopwords]
    wswstory.append(removed_sw)

wswquery =  [word for word in npquery if not word in stopwords]


### Implementation ###

#Create vocabulary
vocab = create_vocab(wswstory, wswquery)

#Convert to tensors
sentences = []
for sentence in wswstory:
    sent = toTensor(sentence)
    sentences.append(sent)
story = torch.stack(sentences)

query = toTensor(wswquery)

V = 40
d = 20

class EMN (nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

        #Create Embeddings
        self.embedA = nn.Embedding(V, d)
        self.embedB = nn.Embedding(V, d)
        self.embedC = nn.Embedding(V, d)
        self.LinW = nn.Linear(d, V)

    def forward(self, story, query):
        self.story = story
        self.query = query

        a_embed = self.embedA(self.story)
        a_embed = torch.sum(a_embed, dim=0)

        u = self.embedB(self.query)
        u = torch.sum(u, dim=1)

        inner_product = torch.matmul(torch.t(u), a_embed)

        p = torch.softmax(inner_product.float(), dim=-1)

        c_embed = self.embedC(self.story)
        c_embed = torch.sum(c_embed, dim=0)

        p = torch.unsqueeze(p, dim=1)

        ci = torch.matmul(c_embed, p)

        o = torch.sum(ci, dim=1)

        tensor_3 = torch.add(o, u)

        tensor_4 = self.LinW(tensor_3)

        a = torch.softmax(tensor_4, dim=-1)

        a = torch.argmax(a, dim=-1, keepdim=True)

        a = toAnswer(a)

        return a

FILE = 'EMN.pth'

model = EMN()

model.load_state_dict(torch.load(FILE))
model.eval()

output = model(story,query)

print("Predicted answer: ", output)

if output == answer:
    print("\nThe answer is correct")
else:
    print("\nIncorrect. The answer is", answer)




















