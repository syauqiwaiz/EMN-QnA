import torch
import torch.nn as nn

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#define corpusA for sentences, corpusB for query and corpusC for answer

corpusA = ["Ali go to the kitchen",
          "Mary is in the bathroom",
          "James were in the bedroom",
          "Ali dropped the milk",
          "Ali go to the studyroom"
          ]

corpusB = "Where is the milk?"

corpusC = "kitchen"

#create function for pre-processing sentence
def prep_sentence(sentence):
    return sentence.lower().split

train_sentences = [sent.lower().split() for sent in corpusA]
train_sentences2 = corpusB.lower().split()
train_sentences3 = corpusC.lower().split()

#create vocab dict and add <unk> and <pad>
vocab = set(w for s in train_sentences for w in s)
vocab.add("<unk>")
vocab.add("<pad>")

#create pad window
vector_size = 7
def pad_window(sentence, vector_size, pad_token="<pad>"):
  window = [pad_token] * vector_size
  for i in range(len(sentence)):
      window[i+1] = sentence[i]
  return window

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

#convert all sentences to indices
example_padded_indices = [convert_token_to_indices(s, word_to_ix) for s in train_sentences]

#create function to convert sentences to tensor
def toTensor(sentence):
    # sentence = sentence.lower().split()
    sent = pad_window(sentence, vector_size=vector_size)
    ind = convert_token_to_indices(sent, word_to_ix)
    sentence_tensor = torch.tensor(ind, dtype=torch.long)
    return sentence_tensor

story = []
for i in train_sentences:
    sent = toTensor(i)
    story.append(sent)

query = toTensor(train_sentences2)

answer = toTensor(train_sentences3)


### Variables ###
V = len(vocab) #number of words in all sentences #20
W = 7 #number of words in each sentences
d = 2 # dimension used for embeddings

class EMN (nn.Module):

    def __init__(self):
        super(EMN, self).__init__()

        #Create Embeddings
        self.embedA = nn.Embedding(V, d)
        self.embedB = nn.Embedding(V, d)
        self.embedC = nn.Embedding(V, d)
        self.LinW = nn.Linear(d, W)

    def forward(self, story, query):

        self.story = story
        self.query = query
        ext_memory = []


        for sentence in self.story:
            mi = torch.sum(self.embedA(sentence), dim=0) # 1x2 Tensor
            ext_memory.append(mi)

        a_embed = torch.stack(ext_memory) #5 1x2 Tensors
        c_embed = torch.stack(ext_memory) #5 1x2 Tensors

        u = torch.sum(self.embedB(self.query), dim=0) # 1x2 Tensor

        inner_product = torch.matmul(torch.t(u), torch.t(a_embed)) # 1x5 Tensor

        p = torch.softmax(inner_product.float(), dim=-1) # 1x5 Tensor

        ci = torch.matmul(p, c_embed) # 1x2 Tensors

        o = torch.sum(ci, 0) # 1x1 Tensor

        ten3 = o + u #1x2 Tensor

        ten4 = self.LinW(ten3) #1x7 Tensor

        a = torch.softmax(ten4, dim=-1) # 1x7 Tensor

        return a

model = EMN()
output = model(story, query)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


EPOCH = 20

for epoch in range(EPOCH):

    model.zero_grad()
    output = model(story, query)
    output = output.to(torch.float32)
    answer = answer.to(torch.float32)
    l = loss(output, answer)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    if epoch <= EPOCH:
        print(f'Epoch [{epoch + 1}/{EPOCH}], Loss: {l.item():.4f}')

























