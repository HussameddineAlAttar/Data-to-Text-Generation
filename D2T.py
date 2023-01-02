!pip install transformers
!pip install sentencepiece

import pandas as pd
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import  Adafactor 
import time
import warnings
warnings.filterwarnings('ignore')

import urllib.request
import zipfile
url = 'https://gitlab.com/shimorina/webnlg-dataset/-/archive/master/webnlg-dataset-master.zip?path=release_v3.0/en'
urllib.request.urlretrieve(url, 'webNLG_v3.0.zip')
with zipfile.ZipFile('webNLG_v3.0.zip', 'r') as zip_ref:
    zip_ref.extractall('webNLG_v3.0')

import glob
import os
import re
import xml.etree.ElementTree as ET

myDict = {}
special = {'á' : 'a','Á':'A','à':'a','À':'A','â':'a','Â': 'A','ä':'a','Ä':'A','ã':'a','Ã':'A','å':'a','Å':'A','æ':'ae','Æ':'Ae',
           'ç':'c','Ç':'C','é':'e','É':'E','è':'e','È':'E','ê':'e','Ê':'E','ë':'e','Ë':'E','í':'i','Í':'I','ì':'i','Ì':'I','î':'i','Î':'I','ï':'i','Ï':'I', 'İ':'I',
           'ñ':'n','Ñ':'N','ó':'o','Ó':'O','ò':'o','Ò':'O','ô':'o','Ô':'O','ö':'o','õ':'o','Õ':'O','ø':'o','œ':'oe','ß':'B','Ú':'U','ù':'u','Ù':'U','û':'u','Û':'U','ü':'u','Ü':'U', 'ú':'u'}

def removeSpecial(text):
    wordList = text.split()
    for j in range(len(wordList)):
        temp = wordList[j]
        add = False
        for i in range(len(wordList[j])):
            if wordList[j][i] in special:
                add = True
                wordList[j] = wordList[j].replace(wordList[j][i], special[wordList[j][i]])
        if add == True:
            myDict[wordList[j]] = temp

    no_special = " ".join(wordList)
    return no_special

def addSpecial(text):
    result = ""
    wordList = text.split()
    for word in wordList:
        if word in myDict:
            result += myDict[word]
        elif word + ',' in myDict:
            result += myDict[word] + ','
        elif word + '.' in myDict:
            result += myDict[word] + '.'
        elif word[-1] == '.' and word[:-1] in myDict:
            result += myDict[word[:-1]] + '.'
        elif word[-1] == ',' and word[:-1] in myDict:
            result += myDict[word[:-1]] + ','
        else:
            result += word
        result += ' '
    return result

files = glob.glob('/content/webNLG_v3.0/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/train/**/*.xml', recursive=True)
triple_re=re.compile('(\d)triples')
data={}
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    triples_num=int(triple_re.findall(file)[0])
    for sub_root in root:
        for subsub_root in sub_root:
            lexf=[]
            trpl=[]
            for entry in subsub_root:
              trpl.append(entry.text)
              lex=[triple.text for triple in entry]
              lexf.extend(lex)
            trpl=[i for i in trpl if i.replace('\n','').strip()!='' ]
            lexf=lexf[-triples_num:]
            lexf_str=(' && ').join(lexf)
            data[lexf_str]=trpl

dct={"input_text":[], "target_text":[]}
for st,unst in data.items():
    for i in unst:
        dct['input_text'].append(removeSpecial(st))
        dct['target_text'].append(removeSpecial(i))

train=pd.DataFrame(dct)
train.head()

train=train.sample(frac=1)
train.to_csv("train.csv")
train.head()

files = glob.glob('/content/webNLG_v3.0/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/dev/**/*.xml', recursive=True)
triple_re=re.compile('(\d)triples')
data={}
for file in files:
    tree = ET.parse(file)
    root = tree.getroot()
    triples_num=int(triple_re.findall(file)[0])
    for sub_root in root:
        for subsub_root in sub_root:
            lexf=[]
            trpl=[]
            for entry in subsub_root:
              trpl.append(entry.text)
              lex=[triple.text for triple in entry]
              lexf.extend(lex)
            trpl=[i for i in trpl if i.replace('\n','').strip()!='' ]
            lexf=lexf[-triples_num:]
            lexf_str=(' && ').join(lexf)
            data[lexf_str]=trpl

dct={"input_text":[], "target_text":[]}
for st,unst in data.items():
    for i in unst:
        dct['input_text'].append(removeSpecial(st))
        dct['target_text'].append(removeSpecial(i))

val=pd.DataFrame(dct)
val.head()

val=val.sample(frac=1)
val.to_csv("val.csv")
val.head()

tst='/content/webNLG_v3.0/webnlg-dataset-master-release_v3.0-en/release_v3.0/en/test/rdf-to-text-generation-test-data-with-refs-en.xml'
dt={}
tree = ET.parse(tst) 
root = tree.getroot()
for sub_root in root:
  for subsub_root in sub_root:
    lexf=[]
    trpl=[]
    for entry in subsub_root:
      trpl.append(entry.text)
      lex=[triple.text for triple in entry]
      nb=len(lex)
      lexf.extend(lex)
      lexf=lexf[-nb:]
    trpl=[i for i in trpl if i.replace('\n','').strip()!='' ]
    lexf_str=(' && ').join(lexf)
    dt[lexf_str]=trpl

dc={"input_text":[], "target_text":[]}
dc2={"input_text":[], "target_text":[]}
for st,unst in dt.items():
    for i in unst:
        dc['input_text'].append(removeSpecial(st))
        dc['target_text'].append(removeSpecial(i))
        dc2['input_text'].append(st)
        dc2['target_text'].append(i)

test=pd.DataFrame(dc)
test.head()

test=test.sample(frac=1)
test.to_csv('output.csv')
test.head()

batch_size=8
nb_batches=int(len(train)/batch_size)
epochs=5

if torch.cuda.is_available():
    dev = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
model.to(dev)

optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False)

from IPython.display import HTML, display

def progress(loss,value, max=100):
    return HTML(""" Batch loss :{loss}
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(loss=loss,value=value, max=max))

model.train()
losses=[]
for epoch in range(1,epochs+1):
  print('Running epoch: {}'.format(epoch))
  
  running_loss=0

  out = display(progress(1, nb_batches+1), display_id=True)
  for i in range(nb_batches):
    inputbatch=[]
    labelbatch=[]
    new_df=train[i*batch_size:i*batch_size+batch_size]
    for indx,row in new_df.iterrows():
      input = row['input_text']+'</s>' 
      labels = row['target_text']+'</s>'   
      inputbatch.append(input)
      labelbatch.append(labels)
    inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
    labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors='pt') ["input_ids"]
    inputbatch=inputbatch.to(dev)
    labelbatch=labelbatch.to(dev)

    optimizer.zero_grad()
    
    outputs = model(input_ids=inputbatch, labels=labelbatch)
    loss = outputs.loss
    loss_num=loss.item()
    logits = outputs.logits
    running_loss+=loss_num

    out.update(progress(loss_num,i, nb_batches+1))

    loss.backward()

    optimizer.step()
    
  running_loss=running_loss/int(nb_batches)
  losses.append(running_loss)
  
  print('Epoch: {} , Running loss: {}'.format(epoch,running_loss))

nb_b=int(len(val)/batch_size)

model.eval()

val_loss=[]
with torch.no_grad():

  for epoch in range(1,epochs+1):
    print('Running epoch: {}'.format(epoch))
  
    running_loss=0

    out = display(progress(1, nb_b+1), display_id=True)
    for i in range(nb_b):
      inputbatch=[]
      labelbatch=[]
      new_df=val[i*batch_size:i*batch_size+batch_size]
      for indx,row in new_df.iterrows():
        input = row['input_text']+'</s>' 
        labels = row['target_text']+'</s>'   
        inputbatch.append(input)
        labelbatch.append(labels)
      inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,max_length=400,return_tensors='pt')["input_ids"]
      labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,max_length=400,return_tensors='pt') ["input_ids"]
      inputbatch=inputbatch.to(dev)
      labelbatch=labelbatch.to(dev)
    
      outputs = model(input_ids=inputbatch, labels=labelbatch)
      loss = outputs.loss
      loss_num=loss.item()
      logits = outputs.logits
      running_loss+=loss_num

      out.update(progress(loss_num,i, nb_batches+1))

    running_loss=running_loss/int(nb_batches)
    val_loss.append(running_loss)
  
    print('Epoch: {} , Running val loss: {}'.format(epoch,running_loss))

import matplotlib.pyplot as plt

epch=[i for i in range(1,6)]

plt.plot(epch, losses,label='train')
plt.plot(epch,val_loss,label='val')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

model.eval()
input_ids = tokenizer.encode(removeSpecial("11th_Mississippi_Infantry_Monument | country | United States && 11th_Mississippi_Infantry_Monument | location | Adams_County,_Pennsylvania && 11th_Mississippi_Infantry_Monument | state | Pennsylvania && 11th_Mississippi_Infantry_Monument | established | 2000 && 11th_Mississippi_Infantry_Monument | category | Contributing_property && 11th_Mississippi_Infantry_Monument | municipality | Gettysburg,_Pennsylvania</s>"), return_tensors="pt")
input_ids = input_ids.to(dev)
outputs = model.generate(input_ids, max_length=400, num_beams=5, early_stopping=True)
addSpecial(tokenizer.decode(outputs[0][1:-1]))

model.eval()
input_ids = tokenizer.encode(removeSpecial("WebNLG: Mermaid_(Train_song) | recordLabel | Columbia_Records && Mermaid_(Train_song) | runtime | 3.16 && Mermaid_(Train_song) | recordLabel | Sony_Music_Entertainment && Mermaid_(Train_song) | writer | Espen_Lind && Mermaid_(Train_song) | genre | Reggae && Mermaid_(Train_song) | writer | Amund_Bjørklund </s>"), return_tensors="pt")
input_ids = input_ids.to(dev)
outputs = model.generate(input_ids, max_length=400, num_beams=5, early_stopping=True)
addSpecial(tokenizer.decode(outputs[0][1:-1]))

model.eval()
input_ids = tokenizer.encode(removeSpecial('Allen_Forrest | birthYear | 1981 && Allen_Forrest | genre | Acoustic_music && Allen_Forrest | birthPlace | "Fort Campbell, KY, raised in Dothan, AL" && Allen_Forrest | background | "solo_singer" && Allen_Forrest | birthPlace | Fort_Campbell</s>'), return_tensors="pt")
input_ids = input_ids.to(dev)
outputs = model.generate(input_ids, max_length=400, num_beams=5, early_stopping=True)
addSpecial(tokenizer.decode(outputs[0][1:-1]))

model.eval()
input_ids = tokenizer.encode(removeSpecial('Neymar | fullname | Neymar da Silva Santos Júnior && Neymar | country | Brazil && Neymar | city | São Paulo && Neymar | job | footballer </s>'), return_tensors = "pt")
input_ids = input_ids.to(dev)
outputs = model.generate(input_ids, max_length=400, num_beams=5, early_stopping=True)
addSpecial(tokenizer.decode(outputs[0][1:-1]))

model.eval()
l=[]
w=[]
for indx,row in test.iterrows():
  input=row['input_text']+' </s>'
  input_ids=tokenizer.encode(input, return_tensors="pt")
  input_ids=input_ids.to(dev)
  outputs = model.generate(input_ids, max_length=400, num_beams=4, early_stopping=True)
  x=addSpecial(tokenizer.decode(outputs[0][1:-1]))
  y=tokenizer.decode(outputs[0][1:-1])
  l.append(x)
  w.append(y)

r= []
for indx,row in test.iterrows():
  r.append(row['input_text'])

print(l)
print(len(l))
print(test.shape)

removedDup = {'input_text':[],'target_text':[]}
for i in range(len(l)):
  if r[i] not in removedDup['input_text']:
    removedDup['input_text'].append(r[i])
    removedDup['target_text'].append(w[i])

print(len(removedDup['input_text']))
print(len(removedDup['target_text']))
print(removedDup)

rdf = pd.DataFrame(removedDup)
rdf.to_csv('removedDup.csv')

dcCopy = dc.copy()
dcFixed = {}
for i in dcCopy['input_text']:
  if i not in dcFixed:
    dcFixed[i] = []

for i in range(len(dcCopy['target_text'])):
  dcFixed[dcCopy['input_text'][i]].append(dcCopy['target_text'][i])

#BlEU Score Evaluation
from nltk.translate.bleu_score import corpus_bleu

refs = []
hyp = []

for i in range(len(removedDup['input_text'])):
  hyp.append(removedDup['target_text'][i].split())
  re = dcFixed[removedDup['input_text'][i]]
  for x in range(len(re)):
    re[x] = list(re[x].split())
  refs.append(dcFixed[removedDup['input_text'][i]])

print(hyp[2])
print(refs[2])

bleuscore = corpus_bleu(refs,hyp)
print(bleuscore)
