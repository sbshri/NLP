import sys
import math
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import re
import string
from collections import Counter
import operator

directory="C:\\Users\\Abha Saxena\\Downloads\\P1\\P1\\DATASET\\DATASET\\train"

os.chdir(directory)
deceptive_txt=""
truthful_txt=""
for path,subdirs,files in os.walk(directory):
        #print files
        for filename in files:
            if filename.endswith('deceptive.txt'):
                f=open(filename,"r")
                deceptive_txt=f.read()
                f.close()
            elif filename.endswith('truthful.txt'):
                f=open(filename,"r")
                truthful_txt=f.read()
                f.close()
                
                #a.write(filename+"\n")
words_deceptive=deceptive_txt.split(" ")
words_truthful=truthful_txt.split(" ")

print (words_deceptive[0:20])
print (words_truthful[0:20])
deceptive_txt = re.sub(r'[^\w\s]','',deceptive_txt)
truthful_txt = re.sub(r'[^\w\s]','',truthful_txt)

stop_words=set(stopwords.words("english"))
words_deceptive=word_tokenize(deceptive_txt)
words_truthful=word_tokenize(truthful_txt)


filtered_dec=[]
for w in words_deceptive:
    if w.lower() not in stop_words:
        filtered_dec.append(w)
        
filtered_tru=[]
for w in words_truthful:
    if w.lower() not in stop_words:
        filtered_tru.append(w)

from collections import Counter
import operator

#wordfreq_dec = [filtered_dec.count(p) for p in filtered_dec]
#D = dict(zip(filtered_dec,wordfreq_dec))


##counts = Counter(filtered_dec)
##D=counts
##dict_dec = dict(sorted(D.iteritems(), key=operator.itemgetter(1), reverse=True))
##counts = Counter(filtered_tru)
##T=counts
##dict_tru = dict(sorted(T.iteritems(), key=operator.itemgetter(1), reverse=True))

#print(D)
#print(dict_tru[:40])

def createNgram(tokens):
    
    biCount = {}
    uniCount = {}

##    for word1 in tokens:
##        for word2 in tokens:
##            biCount[(word1, word2)]=0

    for i in range(len(tokens)):
        if i < len(tokens) - 1:
            if (tokens[i], tokens[i+1]) in biCount:
                biCount[(tokens[i], tokens[i + 1])] += 1
            else:
                biCount[(tokens[i], tokens[i + 1])] = 1
        if tokens[i] in uniCount:
            uniCount[tokens[i]] += 1
        else:
            uniCount[tokens[i]] = 1
        

    return uniCount, biCount


uniCount_dec,biCount_dec = createNgram(filtered_dec)
#print (uniCount_dec)
#print (biCount_dec)

print(sorted(uniCount_dec.items(), key=lambda x: x[1])[:20])
print(sorted(biCount_dec.items(), key=lambda x: x[1])[:20])

uniCount_tru,biCount_tru = createNgram(filtered_tru)
print(sorted(uniCount_tru.items(), key=lambda x: x[1])[:20])
print(sorted(biCount_tru.items(), key=lambda x: x[1])[:20])

#add-k smoothing
def bigram_smoothing(biCount,k):
    for x in biCount.keys():
        biCount[x]+=k
    return biCount

biCount_dec_smooth = bigram_smoothing(biCount_dec,1)
print(sorted(biCount_dec.items(), key=lambda x: x[1])[:20])

biCount_tru_smooth = bigram_smoothing(biCount_tru,1)
print(sorted(biCount_tru.items(), key=lambda x: x[1])[:20])

def biProb(uniCount, biCount):
    biProb = {}
    for bigram in biCount.keys():
        word1 = bigram[0]
        word2 = bigram[1]
        #print(bigram)
        biProb[bigram] = biCount[bigram]/uniCount[word1]
        #print (biProb[bigram])
    return biProb

bigram_dec_probs = biProb(uniCount_dec,biCount_dec)
print(sorted(bigram_dec_probs.items(), key=lambda x: x[1])[:20])

bigram_tru_probs = biProb(uniCount_tru,biCount_tru)
print(sorted(bigram_tru_probs.items(), key=lambda x: x[1])[:20])

def perplexity(bigram_probs, count):
    log_sum = 0
    for x in bigram_probs.values():
        log_sum+=(-math.log2(x))
    log_sum_avg = log_sum/count
    print(log_sum)
    print(log_sum_avg)
    return 2**(log_sum_avg)

print(perplexity(bigram_dec_probs, len(filtered_dec)))
print(perplexity(bigram_tru_probs, len(filtered_tru)))

