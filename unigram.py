import sys
import math
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
import re
import string
import collections
from collections import Counter
import operator


directory="/Users/sahanabs/Desktop/P1/DATASET/train"

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


def tokenize_remove_stuff(text):
    text = re.sub(r'[^\w\s]','',text)
    stop_words=set(stopwords.words("english"))
    words=word_tokenize(text)
    filtered_words=[]
    for w in words:
        if w.lower() not in stop_words:
            filtered_words.append(w)
    return filtered_words


def tokenize_and_process(text):
    text = re.sub(r'[^\w\s]','',text)
    words = text.split(" ")
    return words


def plug_unk_words_less_accuracy(tokens):
    seen = []
    for i in range(len(tokens)):
        if tokens[i] not in seen:
            seen.append(tokens[i])
            tokens[i] = "UNK"
    return tokens

def plug_unk_words(tokens):
    seen = []
    for i in range(len(tokens)):
        if tokens[i] not in seen:
            seen.append(tokens[i])
            tokens[i] = "UNK"
    return tokens


def createUnigram(tokens):
    # uniCount = collections.defaultdict(lambda: 0.001)
    uniCount = {}
    for i in range(len(tokens)):
        if tokens[i] in uniCount:
            uniCount[tokens[i]] += 1
        else:
            uniCount[tokens[i]] = 1
    return uniCount

def unigram_probs(uniCount):
    n = float(sum(uniCount.values()))
    unigram_probs = {}
    for x in uniCount.keys():
        unigram_probs[x] = uniCount[x]/n
    return unigram_probs


def perplexity_unk(unigram_probs, tokens):
    log_sum = 0
    count = len(tokens)
    for x in tokens:
        if x not in unigram_probs.keys():
            x = "UNK"
        log_sum+=(-math.log2(unigram_probs[x]))
    log_sum_avg = log_sum/count
    return 2**(log_sum_avg)

def perplexity(unigram_probs, tokens):
    log_sum = 0
    count = len(tokens)
    for x in tokens:
        if x not in unigram_probs.keys():
            log_sum+=(-math.log2(0.01/len(unigram_probs)))
        else:
            log_sum+=(-math.log2(unigram_probs[x]))

    log_sum_avg = log_sum/count
    return 2**(log_sum_avg)

########################################################################################################################################################################

filtered_dec = tokenize_and_process(deceptive_txt)
filtered_tru = tokenize_and_process(truthful_txt)

print(filtered_dec[:20])

# print("Adding unknown words to tokens")
# filtered_dec = plug_unk_words(filtered_dec)
# filtered_tru = plug_unk_words(filtered_tru)

print(filtered_dec[:20])

print("Creating unigram model")
uniCount_dec = createUnigram(filtered_dec)
uniCount_tru = createUnigram(filtered_tru)


print("Calculating unigram probability")
uniCount_dec_probs = unigram_probs(uniCount_dec)
uniCount_tru_probs = unigram_probs(uniCount_tru)


# print(sorted(uniCount_dec_probs.items(), key=lambda x: x[1],reverse=True)[:20])
# print(sorted(uniCount_tru_probs.items(), key=lambda x: x[1],reverse=True)[:20])


def test(list_of_reviews):
    classified_as_true = 0
    classified_as_deceptive = 0
    for review in list_of_reviews:
        tokens = tokenize_and_process(review)
        if(len(tokens) > 0):
            unimodel_deceptive = perplexity(uniCount_dec_probs, tokens)
            unimodel_true = perplexity(uniCount_tru_probs, tokens)
            if(unimodel_deceptive > unimodel_true):
                classified_as_true+=1
            else:
                classified_as_deceptive+=1

    return classified_as_true, classified_as_deceptive


#Validation

directory_val="/Users/sahanabs/Desktop/P1/DATASET/validation"

os.chdir(directory_val)
deceptive_txt_val=""
truthful_txt_val=""
for path,subdirs,files in os.walk(directory_val):
        #print files
        for filename in files:
            if filename.endswith('deceptive.txt'):
                f=open(filename,"r")
                deceptive_txt_val=f.read()
                f.close()
            elif filename.endswith('truthful.txt'):
                f=open(filename,"r")
                truthful_txt_val=f.read()
                f.close()


validation_reviews_tru = truthful_txt_val.split('\n')


true,dec = test(validation_reviews_tru)
print("#################################################################################################################################################################")

print("Input data: Validation set, TRUE DATA")
print("Model classified the following as TRUE: ", true)
print("Model classified the following as DECEPTIVE: ", dec)

print("ACCURACY OF THE MODEL = " , true/(true+dec)*100, "%")



validation_reviews_dec = deceptive_txt_val.split('\n')

true,dec = test(validation_reviews_dec)

print("#################################################################################################################################################################")

print("Input data: Validation set, DECEPTIVE DATA")
print("Model classified the following as TRUE: ", true)
print("Model classified the following as DECEPTIVE: ", dec)

print("ACCURACY OF THE MODEL = " , dec/(true+dec)*100, "%")

print("#################################################################################################################################################################")

print("FINAL TEST - WE HOPE")

directory="/Users/sahanabs/Desktop/P1/DATASET/test"

os.chdir(directory)
txt_val=""

for path,subdirs,files in os.walk(directory):
        #print files
        for filename in files:
            f=open(filename,"r")
            txt_val=f.read()
            f.close()

test_reviews = txt_val.split('\n')


print ('Id,Prediction')
for i in range(len(test_reviews)):
    tokens = tokenize_and_process(test_reviews[i])
    if(len(tokens) > 0):
        unimodel_deceptive = perplexity(uniCount_dec_probs, tokens)
        unimodel_true = perplexity(uniCount_tru_probs, tokens)
        if(unimodel_deceptive > unimodel_true):
            print (str(i)+',0')
        else:
            print (str(i)+',1')


