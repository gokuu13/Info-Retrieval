import pandas as pd
import numpy as np
import string
import math
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

exclude = set(string.punctuation)
data = pd.read_csv('movie_plots.csv', usecols=[1, 7])
allDocsList = data.values.tolist()

print("Removing punctuation")
for i in allDocsList[0:100]:
    print(i)
    text = i[1]
    newText = ''.join(ch for ch in text if ch not in exclude)
    i[1] = newText

print("Punctuation removed")
plot_data = [[]] * len(allDocsList)
index = 0
print("Tokenizing")
for i in range(0, 100):       #change range here
    print(i)
    text = allDocsList[i][1]
    tokenText = word_tokenize(text)
    plot_data[index].append(tokenText)
    index = index + 1

print("Tokenization done")
print("Lowering case")
for x in range(0, 100):       #change range here
    print(x)
    lowers = [word.lower() for word in plot_data[0][x]]
    plot_data[0][x] = lowers

print("Case lowered")
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()
print("Stemming and lemetizing")
for x in range(0, 100):       #change range here
    print(x)
    filtered_sentence = [w for w in plot_data[0][x] if not w in stop_words]
    plot_data[0][x] = filtered_sentence
    stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
    plot_data[0][x] = stemmed_sentence

print("Stemming and lemetization done")
l = plot_data[0]
flatten = [item for sublist in l for item in sublist]
words = flatten
words_unique = set(words)
words_unique = list(words_unique)

def tf(word, doc):
    return doc.count(word) / len(doc)

def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)

def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))

def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))

worddic = {}
print("Creating dictonary")
for doc in plot_data[0][0:100]:
    print(plot_data[0].index(doc))
    for word in words_unique:
        if word in doc:
            word = str(word)
            index = plot_data[0].index(doc)
            positions = list(np.where(np.array(plot_data[0][index]) == word)[0])
            idfs = tfidf(word, doc, plot_data[0])
            try:
                worddic[word].append([index, positions, idfs])
            except:
                worddic[word] = []
                worddic[word].append([index, positions, idfs])

print("Dictonary done")

def search(searchsentence):
    try:
        # split sentence into individual words
        searchsentence = searchsentence.lower()
        try:
            words = searchsentence.split(' ')
        except:
            words = list(words)
        enddic = {}
        idfdic = {}
        closedic = {}

        # remove words if not in worddic
        realwords = []
        for word in words:
            if word in list(worddic.keys()):
                realwords.append(word)
        words = realwords
        numwords = len(words)

        # make metric of number of occurances of all words in each doc & largest total IDF
        for word in words:
            for indpos in worddic[word]:
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                enddic[index] = amount
                idfdic[index] = idfscore
                fullcount_order = sorted(enddic.items(), key=lambda x: x[1], reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x: x[1], reverse=True)

        # make metric of what percentage of words appear in each doc
        combo = []
        alloptions = {k: worddic.get(k, None) for k in (words)}
        for worddex in list(alloptions.values()):
            for indexpos in worddex:
                for indexz in indexpos:
                    combo.append(indexz)
        comboindex = combo[::3]
        combocount = Counter(comboindex)
        for key in combocount:
            combocount[key] = combocount[key] / numwords
        combocount_order = sorted(combocount.items(), key=lambda x: x[1], reverse=True)

        # make metric for if words appear in same order as in search
        if len(words) > 1:
            x = []
            y = []
            for record in [worddic[z] for z in words]:
                for index in record:
                    x.append(index[0])
            for i in x:
                if x.count(i) > 1:
                    y.append(i)
            y = list(set(y))

            closedic = {}
            for wordbig in [worddic[x] for x in words]:
                for record in wordbig:
                    if record[0] in y:
                        index = record[0]
                        positions = record[1]
                        try:
                            closedic[index].append(positions)
                        except:
                            closedic[index] = []
                            closedic[index].append(positions)

            x = 0
            fdic = {}
            for index in y:
                csum = []
                for seqlist in closedic[index]:
                    while x > 0:
                        secondlist = seqlist
                        x = 0
                        sol = [1 for i in firstlist if i + 1 in secondlist]
                        csum.append(sol)
                        fsum = [item for sublist in csum for item in sublist]
                        fsum = sum(fsum)
                        fdic[index] = fsum
                        fdic_order = sorted(fdic.items(), key=lambda x: x[1], reverse=True)
                    while x == 0:
                        firstlist = seqlist
                        x = x + 1
        else:
            fdic_order = 0

        # also the one above should be given a big boost if ALL found together
        # could make another metric for if they are not next to each other but still close
        return (searchsentence, words, fullcount_order, combocount_order, fullidf_order, fdic_order)

    except:
        return ("")

def rank(term):
    results = search(term)
    if len(results) == 0:
        print("No Results")
        return

    print(results)

    # get metrics
    num_score = results[2]
    per_score = results[3]
    tfscore = results[4]
    order_score = results[5]

    final_candidates = []

    # rule1: if high word order score & 100% percentage terms then put at top position
    try:
        first_candidates = []

        for candidates in order_score:
            if candidates[1] > 1:
                first_candidates.append(candidates[0])

        second_candidates = []

        for match_candidates in per_score:
            if match_candidates[1] == 1:
                second_candidates.append(match_candidates[0])
            if match_candidates[1] == 1 and match_candidates[0] in first_candidates:
                final_candidates.append(match_candidates[0])

        # rule2: next add other word order score which are greater than 1

        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates), each)

        # rule3: next add top td-idf results
        final_candidates.insert(len(final_candidates), tfscore[0][0])
        final_candidates.insert(len(final_candidates), tfscore[1][0])

        # rule4: next add other high percentage score
        t3_per = second_candidates[0:3]
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates), each)

        # rule5: next add any other top results for metrics
        othertops = [num_score[0][0], per_score[0][0], tfscore[0][0], order_score[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates), top)

    # unless single term searched, in which case just return
    except:
        try:
            othertops = [num_score[0][0], num_score[0][0], num_score[0][0], per_score[0][0], tfscore[0][0]]
            for top in othertops:
                if top not in final_candidates:
                    final_candidates.insert(len(final_candidates), top)
        except Exception as ee:
            print(ee)

    final_results = []

    for index, results in enumerate(final_candidates):
        if index < 5:
            if allDocsList[results] not in final_results:
                print("RESULT", index + 1, ":", allDocsList[results][0:100], "...")
                final_results.append(allDocsList[results])

rank('Chaplin plays a waiter who fakes being a Greek Ambassador to impress a girl He then is invited to a garden party where he gets in trouble with the girls jealous boyfriend Mabel Normand wrote and directed comedies before Chaplin and mentored her young costar')