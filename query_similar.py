import argparse
import sys
import os
import ast
import re
import nltk
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.word2vec import Word2Vec
import operator


def clause_preprocess(clause,stoplist):
    return remove_stop_words(remove_grammar(clause),stoplist)

def remove_stop_words(clause,stoplist):
    return [lemma.lemmatize(word.lower()).encode("utf8") for word in re.findall(r"[\w']+", clause) if word not in stoplist]

def remove_grammar(clause):
    return clause.lower().replace(",", "").replace("'", "").replace('"', "")

def get_top_similar_clauses_wmd(query,raw_clauses,processed_clauses, method):
    similarities = {}
    if method == 'wmd':
        for i in range(len(processed_clauses)):
            try:
                similarities[str(raw_clauses[i])] = wv.wmdistance(query, precessed_clause[i])
            except:
                pass
        return sorted(similarities.items(), key=operator.itemgetter(1), reverse=False)[:10]
    elif method == 'nsimilar':
        for i in range(len(processed_clauses)):
            try:
                similarities[str(raw_clauses[i])] = wv.wmdistance(query, precessed_clause[i])
            except:
                pass
        return sorted(similarities.items(), key=operator.itemgetter(1), reverse=False)[:10]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="find_similar_clauses")
    
    parser.add_argument("-m",  metavar='Method', type=str, default = 'wmd',
                        help="Choose method to calculate similarity. \
                        Word Mover Distance:'wmd'\
                        n_similarity:'nsimilar'")
                
    args = parser.parse_args()
    
    wv = Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    wv.init_sims(replace=True)
    print "succesfully loaded word to verctor model"
    
    f1 = open("stoplist.txt")
    stoplist = f1.readline().split(',')
    lemma = nltk.wordnet.WordNetLemmatizer()
    
    f = open("clauses_02232017.txt")
    data = f.readlines()
    for i in range(len(data)):
        data[i]=data[i].split('|')

    clauses = np.array(data)
    raw_clauses = clauses[1:,5]

    processed_clauses = []
    for i in range(len(raw_clauses)):
        processed_clauses.append(clause_preprocess(raw_clauses[i],stoplist))
    

    print "Input:"
    raw = input()
    method = args.m
    query = clause_preprocess(raw, stoplist)

    while raw != "stop":
        sorted_similar_clauses = get_top_similar_clauses_wmd(query,raw_clauses,processed_clauses, method)
        #print "Top",len(sorted_similar_clauses), "similar clauses are:"
        for i in range(len(sorted_similar_clauses)):
            print i+1,".",sorted_similar_clauses[i][0],":", "score is", sorted_similar_clauses[i][1]

        raw = input()
        method = args.m
        query = clause_preprocess(raw, stoplist)







