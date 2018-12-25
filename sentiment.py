import itertools
import math
import pickle
import re

import collections
import nltk
import nltk.classify.util
import nltk.metrics
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import *
from nltk.probability import FreqDist, ConditionalFreqDist

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
dataset1 = pd.read_csv('restaurant-train.tsv', delimiter='\t', quoting=3)


# this function takes a feature selection mechanism and returns its performance in a variety of metrics
def evaluate_features(feature_select):
    posFeatures = []
    negFeatures = []
    # splitting a string into words and punctuation
    # breaks up the sentences into lists of individual words and appends 'pos' or 'neg' after each list
    for i in range(0, len(dataset.index)):
        if dataset['Liked'][i] == 1:
            #posWords = re.findall(r"[\w']+|[.,!?;]", dataset['Review'][i].rstrip())
            posWords = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
            posWords = posWords.lower().split()
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
        else:
            negWords = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
            negWords = negWords.lower().split()
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)

    for i in range(0, len(dataset1.index)):
        if dataset1['Liked'][i] == 5 or dataset1['Liked'][i] == 4:
            posWords = re.sub('[^a-zA-Z]', ' ', dataset1['Review'][i])
            posWords = posWords.lower().split()
            posWords = [feature_select(posWords), 'pos']
            posFeatures.append(posWords)
        elif dataset1['Liked'][i] == 1:
            negWords = re.sub('[^a-zA-Z]', ' ', dataset1['Review'][i])
            negWords = negWords.lower().split()
            negWords = [feature_select(negWords), 'neg']
            negFeatures.append(negWords)

    posCutoff = int(math.floor(len(posFeatures) * 3 / 4))
    negCutoff = int(math.floor(len(negFeatures) * 3 / 4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    # trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)

    # initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)

    # puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    # prints metrics to show how well the feature selection did
    print('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    print('pos precision:', precision(referenceSets['pos'], testSets['pos']))
    print('pos recall:', recall(referenceSets['pos'], testSets['pos']))
    print('neg precision:', precision(referenceSets['neg'], testSets['neg']))
    print('neg recall:', recall(referenceSets['neg'], testSets['neg']))
    f = open('classifier.pkl', 'wb')
    pickle.dump(classifier, f)
    f.close()


# creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])


# tries using all words as the feature selection mechanism
# print( 'using all words as features')
# evaluate_features(make_full_dict)

# scores words based on chi-squared test to show information gain
def create_word_scores():
    # creates lists of all positive and negative words
    posWords = []
    negWords = []
    for i in range(0, len(dataset.index)):
        if dataset['Liked'][i] == 1:
            posWord = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
            posWord = posWord.lower().split()
            posWords.append(posWord)
        else:
            negWord = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
            negWord = negWord.lower().split()
            negWords.append(negWord)

    for i in range(0, len(dataset1.index)):
        if dataset1['Liked'][i] == 5 or dataset1['Liked'][i] == 4:
            posWord = re.sub('[^a-zA-Z]', ' ', dataset1['Review'][i])
            posWord = posWord.lower().split()
            posWords.append(posWord)
        elif dataset1['Liked'][i] == 1:
            negWord = re.sub('[^a-zA-Z]', ' ', dataset1['Review'][i])
            negWord = negWord.lower().split()
            negWords.append(negWord)

    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    # build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()

    for word in posWords:
        word_fd[word.lower()] += 1
        cond_word_fd['pos'][word.lower()] += 1
    for word in negWords:
        word_fd[word.lower()] += 1
        cond_word_fd['neg'][word.lower()] += 1

    # finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    # builds dictionary of word scores based on chi-squared test
    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
    return word_scores


# finds word scores
word_scores = create_word_scores()


# finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


# creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_features(words))
    return d


# numbers of features to select
numbers_to_test = [17500]
# tries the best_word_features mechanism with each of the numbers_to_test of features
for num in numbers_to_test:
    print('evaluating best %d word features' % (num))
    best_words = find_best_words(word_scores, num)
    evaluate_features(best_bigram_word_feats)
