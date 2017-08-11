#! /usr/bin/env python
# -*- coding: utf-8 -*-
import difflib

def findMaxSimilarityWordIndex(classified_word, word_index_dict):
    max_sim = 0.0
    max_index = -1
    for word, index in word_index_dict.items():
        sim = difflib.SequenceMatcher(None, word, classified_word).ratio()
        if sim > max_sim:
            max_sim = sim
            max_index = index
    return max_index, max_sim

def getProperWord(classified_word, words):
    word_index_dict = {word:i for i, word in enumerate(words)}
    for i in range(len(words) - 1):
        joined_word1 = words[i] + words[i + 1][0]
        joined_word2 = words[i][-1] + words[i + 1]
        word_index_dict[joined_word1] = i
        word_index_dict[joined_word2] = i + 1
    index = findMaxSimilarityWordIndex(classified_word, word_index_dict)[0]
    return words[index]

def getGTword(classified_word, title):
    words = title.split(' ')
    gt_word = getProperWord(classified_word, words)
  
