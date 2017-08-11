#! /usr/bin/env python
# -*- coding: utf-8 -*-


def getHighProbClassIndices(results, top_N=1):
    if top_N == 1:
        highest_prob_indices = [None] * results.shape[0]
        for i, probs in enumerate(results):
            highest_prob_indices[i] = probs.argsort()[::-1][0]
    else:
        high_prob_indices = [None] * results.shape[0] 
        for i, probs in enumerate(results):
            high_prob_indices[i] = probs.argsort()[::-1][0:top_N - 1]
    return high_prob_indices

def getFontNamesFromIndices(index_font_dict, indices):
    fonts = [None] * len(indices)
    for i, index in enumerate(indices):
        fonts[i] = index_font_dict[index]
    return fonts