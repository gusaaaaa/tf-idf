#!/usr/bin/env python
# encoding: utf-8
#
# tf-idf example in Python
# by Tim Trueman provided under:
# 
# The MIT License
# 
# Copyright (c) 2009 Tim Trueman
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# http://www.opensource.org/licenses/mit-license.php

import math
from operator import itemgetter

from os import listdir
from os.path import isfile, join

import pdb

def freq(word, document):
    return document.split(None).count(word)

def wordCount(document):
    return len(document.split(None))

def numDocsContaining(word,documentList):
    count = 0
    for document in documentList:
        if freq(word,document) > 0:
            count += 1
    return count

def tf(word, document):
    return (freq(word,document) / float(wordCount(document)))

def idf(word, documentList):
    tf = float(numDocsContaining(word,documentList))
    if tf == 0:
        return 0
    else:
        return math.log(len(documentList) / tf)

def tfidf(word, document, documentList):
    # pdb.set_trace()
  return (tf(word,document) * idf(word,documentList))

def dot(v1, v2):
    # vector v is of the form v["term"] = value, where value is the
    # result of calculating tf-idf of "term" in the document list.
    terms = set(v1.keys() + v2.keys())
    result = 0
    for term in terms:
        a = v1[term] if v1.has_key(term) else 0
        b = v2[term] if v2.has_key(term) else 0
        result = result + a * b
    return result

def norm(v):
    values = v.values()
    result = 0
    for q in values:
        result = result + q*q
    return math.sqrt(result)

def cosine(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))

def similarity(v, index):
    result = []
    for w in index:
        result.append(cosine(v, w))
    return result

if __name__ == '__main__':
    documentList = []
    path = "docs"
    files = [ fname for fname in listdir(path) if isfile(join(path, fname)) ]
    for fname in files:
        with open(join(path, fname), "r") as f:
            documentList.append(f.read())

    # build index
    index = []
    for document in documentList:
        terms = {}
        for term in document.split(None):
            terms[term] = tfidf(term, document, documentList)
        index.append(terms)

    query = "DOCUMENT #1 a TEXT"

    q = {}
    for term in query.split(None):
        q[term] = tfidf(term, query, documentList)

    print(similarity(q, index))

    v = index[0]
    print(similarity(v, index))
