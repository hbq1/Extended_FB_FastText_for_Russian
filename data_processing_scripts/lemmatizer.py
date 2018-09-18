# -*- coding: utf-8 -*-

import sys
import os
import argparse
import re

from nltk.corpus import stopwords
from pymystem3 import Mystem

stop_words = set(stopwords.words('russian'))
stops = ['что', 'этот', 'где', 'чем', 'кем', 'кому', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на', 'и']
for w in stops:
	stop_words.add(w)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', required=True, help='path to file to lemmatize')
	args = parser.parse_args()
    
	m = Mystem()
	with open(args.file, 'r') as f:
		for line in f:
			print ''.join( filter(lambda w: w not in stop_words, m.lemmatize(line.strip()))).strip()
	pass

if __name__=='__main__':
	main()
