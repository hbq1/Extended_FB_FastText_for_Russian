# -*- coding: utf-8 -*-

import sys
import os
import argparse
from collections import Counter

import pymorphy2 as pm
from pymystem3 import Mystem

mystem = Mystem()

def syn_splitter(s):
	try:
		(w, p) = s.split(':')
	except:
		print s
	return (mystem.lemmatize(w), float(p))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab', required=True, help='path to vocab')
	parser.add_argument('--syns', required=True, help='path to syns')
	args = parser.parse_args()
	
	inp_f_vocab = open(args.vocab, 'r')
	inp_f_syns  = open(args.syns, 'r')
	out_f_syns = open(args.syns+"_short", 'w')

	word_cnt = Counter()
	lex_info_cnt = Counter()
	lex_info_sum = Counter()
	for line in inp_f_vocab:
		line = line.strip()
		try:
			line = line.decode('utf8')
		except:
			pass
		(rep, word) = line.split()
		word_cnt[word] = int(rep)
	inp_f_vocab.close()

	word_syn_d = dict()
	for line in inp_f_syns:
		line = line.strip()
		try:
			line = line.decode('utf8')
		except:
			pass
		word = mystem.lemmatize(line[0:line.find('\t')].strip())[0]
		if not word in word_cnt:
			continue
		out_f_syns.write( (line+'\n').encode("utf8") )
	inp_f_syns.close()
	out_f_syns.close()
	
if __name__=='__main__':
	main()
