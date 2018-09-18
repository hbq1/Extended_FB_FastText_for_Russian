# -*- coding: utf-8 -*-

import sys
import os
import argparse
import math
from collections import Counter

import pymorphy2 as pm

def get_main_part(word, lexems):
	count = Counter()
	score = Counter()
	cnt_lexems = 0
	for lex in lexemes:
		cnt_lexems += 1
		length = len(lex)
		was = set()
		for i in xrange(length-1):
			cand = (lex[i:j] for j in xrange(i+2, length))
			for c in cand:
				if c in was:
					continue
				was.add(c)
				count[c] += 1
				score[c] += len(c)

	cnt_pairs = filter(lambda x: x[1] == cnt_lexems, count.most_common())

	size = len(cnt_pairs)
	if size > 0:
		root_i = 0
		i = 0
		while i + 1 < size and cnt_pairs[i][1] == cnt_pairs[i+1][1]:
			if (len(cnt_pairs[i][0]) > len(cnt_pairs[root_i][0])):
				root_i = i
			i += 1
		print count[cnt_pairs[root_i][0]]
		return ('have', cnt_pairs[root_i][0])
		return (l[0] for l in r[:last]) 
	return ()

def get_morphs(word, lexemes):
	count = Counter()
	score = Counter()
	cnt_lexems = 0
	for lex in lexemes:
		cnt_lexems += 1
		length = len(lex)
		for i in xrange(length-1):
			cand = (lex[i:j] for j in xrange(i+2, length))
			for c in cand:
				count[c] += 1
				score[c] += math.sqrt(len(c))
	if len(word) < 3:
		return [word]
	if len(word) < 5:
		return count.keys()[0:min(20,len(count))]

	pairs = filter(lambda x: count[x[0]] > cnt_lexems / 2, score.most_common())

	size = len(pairs)
	last = min(20, size/2)
	if size > 0:
		return [l[0] for l in pairs[:last]]
	return ()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab', required=True, help='path to vocab')
	parser.add_argument('--word_morphems', required=True, help='path to word_morphems')
	parser.add_argument('--morphem_info', required=True, help='path to morphem_info')
	args = parser.parse_args()
	
	inp_f = open(args.vocab, 'r')
	out_wm_f = open(args.word_morphems, 'w')
	out_mi_f = open(args.morphem_info, 'w')

	morph_info_cnt = Counter()
	morph_info_sum = Counter()
	ma = pm.MorphAnalyzer()
	for line in inp_f:
		line = line.strip()
		try:
			line = line.decode('utf8')
		except:
			pass
		(rep, word) = line.split()
		morphs = get_morphs(word, (l[0] for l in ma.parse(word)[0].lexeme))
		for morph in morphs:
			morph_info_cnt[morph] += 1
			morph_info_sum[morph] += int(rep)
		out_wm_f.write( (word + '\t' + ' '.join(morphs)).encode("utf8") + '\n' )
	inp_f.close()
	out_wm_f.close()
	for (morph, cnt) in morph_info_cnt.iteritems():
		out_mi_f.write( '\t'.join( [morph, str(cnt), str(morph_info_sum[morph])] ).encode("utf8") + '\n' )

	out_mi_f.close()

if __name__=='__main__':
	main()
