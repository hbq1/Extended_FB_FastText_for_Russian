# -*- coding: utf-8 -*-

import sys
import os
import argparse
from collections import Counter

import pymorphy2 as pm
from pymystem3 import Mystem

mystem = Mystem()

def syn_splitter(s):
	(w, p) = s.split(':')
	return (mystem.lemmatize(w)[0], float(p))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab', required=True, help='path to vocab')
	parser.add_argument('--syns', required=True, help='path to syns')
	parser.add_argument('--word_synonyms', required=True, help='path to word_synonyms')
	parser.add_argument('--synonym_info', required=True, help='path to synonym_info')
	args = parser.parse_args()
	
	inp_f_vocab = open(args.vocab, 'r')
	inp_f_syns  = open(args.syns, 'r')
	out_wl_f = open(args.word_synonyms, 'w')
	out_li_f = open(args.synonym_info, 'w')

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
	cnt=0
	for line in inp_f_syns:
		line = line.strip()
		try:
			line = line.decode('utf8')
		except:
			pass
		word = mystem.lemmatize(line[0:line.find('\t')].strip())[0]
		if not word in word_cnt:
			continue
		syns = map(syn_splitter, filter( lambda s: s.find(':') != -1, line[line.find('\t'):].strip().split(',') ))
		syns = filter(lambda w:  w[0] in word_cnt, syns)

		if not word in word_syn_d:
			word_syn_d[word] = [s for s in syns]
		else:
			word_syn_d[word] += [s for s in syns]
		cnt += 1

	inp_f_syns.close()
	
	for word in word_cnt:
		syns_p_sum = Counter()
		syns_p_cnt = Counter()
		if word in word_syn_d:
			for syn_p in word_syn_d[word]:
				try:
					syns_p_cnt[syn_p[0]] += 1
					syns_p_sum[syn_p[0]] += syn_p[1]
				except:
					print syn_p[0]
			for s in syns_p_sum:
				syns_p_sum[s] /= syns_p_cnt[s]

			lexes = map(lambda x: x[0], syns_p_sum.most_common(30))
			for lex in lexes:
				lex_info_cnt[lex] += 1
				lex_info_sum[lex] += word_cnt[word]
			out_wl_f.write( (word + '\t' + ' '.join(lexes)).encode("utf8") + '\n' )
	inp_f_syns.close()
	out_wl_f.close()
	for (lex, cnt) in lex_info_cnt.iteritems():
		out_li_f.write( '\t'.join( [lex, str(cnt), str(lex_info_sum[lex])] ).encode("utf8") + '\n' )

	out_li_f.close()

if __name__=='__main__':
	main()
