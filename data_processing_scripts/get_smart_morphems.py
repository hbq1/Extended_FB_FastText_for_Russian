# -*- coding: utf-8 -*-

import sys
import os
import argparse
import math
from collections import Counter

#from polyglot.text import Text, Word
import pymorphy2 as pm
from pymystem3 import Mystem


mystem = Mystem()

list_pref_2 = []#u"агит глав гор гос деп дет диа здрав ино кол ком лик маг мат маш мин мол об обл окруж орг парт полит потреб прод пром проп рай рег ред род рос сек сель со сов сот соц студ тер фед фин хоз хос".split(' ')
list_pref = u"анти архи би вице гипер де дез дис им интер ир квази контр макро микро обер пост пре прото псевдо ре суб супер транс ультра экзо экс экстра без бес в во вз взо вс вне внутри воз возо вос все вы до за из изо ис испод кое кой меж междо между на над надо наи не небез небес недо ни низ низо нис о об обо обез обес около от ото па пере по под подo поза после пра пре пред предо преди при про противо раз разо рас роз рос с со сверх среди су сыз тре у чрез через черес".split(' ')

list_flex = u'а я о е ь и ы а ая яя ое ее ый ать ять еть уть у ю ем ешь ете ет ут ют ал ял ала яла али яли ул ула ули ся сь'.split(' ')

list_suff3 = u"ец ани енько онько енеч онеч ин ищ ец ик ок чик ик иц ин инк оч очк ушк уш юшк ышк ишк ушек ышек ень ень оньк ех ехонь ехоньк оханьк ешень ешеньк ошеньк ошень ош ень енько онь онько ашк аш ц оньк ен енк онк еньк ень ".split(' ')
list_suff2 = u"ев ов нич ну ть ств ов ся".split(' ')
list_suff = u"к л н в а е и я к е жды либо нибудь то учи ючи вши вш изм ист алей ан ин ян ин ар ст ель лк ль ик ль ик ль иц ик иц ик иц ик адь ак ан ян ар ач ени от ет есть ость ец изн ик ин их иц ни от ун ыш".split(' ')

set_flex = { p for p in list_flex}
set_suff = { p for p in (list_suff+list_suff2+list_suff3)}
set_pref = { p for p in (list_pref+list_pref_2)}


def cut_flex(word):
	res_p = len(word)
	for flex in set_flex:
		p = word.rfind(flex)
		if (p > 3 and p + len(flex) + 1 == len(word) and res_p > p):
			res_p = p-1
	return (word[:res_p], word[res_p:])

def cut_suff(word):
	res_p = len(word)
	for s in set_suff:
		p = word.rfind(s)
		if (p > 3 and p + len(s) == len(word) and res_p > p):
			res_p = p-1
	return (word[:res_p], word[res_p:])

def cut_pref(word):
	res_p = 0
	for s in set_pref:
		p = word.find(s)
		if (p == 0 and len(word)-len(s) > 3 and res_p < len(s)):
			res_p = len(s)
	return (word[res_p:], word[:res_p])

def get_smart_morphs(word):
	try:
		word = word.decode("utf8")
	except:
		pass
	lemma = mystem.lemmatize(word)[0]
	(no_flex, flex) = cut_flex(lemma)
	(no_suff, suff) = cut_suff(no_flex)
	(main, pref) = cut_pref(no_suff)
	return (pref, main, suff, flex)
											   
def get_ngrams(word):
	l = len(word)
	res = []
	for i in xrange(0, l-2):
		for j in xrange(i+3, l+1):
			res.append(word[i:j])
	return res

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--vocab', required=True, help='path to vocab')
	parser.add_argument('--word_smart_morphems', required=True, help='path to word_morphems')
	parser.add_argument('--smart_morphem_info', required=True, help='path to morphem_info')
	args = parser.parse_args()
	
	inp_f = open(args.vocab, 'r')
	out_wm_f = open(args.word_smart_morphems, 'w')
	out_mi_f = open(args.smart_morphem_info, 'w')

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
		if len(word) == 0:
			continue
		morphs = filter(lambda x: len(x), get_smart_morphs(word))
		if len(morphs) < 2:
			continue
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
