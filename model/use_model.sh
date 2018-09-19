#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

POSTFIX=2

RESULTDIR=result
DATADIR=/home/hbq1/Diploma/Data

DATAFILE=$1
LABEL=$2
ACTION=$3
RESULTFILE=v_"${LABEL}"

CORPUS_PATH="${DATADIR}"/"${DATAFILE}"

DICT_VOCAB_PATH=${DATADIR}/${DATAFILE}_vocabulary
DICT_VOCAB_FREQ_PATH=${DATADIR}/${DATAFILE}_freq_vocabulary

DICT_NGRAMS_PATH=${DATADIR}/${DATAFILE}_ngrams
DICT_NGRAMS_INFO_PATH=${DATADIR}/${DATAFILE}_ngrams_info

DICT_MORPH_PATH=${DATADIR}/${DATAFILE}_morphs
DICT_MORPH_INFO_PATH=${DATADIR}/${DATAFILE}_morphs_info

DICT_SMART_MORPH_PATH=${DATADIR}/${DATAFILE}_smart_morphs
DICT_SMART_MORPH_INFO_PATH=${DATADIR}/${DATAFILE}_smart_morphs_info

DICT_SYNS_PATH=${CORPUS_PATH}_syns
DICT_SYN_INFO_PATH=${CORPUS_PATH}_syn_info

DICT_SYNS_RT_PATH=${CORPUS_PATH}_syns_RT
DICT_SYN_RT_INFO_PATH=${CORPUS_PATH}_syn_RT_info

DICT_CONTEXTS_PATH=${CORPUS_PATH}_contexts_src
DICT_CONTEXT_INFO_PATH=${CORPUS_PATH}_context_src_info

LOG_PATH=${RESULTDIR}/${RESULTFILE}

mkdir -p "${RESULTDIR}"

make

if [[ "$ACTION" == "train" ]]
then 
	echo $ACTION;
	./fasttext skipgram -input "${DATADIR}"/"${DATAFILE}" \
		-output "${RESULTDIR}"/"${RESULTFILE}" \
		-log_path "${LOG_PATH}" \
		-lr 0.025 \
		-dim 300 \
  		-ws 5 \
		-epoch 1 \
		-minCount 5 \
		-neg 5 \
		-loss ns \
		-bucket 2000000 \
  		-minn 3 \
		-maxn 6 \
		-thread 30 \
		-t 1e-4 \
		-lrUpdateRate 100 \
		-dict_vocab_freq_path ${DICT_VOCAB_FREQ_PATH} \
		-source ngram ${DICT_NGRAMS_PATH} ${DICT_NGRAMS_INFO_PATH} \
		-source morph ${DICT_MORPH_PATH} ${DICT_MORPH_INFO_PATH} \
		-source smart_morph ${DICT_SMART_MORPH_PATH} ${DICT_SMART_MORPH_INFO_PATH} \
		-source analogy ${DICT_SYNS_PATH} ${DICT_SYN_INFO_PATH} \
		-source syns_RT ${DICT_SYNS_RT_PATH} ${DICT_SYN_RT_INFO_PATH} \
		-source contexts ${DICT_CONTEXTS_PATH} ${DICT_CONTEXT_INFO_PATH} \
		-context_cooccurences_path ${CORPUS_PATH}_context_info
		#-pretrainedModel "${RESULTDIR}"/"${RESULTFILE}.bin" \
		#2> "${RESULTDIR}"/train_log
fi

if [[ "$ACTION" == "print" ]]
then 
	echo $ACTION;
	cat "${DICT_VOCAB_PATH}" | \
			./fasttext print-vectors "${RESULTDIR}"/"${RESULTFILE}".bin \
			-dict_vocab_freq_path ${DICT_VOCAB_FREQ_PATH} \
			-source ngram ${DICT_NGRAMS_PATH} ${DICT_NGRAMS_INFO_PATH} \
			-source morph ${DICT_MORPH_PATH} ${DICT_MORPH_INFO_PATH} \
			-source smart_morph ${DICT_SMART_MORPH_PATH} ${DICT_SMART_MORPH_INFO_PATH} \
			-source analogy ${DICT_SYNS_PATH} ${DICT_SYN_INFO_PATH} \
			> "${RESULTDIR}"/vectors.txt ;
fi

