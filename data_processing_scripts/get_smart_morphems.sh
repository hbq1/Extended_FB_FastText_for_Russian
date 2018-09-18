VOCABULARY_FILE=$1
OUTPUT_WORD_MORPHS=$2
OUTPUT_MORPH_INFO_FILE=$3

python2.7 get_smart_morphems.py \
                       --vocab ${VOCABULARY_FILE} \
					   --word_smart_morphems ${OUTPUT_WORD_MORPHS} \
					   --smart_morphem_info ${OUTPUT_MORPH_INFO_FILE};

sort -t$'\t' -nr -k3,3 ${OUTPUT_MORPH_INFO_FILE} -o ${OUTPUT_MORPH_INFO_FILE};
LC_ALL=c sort -t$'\t' -k1,1 ${OUTPUT_WORD_MORPHS} -o ${OUTPUT_WORD_MORPHS};
