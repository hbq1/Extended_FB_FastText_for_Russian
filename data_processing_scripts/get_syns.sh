VOCABULARY_FILE=$1
SYNS_FILE=$2
OUTPUT_WORD_SYNS=$3
OUTPUT_SYN_INFO_FILE=$4

python2.7 get_syns.py  --vocab ${VOCABULARY_FILE} \
					   --syns ${SYNS_FILE} \
					   --word_synonyms ${OUTPUT_WORD_SYNS} \
					   --synonym_info ${OUTPUT_SYN_INFO_FILE};

sort -t$'\t' -nr -k3,3 ${OUTPUT_SYN_INFO_FILE} -o ${OUTPUT_SYN_INFO_FILE};
LC_ALL=c sort -t$'\t' -k1,1 ${OUTPUT_WORD_SYNS} -o ${OUTPUT_WORD_SYNS};
