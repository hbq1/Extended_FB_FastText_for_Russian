CORPUS_PATH=$1

FILE_SYNS_PATH="syns/syns_full"

DICT_VOCAB_PATH=${CORPUS_PATH}_vocabulary
DICT_VOCAB_FREQ_PATH=${CORPUS_PATH}_freq_vocabulary

DICT_NGRAMS_PATH=${CORPUS_PATH}_ngrams
DICT_NGRAMS_INFO_PATH=${CORPUS_PATH}_ngrams_info

DICT_MORPHS_PATH=${CORPUS_PATH}_morphs
DICT_MORPHS_INFO_PATH=${CORPUS_PATH}_morphs_info

DICT_SMART_MORPHS_PATH=${CORPUS_PATH}_smart_morphs
DICT_SMART_MORPHS_INFO_PATH=${CORPUS_PATH}_smart_morphs_info

DICT_SYNS_PATH=${CORPUS_PATH}_syns
DICT_SYN_INFO_PATH=${CORPUS_PATH}_syn_info

TOP_N_WORDS=1000000

# 0. shuffle corpus
shuf $1 > $1_tmp ; mv $1_tmp $1;

# 1. prepare vocabulary
./get_vocabulary.sh ${CORPUS_PATH} ${DICT_VOCAB_PATH} ${DICT_VOCAB_FREQ_PATH} ${TOP_N_WORDS};
echo "vocabulary prepared"

# 2. prepare ngrams
./get_ngrams.sh ${DICT_VOCAB_FREQ_PATH} ${DICT_NGRAMS_PATH} ${DICT_NGRAMS_INFO_PATH};
echo "ngrams prepared"

# 3. prepare morphems
./get_morphems.sh ${DICT_VOCAB_FREQ_PATH} ${DICT_MORPHS_PATH} ${DICT_MORPHS_INFO_PATH};
echo "morphems prepared"

# 4. prepare smart morphems
./get_smart_morphems.sh ${DICT_VOCAB_FREQ_PATH} ${DICT_SMART_MORPHS_PATH} ${DICT_SMART_MORPHS_INFO_PATH};
echo "smart morphems prepared"


# 5. prepare synonyms
./get_syns.sh ${DICT_VOCAB_FREQ_PATH} ${FILE_SYNS_PATH} ${DICT_SYNS_PATH} ${DICT_SYN_INFO_PATH};
echo "synonyms prepared"
