CORPUS_PATH=$1
VOCAB_PATH=$2
FREQ_VOCAB_PATH=$3
TOP_N_WORDS=$4


cat ${CORPUS_PATH} | tr '\ ' '\n' | grep -v -PR "[a-zA-Z]" | grep -x '.\{3,\}' |  LC_ALL=c sort | uniq -c | LC_ALL=c sort -nr | head -n${TOP_N_WORDS} > ${FREQ_VOCAB_PATH};

cat ${FREQ_VOCAB_PATH} | sed 's/.*\ //' > ${VOCAB_PATH};