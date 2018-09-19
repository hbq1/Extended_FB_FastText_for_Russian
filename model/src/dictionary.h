/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 *
 * Modified by: https://github.com/hbq1
 */

#ifndef FASTTEXT_DICTIONARY_H
#define FASTTEXT_DICTIONARY_H

#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include <map>
#include <unordered_map>
#include <unordered_set>

#include "args.h"
#include "real.h"

namespace fasttext {

typedef int32_t id_type;

struct lexem_info_t {
  int64_t freq_uniq;
  int64_t freq_full;
  int32_t zipf_rate;
  lexem_info_t(int64_t _freq_uniq, int64_t _freq_full) {
    freq_uniq = _freq_uniq;
    freq_full = _freq_full;
    zipf_rate = 0;
  }
};

struct source_lexem_info_t {
  std::unordered_map<int32_t, lexem_info_t> lexems;
  int64_t sum_freq_uniq;
  int64_t sum_freq_full;

  source_lexem_info_t() { sum_freq_uniq = sum_freq_full = 0; }

  bool containsLexem(const int32_t h) const {
    return lexems.find(h) != lexems.end();
  }

  int32_t size() const { return lexems.size(); }
};

struct word_info_t {
  std::map<std::string, std::vector<int32_t>> source_lexems;
  std::unordered_set<int32_t> synonyms;
  std::vector<std::vector<int32_t>> lexems;

  std::string word;
  int64_t freq;
  int32_t zipf_rate;
  word_info_t(const std::string& _word, int64_t _freq, int32_t _zipf_rate) {
    freq = _freq;
    word = _word;
    zipf_rate = _zipf_rate;
  }

  bool isInSource(const std::string& name) {
    return source_lexems.find(name) != source_lexems.end();
  }

  std::vector<int32_t>& getLexemsFromSource(const std::string& name) {
    return source_lexems[name];
  }
};

struct context_info_t {
  int32_t w_ind;
  real score;
  context_info_t(const int32_t _w_ind, const real _score)
      : score(_score), w_ind(_w_ind) {}
};

class Dictionary {
 private:
  //    static const int32_t MAX_VOCAB_SIZE = 50*1000*1000;
  static const int32_t MAX_LINE_SIZE = 1024;

  void initTableDiscard();
  void initLexems();
  void initNSCounts();

  Args args_;
  std::vector<word_info_t> words_;
  std::unordered_map<std::string, int32_t> word2index_;

  std::vector<std::string> lexems_;
  std::unordered_map<std::string, int32_t> lexem2index_;

  std::unordered_map<std::string, source_lexem_info_t> dict_src_lexem_;

  std::vector<std::vector<context_info_t>> words_context;

  std::vector<lexem_ns_record> lexems_ns_counts_;

  std::vector<real> pdiscard_;

  std::string main_lexems_src_name;

  void loadSynonyms(const std::string&);
  void loadSource(const std::string, const std::string&, const std::string&);
  void loadWordsVocabulary(const std::string&);
  void loadSourceWordLexems(const std::string&, const std::string&,
                            int32_t = 20);
  void loadSourceLexemsInfo(const std::string&, const std::string&);
  void sortSourceLexems(const std::string&);
  void filterSourceByFreq(const std::string&, const real, const real,
                          const int32_t);
  void shrinkLexemsDict();
  void shrinkContexts(const real = 0.1);

  real getContextScore(const int32_t, const int32_t) const;
  void loadContextCooccurences(const std::string&);

  void addWord(const std::string&);
  int32_t getLexemIndex(const std::string&) const;

 public:
  void addLexemToIndex(const std::string&);
  static const std::string EOS;
  static const std::string BOW;
  static const std::string EOW;

  int32_t nwords;
  int32_t nlexems;
  int64_t ntokens;
  int32_t cnt_sources;
  int64_t sum_freq_words_full;
  int64_t sum_freq_words_uniq;

  explicit Dictionary(std::shared_ptr<Args>);

  int32_t getWordIndex(const std::string&) const;
  std::string getWord(int32_t) const;
  bool isWordInVocab(const std::string&) const;
  bool isWordInVocab(const int32_t) const;
  bool isLexemInSource(const int32_t id) const;

  bool isWordsCorrelated(const int32_t, const int32_t) const;
  bool isSynonyms(const int32_t, const int32_t) const;
  const std::vector<std::vector<int32_t>>& getWordLexems(int32_t) const;

  void getLexemsStrings(const std::vector<int32_t>&,
                        std::vector<std::string>&) const;
  real getWordWeight(const int32_t) const;
  real getLexemWeight(const int32_t) const;

  uint32_t hash(const std::string&) const;

  bool readWord(std::istream&, std::string&) const;
  void readFromFile(std::istream&);
  int32_t getLine(std::istream&, std::vector<int32_t>&,
                  std::minstd_rand&) const;

  const std::vector<lexem_ns_record>& getNSCounts() const;
  bool tryDiscard(int32_t, real) const;
  //    void save(std::ostream&) const; //?
  //    void load(std::istream&); //?
};

}  // namespace fasttext

#endif
