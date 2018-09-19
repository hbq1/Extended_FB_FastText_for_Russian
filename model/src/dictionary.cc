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

#include "dictionary.h"

#include <assert.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <unordered_map>

namespace fasttext {

const std::string Dictionary::EOS = "</s>";
const std::string Dictionary::BOW = "<";
const std::string Dictionary::EOW = ">";

Dictionary::Dictionary(std::shared_ptr<Args> args) {
  args_ = *args;

  std::cerr << "---------------------\npreparing dictionary...\n\n";

  loadWordsVocabulary(args_.dict_vocab_freq_path);
  ntokens = sum_freq_words_full;
  //  loadContextCooccurences(args_.context_cooccurences_path);

  for (auto it = args_.dict_source_path.begin();
       it != args_.dict_source_path.end(); it++) {
    loadSource(it->first, it->second.path, it->second.lexems_info_path);
  }

  for (auto& word : words_) {
    int32_t word_ind = getWordIndex(word.word);
    //	std::vector<std::string> names = {"ngram", "morph", "smart_morph",
    //"syns_RT", "analogy"};
    std::vector<std::string> names = {"ngram", "morph", "smart_morph"};

    //, "contexts"};
    // ngram, morph, smart_morph, analogy
    cnt_sources = names.size() + 1;

    word.lexems.resize(cnt_sources);
    // base
    if (word_ind >= 0) word.lexems[0].push_back(word_ind);

    for (size_t i = 0; i < names.size(); ++i) {
      int32_t cnt = 0;
      std::vector<int32_t>& buf_v = word.source_lexems[names[i]];
      for (size_t j = 0; j < buf_v.size() && cnt < 10; ++j) {
        word.lexems[i + 1].push_back(buf_v[j]);
        cnt++;
      }
    }
  }
  //  shrinkContexts(0.1);
  shrinkLexemsDict();
  loadSynonyms("syns_RT");
  nlexems = lexems_.size();
  std::cerr << "words: " << nwords << ", lexems: " << nlexems << std::endl;
  std::cerr << "dictionary prepared!\n---------------------\n\n";

  initTableDiscard();
  initNSCounts();
}

void Dictionary::loadSynonyms(const std::string& src_name) {
  for (size_t i = 0; i < words_.size(); ++i) {
    const auto& src = words_[i].source_lexems[src_name];
    words_[i].synonyms.clear();
    for (size_t j = 0; j < src.size(); ++j) words_[i].synonyms.insert(src[j]);
  }
}

bool Dictionary::isSynonyms(const int32_t w_1, const int32_t w_2) const {
  assert(w_1 < nwords);
  assert(w_1 >= 0);
  return (words_[w_1].synonyms.find(w_2) == words_[w_1].synonyms.end()) ? 0 : 1;
}

bool Dictionary::isWordsCorrelated(const int32_t w_1, const int32_t w_2) const {
  real score = getContextScore(w_1, w_2);
  // bool score = !isSynonyms(w_1, w_2);
  return score > 0;
}

void Dictionary::shrinkContexts(const real threshold) {
  std::cerr << "shrinking contexts...\n";
  int32_t cnt_removed = 0;
  for (size_t i = 0; i < words_context.size(); ++i) {
    auto it = std::remove_if(
        words_context[i].begin(), words_context[i].end(),
        [threshold](const context_info_t& a) { return a.score < threshold; });
    cnt_removed += std::distance(it, words_context[i].end());
    words_context[i].erase(it, words_context[i].end());
  }
  std::cerr << "contexts shrinked! " << cnt_removed << std::endl;
}

bool Dictionary::isLexemInSource(const int32_t id) const {
  for (const auto& source : dict_src_lexem_)
    if (source.second.containsLexem(id)) return true;
  return false;
}

real Dictionary::getContextScore(const int32_t w_1, const int32_t w_2) const {
  auto it = std::lower_bound(
      words_context[w_1].begin(), words_context[w_1].end(), w_2,
      [](const context_info_t& a, const int32_t b) { return a.w_ind < b; });
  return (it == words_context[w_1].end()) ? 0ll : (it->score);
}

void Dictionary::loadContextCooccurences(const std::string& path) {
  words_context.clear();
  words_context.resize(words_.size());
  words_context.shrink_to_fit();

  std::string word_1, word_2;
  int64_t cnt_ocur;
  std::ifstream in(path);
  if (!in.is_open()) {
    std::cerr << "contexts: bad path " << path << std::endl;
    exit(0);
  }
  std::cerr << "loading contexts...\n";

  while (in >> word_1 >> word_2 >> cnt_ocur) {
    int32_t w_1 = getWordIndex(word_1);
    int32_t w_2 = getWordIndex(word_2);
    if (w_1 != -1 && w_2 != -1) {
      real score = (cnt_ocur + .0) / words_[w_1].freq;
      words_context[w_1].push_back(context_info_t(w_2, score));
    }
  }

  for (size_t i = 0; i < words_context.size(); ++i) {
    std::sort(words_context[i].begin(), words_context[i].end(),
              [](const context_info_t& a, const context_info_t& b) {
                return a.w_ind < b.w_ind;
              });
    words_context[i].shrink_to_fit();
  }

  std::cerr << "contexts loaded!\n\n";
  in.close();
}

void Dictionary::shrinkLexemsDict() {
  std::cerr << "shrinking dicts...\n";
  int32_t nlexems = lexems_.size();
  lexems_.clear();
  lexems_.shrink_to_fit();

  std::vector<int32_t> remap(nlexems, -1);
  for (size_t w_ind = 0; w_ind < words_.size(); ++w_ind) {
    for (size_t i = 0; i < words_[w_ind].lexems.size(); ++i) {
      for (size_t j = 0; j < words_[w_ind].lexems[i].size(); ++j)
        remap[words_[w_ind].lexems[i][j]] = 1;
    }
  }

  int32_t last_ind = 0;
  for (auto it_l = lexem2index_.begin(); it_l != lexem2index_.end(); ++it_l) {
    if (remap[it_l->second] == 1) {
      remap[it_l->second] = last_ind++;
      lexems_.push_back(it_l->first);
    } else {
      remap[it_l->second] = -1;
    }
  }
  lexems_.shrink_to_fit();
  for (auto it_s = dict_src_lexem_.begin(); it_s != dict_src_lexem_.end();
       ++it_s) {
    auto lexems(it_s->second.lexems);
    it_s->second.lexems.clear();
    for (auto it_l = lexems.begin(); it_l != lexems.end(); ++it_l) {
      if (remap[it_l->first] != -1)
        it_s->second.lexems.insert(
            std::make_pair(remap[it_l->first], it_l->second));
    }
  }

  for (int32_t w_ind = 0; w_ind < words_.size(); ++w_ind) {
    for (auto it_s = words_[w_ind].source_lexems.begin();
         it_s != words_[w_ind].source_lexems.end(); ++it_s) {
      auto& lexems = it_s->second;
      lexems.erase(
          std::remove_if(lexems.begin(), lexems.end(),
                         [&remap](int32_t ind) { return remap[ind] == -1; }),
          lexems.end());

      for (size_t i = 0; i < lexems.size(); ++i) {
        lexems[i] = remap[lexems[i]];
      }
    }
  }

  for (size_t w_ind = 0; w_ind < words_.size(); ++w_ind) {
    for (size_t i = 0; i < words_[w_ind].lexems.size(); ++i) {
      for (size_t j = 0; j < words_[w_ind].lexems[i].size(); ++j)
        words_[w_ind].lexems[i][j] = remap[words_[w_ind].lexems[i][j]];
    }
  }

  lexem2index_.clear();
  for (size_t i = 0; i < last_ind; ++i)
    lexem2index_.insert(std::make_pair(lexems_[i], i));

  std::cerr << "dicts shrinked! " << nlexems << " -> " << last_ind << std::endl;
}

void Dictionary::loadSource(const std::string src_name,
                            const std::string& src_path,
                            const std::string& src_lexems_info_path) {
  std::cerr << "preparing " << src_name << "..." << std::endl;

  loadSourceLexemsInfo(src_name, src_lexems_info_path);
  loadSourceWordLexems(src_name, src_path);
  std::cerr << "loaded " << src_name << ' ' << dict_src_lexem_[src_name].size()
            << std::endl;

  sortSourceLexems(src_name);
  std::cerr << "sorted " << src_name << std::endl;

  filterSourceByFreq(src_name, 0.99, 0.01, 20);
  std::cerr << "filtered " << src_name << ' '
            << dict_src_lexem_[src_name].size() << std::endl;

  std::cerr << "prepared " << src_name << std::endl << std::endl;
}

void Dictionary::sortSourceLexems(const std::string& src_name) {
  const auto& lexems_info = dict_src_lexem_[src_name].lexems;
  for (auto& word_info : words_)
    if (word_info.isInSource(src_name)) {
      auto& lexems = word_info.source_lexems[src_name];
      std::sort(lexems.begin(), lexems.end(),
                [&lexems_info](const int32_t a, const int32_t b) {
                  auto it_a = lexems_info.find(a);
                  auto it_b = lexems_info.find(b);
                  return it_a->second.freq_full > it_b->second.freq_full ||
                         (it_a->second.freq_full == it_b->second.freq_full &&
                          it_a->second.freq_uniq > it_b->second.freq_uniq);
                });
    }
}

void Dictionary::filterSourceByFreq(const std::string& src_name,
                                    const real upper_quant,
                                    const real lower_quant,
                                    const int32_t threshold) {
  auto& lexems_info = dict_src_lexem_[src_name].lexems;
  const int32_t dict_lexems_size = lexems_info.size();

  std::vector<int64_t> freqs;
  for (auto& lexem_info : lexems_info) {
    freqs.push_back(lexem_info.second.freq_full);
  }
  std::sort(freqs.begin(), freqs.end());
  int32_t upper_ind = std::floor(dict_lexems_size * upper_quant);
  if (upper_ind >= dict_lexems_size) upper_ind = dict_lexems_size - 1;
  int32_t lower_ind = std::floor(dict_lexems_size * lower_quant);
  if (lower_ind >= dict_lexems_size) lower_ind = dict_lexems_size - 1;
  int64_t thres_up = freqs[upper_ind];
  int64_t thres_lw = freqs[lower_ind];

  for (auto& word_info : words_) {
    if (word_info.isInSource(src_name)) {
      auto& lexems = word_info.source_lexems[src_name];

      lexems.erase(std::remove_if(
                       lexems.begin(), lexems.end(),
                       [thres_lw, thres_up, &lexems_info](const int32_t lexem) {
                         auto it_lexem = lexems_info.find(lexem);
                         return it_lexem->second.freq_full <= thres_lw ||
                                it_lexem->second.freq_full >= thres_up;
                       }),
                   lexems.end());

      if (lexems.size() > threshold) {
        lexems.erase(lexems.begin() + threshold, lexems.end());
      }
    }
  }

  for (auto it = lexems_info.begin(); it != lexems_info.end();) {
    if (it->second.freq_full <= thres_lw || it->second.freq_full >= thres_up)
      it = lexems_info.erase(it);
    else
      ++it;
  }
}

void Dictionary::loadWordsVocabulary(const std::string& vocab_path) {
  std::string word;
  int64_t freq;
  std::ifstream in(vocab_path);
  if (!in.is_open()) {
    std::cerr << "vocabulary: bad path " << vocab_path << std::endl;
    exit(0);
  }

  std::vector<std::string> words;
  std::vector<int64_t> freqs;
  std::vector<size_t> indices;
  sum_freq_words_full = sum_freq_words_uniq = 0;
  while (in >> freq >> word) {
    words.push_back(word);
    freqs.push_back(freq);
    indices.push_back(indices.size());

    sum_freq_words_full += freq;
    sum_freq_words_uniq += 1;
  }

  std::sort(indices.begin(), indices.end(), [&freqs](size_t a, size_t b) {
    return freqs[a] > freqs[b] || freqs[a] == freqs[b] && a > b;
  });

  word2index_.clear();
  words_.reserve(words.size());
  for (size_t i = 0; i < words.size(); ++i) {
    word2index_.insert(std::make_pair(words[i], i));
    words_.push_back(word_info_t(words[i], freqs[i], i + 1));
    addLexemToIndex(words[i] + "_base");
  }

  nwords = words_.size();
  std::cerr << "vocabulary loaded, words " << nwords << "\n\n";
  in.close();
}

void Dictionary::loadSourceLexemsInfo(const std::string& src_name,
                                      const std::string& dict_info_path) {
  dict_src_lexem_.insert(std::make_pair(src_name, source_lexem_info_t()));
  source_lexem_info_t& src_lexems_info = dict_src_lexem_[src_name];
  auto& lexems_info = src_lexems_info.lexems;

  int64_t freq_uniq, freq_full;
  src_lexems_info.sum_freq_uniq = 0;
  src_lexems_info.sum_freq_full = 0;

  std::ifstream in(dict_info_path);
  if (!in.is_open()) {
    std::cerr << "source_lexems_info: bad path " << dict_info_path << std::endl;
    exit(0);
  }

  std::vector<int32_t> lexems;
  std::vector<int64_t> freqs;
  std::vector<size_t> indices;
  std::string lexem;

  while (in >> lexem >> freq_uniq >> freq_full) {
    lexem += "_" + src_name;
    addLexemToIndex(lexem);
    int32_t h = getLexemIndex(lexem);
    lexems_info.insert(std::make_pair(h, lexem_info_t(freq_uniq, freq_full)));
    lexems.push_back(h);
    freqs.push_back(freq_full);
    indices.push_back(indices.size());

    src_lexems_info.sum_freq_uniq += freq_uniq;
    src_lexems_info.sum_freq_full += freq_full;
  }

  // Zipf
  std::sort(indices.begin(), indices.end(),
            [&freqs](const size_t a, const size_t b) {
              return freqs[a] > freqs[b] || freqs[a] == freqs[b] && a > b;
            });

  for (size_t i = 0; i < lexems.size(); ++i) {
    auto it = lexems_info.find(lexems[i]);
    it->second.zipf_rate = i + 1;
  }

  in.close();
}

void Dictionary::loadSourceWordLexems(const std::string& src_name,
                                      const std::string& dict_word_lexem_path,
                                      int32_t threshold) {
  std::string word, lexem;
  std::vector<int32_t> lexems;

  std::ifstream in(dict_word_lexem_path);
  if (!in.is_open()) {
    std::cerr << "dict_lexems: bad path " << dict_word_lexem_path << std::endl;
    exit(0);
  }
  readWord(in, word);
  while (true) {
    if (lexem == EOS) {
      int32_t i = getWordIndex(word);
      if (i != -1) {
        words_[i].source_lexems.insert(std::make_pair(src_name, lexems));
      }
      if (!readWord(in, word)) break;
      lexems.clear();
    } else {
      if (lexem != "" && lexems.size() <= threshold)
        lexems.push_back(getLexemIndex(lexem + "_" + src_name));
    }
    readWord(in, lexem);
  }
  in.close();
}

int32_t Dictionary::getWordIndex(const std::string& word) const {
  auto it = word2index_.find(word);
  return it != word2index_.end() ? it->second : -1;
}

const std::vector<std::vector<int32_t>>& Dictionary::getWordLexems(
    int32_t id) const {
  assert(id >= 0);
  assert(id < nwords);
  return words_[id].lexems;
}

void Dictionary::getLexemsStrings(const std::vector<int32_t>& lexems,
                                  std::vector<std::string>& strings) const {
  strings.clear();
  for (auto h : lexems) {
    strings.push_back(lexems_[h]);
  }
}

std::string Dictionary::getWord(int32_t id) const {
  assert(id >= 0);
  assert(id < nwords);
  return words_[id].word;
}

bool Dictionary::tryDiscard(int32_t id, real rand) const {
  assert(id >= 0);
  assert(id < nwords);
  return rand > pdiscard_[id];
}

bool Dictionary::isWordInVocab(const std::string& word) const {
  return getWordIndex(word) != -1;
}

bool Dictionary::isWordInVocab(const int32_t id) const {
  return id >= 0 && id < nwords;
}

void Dictionary::addLexemToIndex(const std::string& lexem) {
  auto it = lexem2index_.find(lexem);
  if (it == lexem2index_.end()) {
    lexem2index_.insert(std::make_pair(lexem, lexems_.size()));
    lexems_.push_back(lexem);
  }
}

int32_t Dictionary::getLexemIndex(const std::string& lexem) const {
  auto it = lexem2index_.find(lexem);
  return it != lexem2index_.end() ? it->second : -1;
}

real Dictionary::getWordWeight(const int32_t ind) const {
  assert(ind >= 0);
  assert(ind < nwords);
  return 1.0 / words_[ind].zipf_rate;
}

real Dictionary::getLexemWeight(const int32_t ind) const {
  assert(ind >= 0);
  assert(ind < nlexems);
  for (auto it = dict_src_lexem_.begin(); it != dict_src_lexem_.end(); it++)
    if (it->second.containsLexem(ind)) {
      auto it_lex = it->second.lexems.find(ind);
      return 1.0 / it_lex->second.zipf_rate;
    }
  return 1.0 / nlexems;
}

uint32_t Dictionary::hash(const std::string& str) const {
  uint32_t h = 2166136261;
  for (size_t i = 0; i < str.size(); i++) {
    h = h ^ uint32_t(str[i]);
    h = h * 16777619;
  }
  return h;
}

bool Dictionary::readWord(std::istream& in, std::string& word) const {
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += EOS;
          return true;
        }
        continue;
      } else {
        if (c == '\n') sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

void Dictionary::readFromFile(std::istream& in) { throw "bad call"; }

void Dictionary::initTableDiscard() {
  pdiscard_.resize(nwords);
  for (size_t i = 0; i < nwords; i++) {
    real f = real(words_[i].freq) / real(sum_freq_words_full);
    pdiscard_[i] = sqrt(args_.t / f) + args_.t / f;
  }
}

void Dictionary::initNSCounts() {
  lexems_ns_counts_.clear();
  for (size_t i = 0; i < words_.size(); ++i) {
    lexems_ns_counts_.push_back(lexem_ns_record(i, words_[i].freq));
  }
  std::cerr << "negatives count: " << lexems_ns_counts_.size() << std::endl;
}

const std::vector<lexem_ns_record>& Dictionary::getNSCounts() const {
  return lexems_ns_counts_;  // shuffle???
}

int32_t Dictionary::getLine(std::istream& in, std::vector<int32_t>& words,
                            std::minstd_rand& rng) const {
  std::uniform_real_distribution<> uniform(0, 1);
  words.clear();
  if (in.eof()) {
    in.clear();
    in.seekg(std::streampos(0));
  }

  std::string token;
  int32_t ntokens = 0;
  while (readWord(in, token)) {
    int32_t id = getWordIndex(token);
    if (id >= 0) {
      ntokens++;
      if (!tryDiscard(id, uniform(rng))) words.push_back(id);
      if (words.size() > MAX_LINE_SIZE) break;
    }
    if (token == EOS) break;
  }
  return ntokens;
}

}  // namespace fasttext
