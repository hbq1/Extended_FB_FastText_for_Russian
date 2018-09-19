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

#ifndef FASTTEXT_ARGS_H
#define FASTTEXT_ARGS_H

#include <istream>
#include <map>
#include <ostream>
#include <string>

namespace fasttext {

enum class model_name : int { cbow = 1, sg, sup };
enum class loss_name : int { hs = 1, ns, softmax };

struct lexem_ns_record {
  int32_t h;
  int64_t cnt;
  lexem_ns_record(int32_t _h, int64_t _cnt) : h(_h), cnt(_cnt) {}
};

struct source_info_t {
  std::string path;
  std::string lexems_info_path;
  source_info_t(std::string p1, std::string p2)
      : path(p1), lexems_info_path(p2) {}
};

class Args {
 public:
  Args();
  std::string input;
  std::string test;
  std::string output;

  std::string log_path;
  std::string pretrainedModel;
  std::string context_cooccurences_path;

  std::map<std::string, source_info_t> dict_source_path;

  std::string dict_vocab_freq_path;

  double lr;
  int lrUpdateRate;
  int dim;
  int ws;
  int epoch;
  int minCount;
  int minCountLabel;
  int neg;
  int wordNgrams;
  loss_name loss;
  model_name model;
  int bucket;
  int minn;
  int maxn;
  int thread;
  double t;
  std::string label;
  int verbose;
  std::string pretrainedVectors;
  int saveOutput;

  void parseArgs(int, char**);
  void printHelp();
  void save(std::ostream&);
  void load(std::istream&);
};

}  // namespace fasttext

#endif
