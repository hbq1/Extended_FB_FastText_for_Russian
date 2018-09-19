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

#ifndef FASTTEXT_FASTTEXT_H
#define FASTTEXT_FASTTEXT_H

#include <time.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "model.h"
#include "real.h"
#include "utils.h"
#include "vector.h"

namespace fasttext {

class FastText {
 private:
  std::shared_ptr<Args> args_;
  std::shared_ptr<Dictionary> dict_;

  std::shared_ptr<Matrix> input_;
  std::shared_ptr<Matrix> output_;
  std::shared_ptr<Model> main_model_;

  std::vector<std::shared_ptr<Matrix>> inputs_;
  std::vector<std::shared_ptr<Matrix>> outputs_;
  std::vector<std::shared_ptr<Model>> models_;
  std::atomic<int64_t> tokenCount;
  clock_t start;

  std::atomic<int32_t> logging_thread;
  std::atomic<int32_t> cnt_threads;
  std::atomic<int32_t> cnt_active_threads;
  std::atomic<int32_t> commonSteps;
  int32_t maxSteps;

  std::mutex normalizer_mutex;

 public:
  void getVector(Vector&, const std::string&);

  void saveVectors();
  void saveOutput();
  void saveModel();
  void loadModel(const std::string&, std::shared_ptr<Args> args = nullptr);
  void loadModel(std::istream&, std::shared_ptr<Args> args = nullptr);
  void printInfo(real, long_real, real);

  void supervised(Model&, real, const std::vector<int32_t>&,
                  const std::vector<int32_t>&);
  void cbow(Model&, real, const std::vector<int32_t>&);
  void skipgram(Model&, real, const std::vector<int32_t>&);
  void test(std::istream&, int32_t);
  void predict(std::istream&, int32_t, bool);
  void predict(std::istream&, int32_t,
               std::vector<std::pair<real, std::string>>&) const;
  void wordVectors();
  void lexemVectors(std::string);
  void textVectors();
  void printVectors();
  void trainThread(int32_t);
  void train(std::shared_ptr<Args>);

  void loadVectors(std::string);
};

}  // namespace fasttext

#endif
