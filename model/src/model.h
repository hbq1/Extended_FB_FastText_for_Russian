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

#ifndef FASTTEXT_MODEL_H
#define FASTTEXT_MODEL_H

#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "args.h"
#include "dictionary.h"
#include "matrix.h"
#include "real.h"
#include "vector.h"

#define SIGMOID_TABLE_SIZE 10000
#define MAX_SIGMOID 8
#define LOG_TABLE_SIZE 10000

namespace fasttext {

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t count;
  bool binary;
};

const real POW_DISCARD = 0.75;

struct matrix_buffer_t {
  Vector vector;
  int32_t row;
  real alpha;
  matrix_buffer_t(const Vector& v, int32_t r, real a)
      : vector(v), row(r), alpha(a) {}
};

struct StateBuffer {
  std::vector<matrix_buffer_t> bufferWI, bufferWO;

  void addWO(const Vector& vector, int32_t row, real alpha) {
    bufferWO.push_back(matrix_buffer_t(vector, row, alpha));
  }

  void addWI(const Vector& vector, int32_t row, real alpha) {
    bufferWI.push_back(matrix_buffer_t(vector, row, alpha));
  }

  void applyBuffer(Matrix& m, std::vector<matrix_buffer_t>& buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
      m.addRow(buf[i].vector, buf[i].row, buf[i].alpha);
    }
    buf.clear();
  }

  void applyMeanBuffer(Matrix& m, std::vector<matrix_buffer_t>& buf) {
    std::map<int32_t, int32_t> c;
    for (size_t i = 0; i < buf.size(); ++i) {
      if (c.find(buf[i].row) == c.end()) {
        c.insert(std::make_pair(buf[i].row, 1));
      } else {
        c[buf[i].row]++;
      }
    }

    for (size_t i = 0; i < buf.size(); ++i) {
      m.addRow(buf[i].vector, buf[i].row, buf[i].alpha / c[buf[i].row]);
    }
    buf.clear();
  }

  void applyWI(Matrix& m) { applyBuffer(m, bufferWI); }

  void applyMeanWI(Matrix& m) { applyMeanBuffer(m, bufferWI); }

  void applyWO(Matrix& m) { applyBuffer(m, bufferWI); }

  void applyMeanWO(Matrix& m) { applyMeanBuffer(m, bufferWO); }
};

class Model {
 private:
  std::shared_ptr<Matrix> wi_;
  std::shared_ptr<Matrix> wo_;
  std::shared_ptr<Args> args_;
  StateBuffer state_buffer_;
  Vector hidden_;
  Vector output_;
  Vector grad_;
  int32_t hsz_;
  int32_t isz_;
  int32_t osz_;
  long_real loss_;
  long_real prev_loss_;
  int64_t nexamples_;
  int64_t nexamples_batch;
  real* t_sigmoid;
  real* t_log;
  std::shared_ptr<Dictionary> dict_;

  std::vector<int32_t> negatives;
  size_t negpos;

  static bool comparePairs(const std::pair<real, int32_t>&,
                           const std::pair<real, int32_t>&);

  int32_t getNegative(const int32_t);

  void initSigmoid();
  void initLog();

  static const int32_t NEGATIVE_TABLE_SIZE = 10 * 1000 * 1000;

 public:
  Model(std::shared_ptr<Matrix>, std::shared_ptr<Matrix>, std::shared_ptr<Args>,
        std::shared_ptr<Dictionary>, int32_t);
  ~Model();

  real binaryLogistic(const int32_t, bool, const real, bool = false);
  real negativeSampling(const int32_t, const real, bool = false);
  void doGradientStep();
  void doGradientStepMean();

  void normalizeModel();
  void update(const std::vector<int32_t>&, const int32_t, const real,
              bool = false);
  void computeHidden(const std::vector<int32_t>&, Vector&) const;

  void setTargetCounts(const std::vector<lexem_ns_record>&);
  void initTableNegatives(const std::vector<lexem_ns_record>&);
  long_real getLoss();
  real sigmoid(real) const;
  real log(real) const;

  std::minstd_rand rng;
};

}  // namespace fasttext

#endif
