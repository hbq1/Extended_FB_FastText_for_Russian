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

#include "model.h"

#include <assert.h>

#include <algorithm>

namespace fasttext {

Model::Model(std::vector<std::shared_ptr<Matrix>>& vec_wi,
             std::shared_ptr<Matrix> wo, std::shared_ptr<Args> args,
             int32_t seed)
    : hidden_(args->dim), output_(wo->m_), grad_(args->dim), rng(seed) {
  vec_wi_ = vec_wi;
  wo_ = wo;
  args_ = args;
  isz_ = wi->m_;
  osz_ = wo->m_;
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;

  cnt_wi_ = vec_wi.size();
  assert(cnt_wi_ > 0);

  initSigmoid();
  initLog();
}

Model::~Model() {
  delete[] t_sigmoid;
  delete[] t_log;
}

real Model::binaryLogistic(int32_t target, bool label, real lr) {
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);
  grad_.addRow(*wo_, target, alpha);
  wo_->addRow(hidden_, target, alpha);
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::negativeSampling(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  std::vector<int32_t> was;
  for (int32_t n = 0; n <= args_->neg;) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr);
      was.push_back(target);
      n++;
    } else {
      int32_t negative = getNegative(target);
      if (std::find(was.begin(), was.end(), negative) == was.end()) {
        loss += binaryLogistic(getNegative(target), false, lr);
        was.push_back(negative);
        n++;
      }
    }
  }
  return loss;
}

void Model::computeHidden(const std::vector<int32_t>& input,
                          Vector& hidden) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    for (size_t i = 0; i < cnt_wi_; ++i) hidden.addRow(*(vec_wi_[i]), *it);
  }
  hidden.mul(1.0 / (input.size() * cnt_wi_));
}

bool Model::comparePairs(const std::pair<real, int32_t>& l,
                         const std::pair<real, int32_t>& r) {
  return l.first > r.first;
}

void Model::update(const std::vector<int32_t>& input, int32_t target, real lr) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;
  computeHidden(input, hidden_);
  loss_ += negativeSampling(target, lr);
  nexamples_ += 1;

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    for (size_t i = 0; i < cnt_wi_; ++i) vec_wi_[i]->addRow(grad_, *it, 1.0);
  }
}

void Model::setTargetCounts(const std::vector<lexem_ns_record>& counts) {
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
}

void Model::initTableNegatives(const std::vector<lexem_ns_record>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i].cnt, POW_DISCARD);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i].cnt, POW_DISCARD);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives.push_back(counts[i].h);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  return negative;
}

real Model::getLoss() const { return loss_ / nexamples_; }

void Model::initSigmoid() {
  t_sigmoid = new real[SIGMOID_TABLE_SIZE + 1];
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid[i] = 1.0 / (1.0 + std::exp(-x));
  }
}

void Model::initLog() {
  t_log = new real[LOG_TABLE_SIZE + 1];
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log[i] = std::log(x);
  }
}

real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int i = int(x * LOG_TABLE_SIZE);
  return t_log[i];
}

real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

}  // namespace fasttext
