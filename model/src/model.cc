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

Model::Model(std::shared_ptr<Matrix> wi, std::shared_ptr<Matrix> wo,
             std::shared_ptr<Args> args, std::shared_ptr<Dictionary> dict,
             int32_t seed)
    : hidden_(args->dim),
      output_(wo->m_),
      grad_(args->dim),
      dict_(dict),
      rng(seed) {
  wi_ = wi;
  wo_ = wo;
  args_ = args;
  isz_ = wi->m_;
  osz_ = wo->m_;
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  prev_loss_ = 0.0;
  nexamples_ = 1;
  nexamples_batch = 0;
  initSigmoid();
  initLog();
}

Model::~Model() {
  delete[] t_sigmoid;
  delete[] t_log;
}

void Model::normalizeModel() {
  real c1 = wo_->max();
  real c2 = wi_->max();
  if (c2 > c1) c1 = c2;
  c1 = 1.0 / c1;
  std::cerr << std::setprecision(10) << c1 << std::endl;
  if (c1 < 1e-2) {
    wo_->mulMatrix(c1);
    wi_->mulMatrix(c1);
  }
}

void Model::doGradientStep() {
  state_buffer_.applyWO(*wo_);
  state_buffer_.applyWI(*wi_);
}

void Model::doGradientStepMean() {
  state_buffer_.applyMeanWO(*wo_);
  state_buffer_.applyMeanWI(*wi_);
}

real Model::binaryLogistic(const int32_t target, bool label, const real lr,
                           bool use_buffer) {
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = lr * (real(label) - score);

  //  grad_.addRow(*wi_, target, alpha);

  grad_.addRow(*wo_, target, alpha);
  if (!use_buffer) {
    wo_->addRow(hidden_, target, alpha);
  } else {
    state_buffer_.addWO(hidden_, target, alpha);
    //  state_buffer_.applyWO(*wo_);
  }

  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::negativeSampling(const int32_t target, const real lr,
                             bool use_buff) {
  real loss = 0.0;
  std::vector<int32_t> was;
  for (int32_t n = 0; n <= args_->neg;) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr, use_buff);
      was.push_back(target);
      n++;
    } else {
      int32_t negative = getNegative(target);
      if (std::find(was.begin(), was.end(), negative) == was.end()) {
        loss += binaryLogistic(getNegative(target), false, lr, use_buff);
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
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

bool Model::comparePairs(const std::pair<real, int32_t>& l,
                         const std::pair<real, int32_t>& r) {
  return l.first > r.first;
}

void Model::update(const std::vector<int32_t>& input, const int32_t target,
                   const real lr, const bool use_buff) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0) return;
  computeHidden(input, hidden_);
  grad_.zero();
  loss_ += negativeSampling(target, lr, use_buff);
  nexamples_ += 1;
  nexamples_batch += 1;

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    if (!use_buff) {
      wi_->addRow(grad_, *it, 1.0);
    } else {
      state_buffer_.addWI(grad_, *it, 1.0);
      //	state_buffer_.applyWI(*wi_);
    }
  }
}

void Model::setTargetCounts(const std::vector<lexem_ns_record>& counts) {
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
}

void Model::initTableNegatives(const std::vector<lexem_ns_record>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); ++i) {
    z += pow(counts[i].cnt, POW_DISCARD);
  }
  for (size_t i = 0; i < counts.size(); ++i) {
    real c = pow(counts[i].cnt, POW_DISCARD);
    int64_t cnt = (c / z) * NEGATIVE_TABLE_SIZE;
    for (size_t j = 0; j < cnt; ++j) {
      negatives.push_back(counts[i].h);
    }
  }
  std::shuffle(negatives.begin(), negatives.end(), rng);
}

int32_t Model::getNegative(const int32_t target) {
  int32_t negative;
  do {
    negative = negatives[negpos];
    negpos = (negpos + 1) % negatives.size();
  } while (target == negative);
  //  } while (target == negative || dict_->isWordsCorrelated(target,
  //  negative));
  //} while (target == negative || dict_->isSynonyms(target, negative));

  return negative;
}

long_real Model::getLoss() {
  if (nexamples_batch > 0) {
    prev_loss_ = ((long_real)(nexamples_ - nexamples_batch) / (nexamples_)) *
                     prev_loss_ +
                 loss_ / nexamples_;
    nexamples_batch = 0;
    loss_ = 0.0;
  }
  return prev_loss_;
  return loss_ / nexamples_;
}

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
    return 1e-8;
  } else if (x > MAX_SIGMOID) {
    return 1.0 - 1e-8;
  } else {
    int i = int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid[i];
  }
}

}  // namespace fasttext
