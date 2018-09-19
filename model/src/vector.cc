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

#include "vector.h"

#include <assert.h>
#include <cmath>

#include <iomanip>

#include "matrix.h"

namespace fasttext {

Vector::Vector(int64_t m) {
  m_ = m;
  n_copies_ = new int32_t;
  *n_copies_ = 1;
  data_ = new real[m];
}

Vector::Vector(const Vector& v) {
  m_ = v.m_;
  data_ = v.data_;
  n_copies_ = v.n_copies_;
  (*n_copies_)++;
  //	data_ = new real[m_];
  //	for (size_t i=0; i<m_; ++i) data_[i] = v.data_[i];
}

Vector::~Vector() {
  (*n_copies_)--;
  if ((*n_copies_) == 0) {
    delete[] data_;
    delete n_copies_;
  }
}

int64_t Vector::size() const { return m_; }

void Vector::zero() {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
  }
}

void Vector::mul(real a) {
  for (int64_t i = 0; i < m_; i++) {
    data_[i] *= a;
  }
}

void Vector::l2_normalize() {
  real sum = 1e-8;
  for (int64_t i = 0; i < m_; i++) {
    sum += data_[i] * data_[i];
  }
  sum = pow(sum, 0.5);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] /= sum;
  }
}

real Vector::norm() const {
  real norm = 1e-8;
  for (int64_t i = 0; i < m_; i++) {
    norm += data_[i] * data_[i];
  }
  return pow(norm, 0.5);
}


void Vector::l1_normalize() {
  real sum = 1e-8;
  for (int64_t i = 0; i < m_; i++) {
        sum += (data_[i] < 0 ? -data_[i]: data_[i]);
  }
  for (int64_t i = 0; i < m_; i++) {
        data_[i] /= sum;
  }
}


void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += A.data_[i * A.n_ + j];
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.m_);
  assert(m_ == A.n_);
  for (int64_t j = 0; j < A.n_; j++) {
    data_[j] += a * A.data_[i * A.n_ + j];
  }
}

void Vector::mul(const Matrix& A, const Vector& vec) {
  assert(A.m_ == m_);
  assert(A.n_ == vec.m_);
  for (int64_t i = 0; i < m_; i++) {
    data_[i] = 0.0;
    for (int64_t j = 0; j < A.n_; j++) {
      data_[i] += A.data_[i * A.n_ + j] * vec.data_[j];
    }
  }
}

int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < m_; i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

real& Vector::operator[](int64_t i) { return data_[i]; }

const real& Vector::operator[](int64_t i) const { return data_[i]; }

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.m_; j++) {
    os << v.data_[j] << ' ';
  }
  return os;
}

}  // namespace fasttext
