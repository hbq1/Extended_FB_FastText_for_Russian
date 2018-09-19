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

#include "matrix.h"

#include <assert.h>
#include <random>

#include "utils.h"
#include "vector.h"

namespace fasttext {

Matrix::Matrix() {
  m_ = 0;
  n_ = 0;
  data_ = nullptr;
}

Matrix::Matrix(int64_t m, int64_t n) {
  m_ = m;
  n_ = n;
  data_ = new real[m * n];
}

Matrix::Matrix(const Matrix& other) {
  m_ = other.m_;
  n_ = other.n_;
  data_ = new real[m_ * n_];
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = other.data_[i];
  }
}

Matrix& Matrix::operator=(const Matrix& other) {
  Matrix temp(other);
  m_ = temp.m_;
  n_ = temp.n_;
  std::swap(data_, temp.data_);
  return *this;
}

Matrix::~Matrix() { delete[] data_; }

void Matrix::zero() {
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = 0.0;
  }
}

void Matrix::uniform(real a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = uniform(rng);
  }
  //  for (int64_t i = 0; i < m_; ++i)
  //	normalizeRow(i);
}

void Matrix::mulRow(int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] *= a;
  }
}

void Matrix::mulMatrix(const real a) {
  for (int64_t i = 0; i < (m_ * n_); ++i) {
    data_[i] *= a;
  }
}

real Matrix::max() const {
  real a = 0;
  for (int64_t i = 0; i < (m_ * n_); ++i) {
    if (data_[i] > a) a = data_[i];
  }
  return a;
}

void Matrix::normalizeRow(int64_t i) {
  assert(i >= 0);
  assert(i < m_);
  real norm = rowNorm(i);
  for (int64_t j = 0; j < n_; j++) data_[i * n_ + j] /= norm;
}

real Matrix::rowNorm(int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  real norm = 1e-10;
  for (int64_t j = 0; j < n_; j++)
    norm += data_[i * n_ + j] * data_[i * n_ + j];
  return pow(norm, 0.5);
}

void Matrix::addRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.m_ == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec.data_[j];
  }
}

real Matrix::dotRow(const Vector& vec, int64_t i) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.m_ == n_);
  real d = 0.0;
  real div = 1.0;  //(rowNorm(i) * vec.norm());
  for (int64_t j = 0; j < n_; j++) {
    d += (data_[i * n_ + j] * vec.data_[j] / div);
    //	if (d > 1e6) d /= div;
  }
  return d;
}

void Matrix::save(std::ostream& out) {
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  out.write((char*)data_, m_ * n_ * sizeof(real));
}

void Matrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  delete[] data_;
  data_ = new real[m_ * n_];
  in.read((char*)data_, m_ * n_ * sizeof(real));
}

}  // namespace fasttext
