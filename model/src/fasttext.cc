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

#include "fasttext.h"

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fasttext {

void FastText::getVector(Vector& vec, const std::string& word) {
  int32_t id = dict_->getWordIndex(word);
  vec.zero();
  if (id > 0) {
    const auto& lexems = dict_->getWordLexems(id);
    int32_t cnt_lexems = 0;
    for (size_t i = 0; i < lexems.size(); ++i) {
      for (size_t j = 0; j < lexems[i].size(); ++j) {
        vec.addRow(*input_, lexems[i][j]);
        cnt_lexems++;
      }
    }

    if (cnt_lexems > 0) vec.mul(1.0 / cnt_lexems);
  } else {
    std::cerr << "word '" << word << "' not found" << std::endl;
  }
}

void FastText::saveVectors() {
  std::ofstream ofs(args_->output + ".vec");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords; i++) {
    std::string word = dict_->getWord(i);
    getVector(vec, word);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveOutput() {
  std::ofstream ofs(args_->output + ".vectors");
  if (!ofs.is_open()) {
    std::cerr << "Error opening file for saving vectors." << std::endl;
    exit(EXIT_FAILURE);
  }
  ofs << dict_->nwords << " " << args_->dim << std::endl;
  Vector vec(args_->dim);
  for (int32_t i = 0; i < dict_->nwords; i++) {
    std::string word = dict_->getWord(i);
    vec.zero();
    vec.addRow(*output_, i);
    ofs << word << " " << vec << std::endl;
  }
  ofs.close();
}

void FastText::saveModel() {
  std::cerr << "\nsaving model...\n";
  std::ofstream ofs(args_->output + ".bin", std::ofstream::binary);
  if (!ofs.is_open()) {
    std::cerr << "Model file cannot be opened for saving!" << std::endl;
    exit(EXIT_FAILURE);
  }
  //  args_->save(ofs);
  //  dict_->save(ofs);
  input_->save(ofs);
  output_->save(ofs);
  ofs.close();
  std::cerr << "model saved!\n\n";
}

void FastText::loadModel(const std::string& filename,
                         std::shared_ptr<Args> args) {
  std::ifstream ifs(filename, std::ifstream::binary);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  loadModel(ifs, args);
  ifs.close();
}

void FastText::loadModel(std::istream& in, std::shared_ptr<Args> args) {
  std::cerr << "\nloading model...\n";
  args_ = std::make_shared<Args>();
  if (args != nullptr) args_ = args;
  //  dict_ = std::make_shared<Dictionary>(args_);
  input_ = std::make_shared<Matrix>();
  output_ = std::make_shared<Matrix>();
  //  args_->load(in);
  //  dict_->load(in);
  input_->load(in);
  output_->load(in);
  main_model_ = std::make_shared<Model>(input_, output_, args_, dict_, 0);
  //  if (args_->model == model_name::sup) {
  //  main_model_->setTargetCounts(dict_->getCounts());
  //  } else {
  main_model_->setTargetCounts(dict_->getNSCounts());
  //  }
  std::cerr << "model loaded!\n\n";
}

void FastText::printInfo(real progress, long_real loss, real lr) {
  real t = real(clock() - start) / CLOCKS_PER_SEC;
  real wst = real(tokenCount) / t;
  int eta = int(t / progress * (1 - progress) / args_->thread);
  int etah = eta / 3600;
  int etam = (eta - etah * 3600) / 60;
  std::cerr << std::fixed;
  std::cerr << "\rProgress: " << std::setprecision(1) << 100 * progress << "%";
  std::cerr << "  words/sec/thread: " << std::setprecision(0) << wst;
  std::cerr << "  lr: " << std::setprecision(6) << lr;
  std::cerr << "  loss: " << std::setprecision(6) << loss;
  std::cerr << "  eta: " << etah << "h" << etam << "m ";
  std::cerr << std::flush;
}

void FastText::supervised(Model& model, real lr,
                          const std::vector<int32_t>& line,
                          const std::vector<int32_t>& labels) {
  /*
    if (labels.size() == 0 || line.size() == 0) return;
    std::uniform_int_distribution<> uniform(0, labels.size() - 1);
    int32_t i = uniform(model.rng);
    model.update(line, labels[i], lr);
  */
}

void FastText::cbow(Model& model, real lr, const std::vector<int32_t>& line) {
  /*
    std::vector<int32_t> bow;
    std::uniform_int_distribution<> uniform(1, args_->ws);
    for (int32_t w = 0; w < line.size(); w++) {
          int32_t boundary = uniform(model.rng);
          bow.clear();
          for (int32_t c = -boundary; c <= boundary; c++) {
            if (c != 0 && w + c >= 0 && w + c < line.size()) {
                  const std::vector<int32_t>& lexems = dict_->getLexems(line[w +
    c]); bow.insert(bow.end(), lexems.cbegin(), lexems.cend());
            }
          }
          model.update(bow, line[w], lr);
    }
   */
}

void FastText::skipgram(Model& model, real lr,
                        const std::vector<int32_t>& line) {
  std::uniform_int_distribution<> uniform(1, args_->ws);
  std::uniform_int_distribution<> uniform_lexems(0, 3);
  std::uniform_int_distribution<> one_out_rand(0, dict_->cnt_sources - 1);
  std::uniform_int_distribution<> dropout_rand(0, 1);

  for (int32_t w = 0; w < line.size(); w++) {
    const std::vector<std::vector<int32_t>>& all_lexems =
        dict_->getWordLexems(line[w]);
    std::vector<int32_t> lexems;
    std::vector<int32_t> lexems_2;
    std::vector<size_t> ind(all_lexems.size());
    for (size_t i = 0; i < all_lexems.size(); ++i) ind[i] = i;

    std::uniform_int_distribution<> exp_random(0, 3);
    int EXP_N = -1;  /// exp_random(model.rng);
    // 0 - full
    // 2 - dropout
    // 3 - mean_prior
    // 4 - semi-boost
    // -1 - random net
    if (EXP_N == -1) {
      EXP_N = exp_random(model.rng);
      if (EXP_N == 1) EXP_N = 4;
    }

    if (EXP_N == 3) {
      shuffle(ind.begin(), ind.end(), model.rng);
    }

    // 1 - out
    if (EXP_N == 5) {
      int32_t dropped = one_out_rand(model.rng);
      for (size_t src_i = 0; src_i < dict_->cnt_sources; ++src_i) {
        if (src_i == dropped) continue;
        for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k)
          lexems.push_back(all_lexems[ind[src_i]][k]);
      }
    }

    // full
    if (EXP_N == 0) {
      for (size_t src_i = 0; src_i < dict_->cnt_sources; ++src_i) {
        for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k)
          lexems.push_back(all_lexems[ind[src_i]][k]);
      }
    }

    // exclusion
    if (EXP_N == 1) {
      for (size_t src_i = 0; src_i < dict_->cnt_sources; ++src_i) {
        for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k)
          lexems.push_back(all_lexems[ind[src_i]][k]);
      }
    }

    // dropout
    if (EXP_N == 2) {
      for (size_t src_i = 0; src_i < dict_->cnt_sources; ++src_i) {
        if (dropout_rand(model.rng) == 1) {
          for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k) {
            lexems.push_back(all_lexems[ind[src_i]][k]);
          }
        } else {
          continue;
          for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k) {
            lexems_2.push_back(all_lexems[ind[src_i]][k]);
          }
        }
      }
    }
    //

    //  int32_t boundary = args_->ws;
    int32_t boundary = uniform(model.rng);
    for (int32_t c = -boundary; c <= boundary; c++) {
      if (c != 0 && w + c >= 0 && w + c < line.size()
          //		&& dict_->isWordsCorrelated(line[w], line[w+c])
      ) {
        if (EXP_N == 0 || EXP_N == 5) {
          model.update(lexems, line[w + c], lr);
        }
        if (EXP_N == 2) {
          model.update(lexems, line[w + c], lr * 0.5);
        }

        if (EXP_N == 3 || EXP_N == 1 || EXP_N == 4 || EXP_N == 6 ||
            EXP_N == 7) {
          for (size_t src_i = 0; src_i < dict_->cnt_sources; ++src_i) {
            // semi-RNN
            if (EXP_N == 6) {
              for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k)
                lexems.push_back(all_lexems[ind[src_i]][k]);
              for (size_t src_j = 1; src_j < dict_->cnt_sources; ++src_j) {
                if (src_j == src_i || all_lexems[ind[src_j]].size() == 0)
                  continue;
                lexems.push_back(all_lexems[ind[src_j]][0]);
              }
              model.update(lexems, line[w + c], lr);
              lexems.clear();
            }

            // gradboost
            if (EXP_N == 3) {
              for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k)
                lexems.push_back(all_lexems[ind[src_i]][k]);
              model.update(lexems, line[w + c], lr);
            }

            // exclusion
            if (EXP_N == 1) {
              if (all_lexems[ind[src_i]].size() > 0) {
                auto it = std::find(lexems.begin(), lexems.end(),
                                    all_lexems[ind[src_i]][0]);
                lexems.erase(it, it + all_lexems[ind[src_i]].size());
              }
              model.update(lexems, line[w + c], lr);
              for (size_t k = 0; k < all_lexems[ind[src_i]].size(); ++k)
                lexems.push_back(all_lexems[ind[src_i]][k]);
            }

            // exclusion-boost
            if (EXP_N == 4) {
              //		   for (size_t k = 0; k <
              // all_lexems[src_i].size(); ++k)
              //		     lexems.push_back(all_lexems[src_i][k] +
              // shift[src_i]);

              model.update(all_lexems[ind[src_i]], line[w + c], lr);
              lexems.clear();
            }

            // sync exclusion-boost
            if (EXP_N == 7) {
              model.update(all_lexems[ind[src_i]], line[w + c], lr, true);
            }
          }
          if (EXP_N == 7) {
            // model.doGradientStep();
            model.doGradientStepMean();
          }
        }
        //	    continue;
        /*
                    int32_t target_word = line[w + c];
                    if (dict_->isWordInVocab(target_word)) {
                      const auto& context_lexems =
           dict_->getWordLexems(target_word); int32_t cnt_lexems =
           uniform_lexems(model.rng); if (context_lexems[src_i].size() > 0) {
                        for (size_t j = 0; j < cnt_lexems %
           context_lexems[src_i].size(); ++j) { int32_t rev =
           context_lexems[src_i].size() - j - 1; if (rev == 0) continue; int32_t
           h = context_lexems[src_i][rev] + dict_->nwords; model.update(lexems,
           h, lr);
                        }
                      }
                    }
                */
      }
    }
  }
}

void FastText::test(std::istream& in, int32_t k) {
  /*
    int32_t nexamples = 0, nlabels = 0;
    double precision = 0.0;
    std::vector<int32_t> line, labels;

    while (in.peek() != EOF) {
      dict_->getLine(in, line, labels, model_->rng);
      dict_->addNgrams(line, args_->wordNgrams);
      if (labels.size() > 0 && line.size() > 0) {
        std::vector<std::pair<real, int32_t>> modelPredictions;
        model_->predict(line, k, modelPredictions);
        for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend();
    it++) { if (std::find(labels.begin(), labels.end(), it->second) !=
    labels.end()) { precision += 1.0;
          }
        }
        nexamples++;
        nlabels += labels.size();
      }
    }
    std::cerr << std::setprecision(3);
    std::cerr << "P@" << k << ": " << precision / (k * nexamples) << std::endl;
    std::cerr << "R@" << k << ": " << precision / nlabels << std::endl;
    std::cerr << "Number of examples: " << nexamples << std::endl;
   */
}

void FastText::predict(
    std::istream& in, int32_t k,
    std::vector<std::pair<real, std::string>>& predictions) const {
  /*
    std::vector<int32_t> words, labels;
    dict_->getLine(in, words, labels, model_->rng);
    dict_->addNgrams(words, args_->wordNgrams);
    if (words.empty()) return;
    Vector hidden(args_->dim);
    Vector output(dict_->nlabels());
    std::vector<std::pair<real,int32_t>> modelPredictions;
    model_->predict(words, k, modelPredictions, hidden, output);
    predictions.clear();
    for (auto it = modelPredictions.cbegin(); it != modelPredictions.cend();
    it++) { predictions.push_back(std::make_pair(it->first,
    dict_->getLabel(it->second)));
    }
  */
}

void FastText::predict(std::istream& in, int32_t k, bool print_prob) {
  /*
    std::vector<std::pair<real,std::string>> predictions;
    while (in.peek() != EOF) {
      predict(in, k, predictions);
      if (predictions.empty()) {
        std::cout << "n/a" << std::endl;
        continue;
      }
      for (auto it = predictions.cbegin(); it != predictions.cend(); it++) {
        if (it != predictions.cbegin()) {
          std::cout << ' ';
        }
        std::cout << it->second;
        if (print_prob) {
          std::cout << ' ' << exp(it->first);
        }
      }
      std::cout << std::endl;
    }
  */
}

void FastText::wordVectors() {
  std::string word;
  Vector vec(args_->dim);
  while (std::cin >> word) {
    getVector(vec, word);
    std::cout << word << " " << vec << std::endl;
  }
}

void FastText::lexemVectors(std::string word) {
  /*
    Vector vec(args_->dim);
    int32_t ind = dict_->getWordIndex(word);
    if (ind >= 0) {
          const std::vector<int32_t>& all_lexems = dict_->getWordLexems(ind);
          std::vector<std::string> lexems_str;
          dict_->getLexemsStrings(lexems, lexems_str);
      vec.zero();
          for (int32_t i = 0; i < lexems.size(); i++) {
            if (lexems[i] >= 0) {
          vec.addRow(*input_, lexems[i]);
        }
        std::cout << lexems_str[i] << " " << vec << std::endl;
      }
    }
   */
}

void FastText::textVectors() {
  /*
    std::vector<int32_t> line, labels;
    Vector vec(args_->dim);
    while (std::cin.peek() != EOF) {
      dict_->getLine(std::cin, line, labels, model_->rng);
      dict_->addNgrams(line, args_->wordNgrams);
      vec.zero();
      for (auto it = line.cbegin(); it != line.cend(); ++it) {
        vec.addRow(*input_, *it);
      }
      if (!line.empty()) {
        vec.mul(1.0 / line.size());
      }
      std::cout << vec << std::endl;
    }
  */
}

void FastText::printVectors() {
  if (args_->model == model_name::sup) {
    textVectors();
  } else {
    wordVectors();
  }
}

void FastText::trainThread(int32_t threadId) {
  cnt_active_threads++;
  std::ifstream ifs(args_->input);
  std::ofstream log_stream_lr;
  std::ofstream log_stream_ls;
  if (args_->log_path != "") {
    log_stream_lr.open(args_->log_path + "_lr_" + std::to_string(threadId));
    log_stream_ls.open(args_->log_path + "_ls_" + std::to_string(threadId));
  }

  utils::seek(ifs, threadId * utils::size(ifs) / args_->thread);

  Model model(input_, output_, args_, dict_, threadId);
  model.setTargetCounts(dict_->getNSCounts());

  const int64_t ntokens = dict_->ntokens;
  const int64_t local_ntokens = ntokens / args_->thread + 1;
  int64_t localTokenCount = 0;
  int64_t localTokenBuffer = 0;

  std::vector<int32_t> line;
  real min_loss = 1e10;
  real c = 1.0;
  const real EPS = 1e-8;
  int32_t steps = 0;
  while (localTokenCount < args_->epoch * local_ntokens) {
    real progress = real(localTokenCount) / (args_->epoch * local_ntokens);
    real lr = args_->lr * (1.0 - progress);  //* c;
                                             //	if (lr < EPS)
    //		tokenCount += args_->epoch * local_ntokens - localTokenCount;
    //		break;
    //	}

    localTokenBuffer += dict_->getLine(ifs, line, model.rng);

    skipgram(model, lr, line);
    if (localTokenBuffer > args_->lrUpdateRate) {
      long_real loss = model.getLoss();
      steps++;

      tokenCount += localTokenBuffer;
      localTokenCount += localTokenBuffer;
      localTokenBuffer = 0;

      if (threadId == logging_thread && args_->verbose > 1) {
        printInfo(progress, loss, lr);
      }
      if (args_->log_path != "") {
        log_stream_ls << std::setprecision(6) << loss << ' ';
        log_stream_lr << std::setprecision(6) << lr << ' ';
      }
    }
  }
  if (threadId == logging_thread && args_->verbose > 0) {
    printInfo(1.0, model.getLoss(), 0.0);
    std::cerr << std::endl;
  }
  if (args_->log_path != "") {
    log_stream_ls.close();
    log_stream_lr.close();
  }

  ifs.close();
  cnt_active_threads--;
}

void FastText::loadVectors(std::string filename) {
  std::ifstream in(filename);
  std::vector<std::string> words;
  std::shared_ptr<Matrix> mat;  // temp. matrix for pretrained vectors
  int64_t n, dim;
  if (!in.is_open()) {
    std::cerr << "Pretrained vectors file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  in >> n >> dim;
  if (dim != args_->dim) {
    std::cerr << "Dimension of pretrained vectors does not match -dim option"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  mat = std::make_shared<Matrix>(n, dim);
  for (size_t i = 0; i < n; i++) {
    std::string word;
    in >> word;
    words.push_back(word);
    dict_->addLexemToIndex(word);  ///!!!!!!!!!!!!!!!
    for (size_t j = 0; j < dim; j++) {
      in >> mat->data_[i * dim + j];
    }
  }
  in.close();

  //  dict_->threshold(1, 0);
  input_ = std::make_shared<Matrix>(dict_->nlexems, args_->dim);
  input_->uniform(1.0 / args_->dim);

  for (size_t i = 0; i < n; i++) {
    int32_t ind = dict_->getWordIndex(words[i]);
    if (ind < 0 || ind >= dict_->nwords) continue;
    for (size_t j = 0; j < dim; j++) {
      input_->data_[ind * dim + j] = mat->data_[i * dim + j];
    }
  }
}

void FastText::train(std::shared_ptr<Args> args) {
  args_ = args;
  dict_ = std::make_shared<Dictionary>(args_);
  if (args_->input == "-") {
    // manage expectations
    std::cerr << "Cannot use stdin for training!" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ifstream ifs(args_->input);
  if (!ifs.is_open()) {
    std::cerr << "Input file cannot be opened!" << std::endl;
    exit(EXIT_FAILURE);
  }
  ifs.close();
  //  if (args_->inputMatrix != "" ) {
  //	loadInputMatrix();
  //  }
  if (args_->pretrainedModel != "") {
    loadModel(args_->pretrainedModel, args_);
  } else {
    input_ = std::make_shared<Matrix>(dict_->nlexems, args_->dim);
    input_->uniform(1.0 / args_->dim);
    output_ = std::make_shared<Matrix>(dict_->nlexems, args_->dim);
    output_->zero();
  }

  // experiment - set input weighted by frequencies

  for (int32_t i = 0; i < dict_->nwords; ++i)
    input_->mulRow(i, dict_->getWordWeight(i));
  for (int32_t i = 0; i < dict_->nlexems - dict_->nwords; ++i)
    input_->mulRow(dict_->nwords + i, dict_->getLexemWeight(i));

  //  inputs_.push_back(input_);
  //  outputs_.push_back(output_);

  //  inputs_.push_back(std::make_shared<Matrix>(*input_));
  //  outputs_.push_back(input_);

  start = clock();
  tokenCount = 0;
  std::vector<std::thread> threads;
  cnt_active_threads = 0;
  cnt_threads = 0;
  commonSteps = 0;
  maxSteps = 10000;
  for (int32_t i = 0; i < args_->thread; i++) {
    threads.push_back(std::thread([=]() { trainThread(i); }));
  }
  for (size_t i = 0; i < threads.size(); ++i) {
    logging_thread = i;
    threads[i].join();
  }
  std::cerr << "all threads joined\n";

  main_model_ = std::make_shared<Model>(input_, output_, args_, dict_, 0);
  models_.push_back(main_model_);

  main_model_->normalizeModel();

  saveModel();
  saveVectors();
  if (args_->saveOutput > 0) {
    saveOutput();
  }
}

}  // namespace fasttext
