// Copyright (c) 2013-2019 Simon Frasch, Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file broyden1_mixer.hpp
 *
 *   \brief Contains definition and implementation sirius::Broyden1.
 */

#ifndef __BROYDEN1_MIXER_HPP__
#define __BROYDEN1_MIXER_HPP__

#include <tuple>
#include <functional>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <exception>
#include <cmath>
#include <numeric>

#include "SDDK/memory.hpp"
#include "mixer/mixer.hpp"
#include "linalg/linalg.hpp"

namespace sirius {
namespace mixer {

/// Broyden mixer.
/** First version of the Broyden mixer, which requires inversion of the Jacobian matrix.
 *  Reference paper: "Robust acceleration of self consistent field calculations for
 *  density functional theory", Baarman K, Eirola T, Havu V., J Chem Phys. 134, 134109 (2011)
 */
template <typename... FUNCS>
class Broyden1 : public Mixer<FUNCS...>
{
  private:
    double beta_;
    double beta0_;
    double beta_scaling_factor_;
    sddk::mdarray<double, 2> S_;
    sddk::mdarray<double, 2> S_factorized_;
    std::size_t history_size_;
  public:
    Broyden1(std::size_t max_history, double beta, double beta0, double beta_scaling_factor)
        : Mixer<FUNCS...>(max_history)
        , beta_(beta)
        , beta0_(beta0)
        , beta_scaling_factor_(beta_scaling_factor)
        , S_(max_history - 1, max_history - 1)
        , S_factorized_(max_history - 1, max_history - 1)
        , history_size_(0)
    {
    }

    void mix_impl() override
    {
        const auto idx_step      = this->idx_hist(this->step_);
        const auto idx_next_step = this->idx_hist(this->step_ + 1);
        const auto idx_prev_step = this->idx_hist(this->step_ - 1);

        const auto history_size = static_cast<int>(this->history_size_);

        const bool normalize = false;

        // beta scaling
        if (this->step_ > this->max_history_) {
            const double rmse_avg = std::accumulate(this->rmse_history_.begin(), this->rmse_history_.end(), 0.0) /
                                    this->rmse_history_.size();
            if (this->rmse_history_[idx_step] > rmse_avg) {
                this->beta_ = std::max(beta0_, this->beta_ * beta_scaling_factor_);
            }
        }

        // Set up the next x_{n+1} = x_n.
        // Can't use this->output_history_[idx_step + 1] directly here,
        // as it's still used when history is full.
        this->copy(this->output_history_[idx_step], this->tmp1_);

        // + beta * f_n
        this->axpy(this->beta_, this->residual_history_[idx_step], this->tmp1_);

        if (history_size > 0) {
            // Compute the difference residual[step] - residual[step - 1]
            // and store it in residual[step - 1], but don't destroy
            // residual[step]
            this->scale(-1.0, this->residual_history_[idx_prev_step]);
            this->axpy(1.0, this->residual_history_[idx_step], this->residual_history_[idx_prev_step]);

            // Do the same for difference x
            this->scale(-1.0, this->output_history_[idx_prev_step]);
            this->axpy(1.0, this->output_history_[idx_step], this->output_history_[idx_prev_step]);

            // Compute the new Gram matrix for the least-squares problem
            for (int i = 0; i <= history_size - 1; ++i) {
                auto j = this->idx_hist(this->step_ - i - 1);
                this->S_(history_size - 1, history_size - i - 1) = this->S_(history_size - i - 1, history_size - 1) = this->template inner_product<normalize>(
                    this->residual_history_[j],
                    this->residual_history_[idx_prev_step]
                );
            }

            // Make a copy because factorizing destroys the matrix.
            for (int i = 0; i < history_size; ++i)
                for (int j = 0; j < history_size; ++j)
                    this->S_factorized_(j, i) = this->S_(j, i);

            sddk::mdarray<double, 1> h(history_size);
            for (int i = 1; i <= history_size; ++i) {
                auto j = this->idx_hist(this->step_ - i);
                h(history_size - i) = this->template inner_product<normalize>(
                    this->residual_history_[j],
                    this->residual_history_[idx_step]
                );
            }

            bool invertible = sddk::linalg(sddk::linalg_t::lapack).sysolve(history_size, this->S_factorized_, h);

            if (invertible) {
                // - beta * (delta F) * h
                for (int i = 1; i <= history_size; ++i) {
                    auto j = this->idx_hist(this->step_ - i);
                    this->axpy(-this->beta_ * h(history_size - i), this->residual_history_[j], this->tmp1_);
                }

                // - (delta X) * h
                for (int i = 1; i <= history_size; ++i) {
                    auto j = this->idx_hist(this->step_ - i);
                    this->axpy(-h(history_size - i), this->output_history_[j], this->tmp1_);
                }
            } else {
                this->history_size_ = 0;
            }
        }

        // In case history is full, set S_[1:end-1,1:end-1] .= S_[2:end,2:end]
        if (this->history_size_ == this->max_history_ - 1) {
            for (int col = 0; col <= history_size - 2; ++col) {
                for (int row = 0; row <= history_size - 2; ++row) {
                    this->S_(row, col) = this->S_(row + 1, col + 1);
                }
            }
        }

        this->copy(this->tmp1_, this->output_history_[idx_next_step]);
        this->history_size_ = std::min(this->history_size_ + 1, this->max_history_ - 1);
    }
};
} // namespace mixer
} // namespace sirius

#endif // __MIXER_HPP__
