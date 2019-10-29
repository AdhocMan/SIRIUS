// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file spline_inner.hpp
 *
 *  \brief TODO
 */

#ifndef __SPLINE_INNER_HPP__
#define __SPLINE_INNER_HPP__

#include <vector>
#include <deque>
#include <list>
#include <tuple>
#include "GPU/acc.hpp"
#include "spline.hpp"
#include "radial_grid.hpp"
#include "SDDK/memory.hpp"

#include <string> // TODO: remove
#include <cstdlib> // TODO: remove

#ifdef __GPU
#include "SDDK/GPU/acc.hpp"

extern "C" void spline_inner_product_gpu_m0(acc_stream_t stream, const int num_splines, const int max_num_points,
                                            const int* num_points, const int* offsets, const double4* coeffs_1,
                                            const double4* coeffs_2, const double* x0, const double* dx,
                                            double* results);

extern "C" void spline_inner_product_gpu_m1(acc_stream_t stream, const int num_splines, const int max_num_points,
                                            const int* num_points, const int* offsets, const double4* coeffs_1,
                                            const double4* coeffs_2, const double* x0, const double* dx,
                                            double* results);

extern "C" void spline_inner_product_gpu_m2(acc_stream_t stream, const int num_splines, const int max_num_points,
                                            const int* num_points, const int* offsets, const double4* coeffs_1,
                                            const double4* coeffs_2, const double* x0, const double* dx,
                                            double* results);
#endif

namespace sirius {

template <typename IDENTIFIER>
class SplineInner
{
  public:
    void add(const Spline<double>& left, const Spline<double>& right, int m, IDENTIFIER ident)
    {
        input_splines_.emplace_back(&left, &right, m, std::move(ident));
    }

    void add(Spline<double>&& left, const Spline<double>& right, int m, IDENTIFIER ident)
    {
        temporary_splines_.emplace_back(std::move(left));
        this->add(temporary_splines_.back(), right, m, std::move(ident));
    }

    void add(const Spline<double>& left, Spline<double>&& right, int m, IDENTIFIER ident)
    {
        this->add(std::move(right), left, m, std::move(ident));
    }

    void add(Spline<double>&& left, Spline<double>&& right, int m, IDENTIFIER ident)
    {
        temporary_splines_.emplace_back(std::move(right));
        this->add(std::move(left), temporary_splines_.back(), m, std::move(ident));
    }

    void reset()
    {
        input_splines_.clear();
        temporary_splines_.clear();
    }

    std::vector<std::pair<double, IDENTIFIER>> compute(sddk::device_t pu)
    {
        std::vector<std::pair<double, IDENTIFIER>> results(input_splines_.size());
        const char* env = std::getenv("PU_UNIT");
        std::string pu_name("CPU");
        if (env) {
            pu_name = std::string(env);
        }

        // if (pu == sddk::device_t::CPU) {
        if (pu_name == std::string("CPU")) {
            PROFILE("sirius::Spline_inner|cpu");

#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < results.size(); ++i) {
                results[i] = std::make_pair(
                    sirius::inner(*std::get<0>(input_splines_[i]), *std::get<1>(input_splines_[i]),
                                  std::get<2>(input_splines_[i]), std::get<0>(input_splines_[i])->num_points()),
                    std::get<3>(input_splines_[i]));
            }
        } else {
#ifdef __GPU
            std::cout << " ===================== GPU INNER SPLINE PREPARATION ===========================" << std::endl;

            PROFILE("sirius::Spline_inner|gpu");
            sddk::mdarray<int, 1> offsets(input_splines_.size(), sddk::memory_t::host_pinned);
            offsets.allocate(sddk::memory_t::device);

            sddk::mdarray<int, 1> num_points(input_splines_.size(), sddk::memory_t::host_pinned);
            num_points.allocate(sddk::memory_t::device);

            std::size_t count  = 0;
            int total_size     = 0;
            int max_num_points = 0;
            for (const auto& input : input_splines_) {
                assert(std::get<0>(input)->num_points() == std::get<1>(input)->num_points());

                offsets[count] = total_size;
                total_size += std::get<0>(input)->num_points();
                max_num_points    = std::max(std::get<0>(input)->num_points(), max_num_points);
                num_points[count] = std::get<0>(input)->num_points();
                ++count;
            }
            offsets.copy_to(sddk::memory_t::device, stream_id(0));
            num_points.copy_to(sddk::memory_t::device, stream_id(0));

            sddk::mdarray<double, 1> device_results(total_size, sddk::memory_t::host_pinned);
            device_results.allocate(sddk::memory_t::device);

            sddk::mdarray<double4, 1> coeffs_1(total_size, sddk::memory_t::host_pinned);
            coeffs_1.allocate(sddk::memory_t::device);

            sddk::mdarray<double4, 1> coeffs_2(total_size, sddk::memory_t::host_pinned);
            coeffs_2.allocate(sddk::memory_t::device);

            sddk::mdarray<double, 1> dx(total_size, sddk::memory_t::host_pinned);
            dx.allocate(sddk::memory_t::device);

            sddk::mdarray<double, 1> x0(total_size, sddk::memory_t::host_pinned);
            x0.allocate(sddk::memory_t::device);

            count = 0;
            for (const auto& input : input_splines_) {
                const auto& input_coeffs_1 = std::get<0>(input)->coeffs();
                const auto& input_coeffs_2 = std::get<1>(input)->coeffs();
                const auto& input_dx       = std::get<0>(input)->dx();
                const auto& input_x0       = std::get<0>(input)->x();

                for (int i = 0; i < std::get<0>(input)->num_points(); ++i, ++count) {
                    coeffs_1[count].x = input_coeffs_1(i, 0);
                    coeffs_1[count].y = input_coeffs_1(i, 1);
                    coeffs_1[count].z = input_coeffs_1(i, 2);
                    coeffs_1[count].w = input_coeffs_1(i, 3);

                    coeffs_2[count].x = input_coeffs_2(i, 0);
                    coeffs_2[count].y = input_coeffs_2(i, 1);
                    coeffs_2[count].z = input_coeffs_2(i, 2);
                    coeffs_2[count].w = input_coeffs_2(i, 3);

                    dx[count] = input_dx(i);
                    x0[count] = input_x0(i);
                }
            }

            coeffs_1.copy_to(sddk::memory_t::device, stream_id(0));
            coeffs_2.copy_to(sddk::memory_t::device, stream_id(0));
            dx.copy_to(sddk::memory_t::device, stream_id(0));
            x0.copy_to(sddk::memory_t::device, stream_id(0));

            // r^m
            const auto m = std::get<2>(input_splines_[0]); // TODO

            std::cout << " ===================== GPU INNER SPLINE kernel ===========================" << std::endl;
            {
                PROFILE("sirius::Spline_inner|gpu-computation");
                switch (m) {
                    case 0: {
                        spline_inner_product_gpu_m0(
                            acc::stream(stream_id(0)), input_splines_.size(), max_num_points,
                            num_points.at(sddk::memory_t::device), offsets.at(sddk::memory_t::device),
                            coeffs_1.at(sddk::memory_t::device), coeffs_2.at(sddk::memory_t::device),
                            x0.at(sddk::memory_t::device), dx.at(sddk::memory_t::device),
                            device_results.at(sddk::memory_t::device));
                    } break;
                    case 1: {
                        spline_inner_product_gpu_m1(
                            acc::stream(stream_id(0)), input_splines_.size(), max_num_points,
                            num_points.at(sddk::memory_t::device), offsets.at(sddk::memory_t::device),
                            coeffs_1.at(sddk::memory_t::device), coeffs_2.at(sddk::memory_t::device),
                            x0.at(sddk::memory_t::device), dx.at(sddk::memory_t::device),
                            device_results.at(sddk::memory_t::device));
                    } break;
                    case 2: {
                        spline_inner_product_gpu_m2(
                            acc::stream(stream_id(0)), input_splines_.size(), max_num_points,
                            num_points.at(sddk::memory_t::device), offsets.at(sddk::memory_t::device),
                            coeffs_1.at(sddk::memory_t::device), coeffs_2.at(sddk::memory_t::device),
                            x0.at(sddk::memory_t::device), dx.at(sddk::memory_t::device),
                            device_results.at(sddk::memory_t::device));
                    } break;

                    default: {
                        throw std::runtime_error("wrong r^m prefactor");
                    }
                }

                device_results.copy_to(sddk::memory_t::host, stream_id(0));
                acc::sync_stream(stream_id(0));
            }

            for (std::size_t i = 0; i < results.size(); ++i) {
                results[i] = std::make_pair(device_results[i], std::get<3>(input_splines_[i]));
            }
#else
            throw std::runtime_error("SplineInner::compute(): Not compiled with GPU support.");
#endif
        }
        // for (std::size_t i = 0; i < results.size(); ++i) {
        //     std::cout << results[i].first << std::endl;
        // }

        return results;
    }

  private:
    std::deque<std::tuple<const Spline<double>*, const Spline<double>*, int, IDENTIFIER>> input_splines_;
    std::list<Spline<double>> temporary_splines_;
};

}; // namespace sirius

#endif
