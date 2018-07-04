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

/** \file utils.h
 *
 *  \brief Contains definition and partial implementation of sirius::Utils class.
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <gsl/gsl_sf_erf.h>
#include <fstream>
#include <string>
#include <complex>
#include "typedefs.h"
#include "constants.h"
#include "sddk.hpp"

using namespace sddk;

/// Utility class.
class Utils // TODO: namespace utils
{
  public:
    /// Maximum number of \f$ \ell, m \f$ combinations for a given \f$ \ell_{max} \f$
    static inline int lmmax(int lmax)
    {
        return (lmax + 1) * (lmax + 1);
    }

    static inline int lm_by_l_m(int l, int m) // TODO lm_by_l_m(l, m) -> lm(l, m)
    {
        return (l * l + l + m);
    }

    //static inline int lmax_by_lmmax(int lmmax__) // TODO: lmax_by_lmmax(lmmax) -> lmax(lmmax)
    //{
    //    int lmax = int(std::sqrt(double(lmmax__)) + 1e-8) - 1;
    //    if (lmmax(lmax) != lmmax__) {
    //        TERMINATE("wrong lmmax");
    //    }
    //    return lmax;
    //}

    static void write_matrix(const std::string& fname,
                             mdarray<double_complex, 2>& matrix,
                             int nrow,
                             int ncol,
                             bool write_upper_only = true,
                             bool write_abs_only   = false,
                             std::string fmt       = "%18.12f")
    {
        static int icount = 0;

        if (nrow < 0 || nrow > (int)matrix.size(0) || ncol < 0 || ncol > (int)matrix.size(1))
            TERMINATE("wrong number of rows or columns");

        icount++;
        std::stringstream s;
        s << icount;
        std::string full_name = s.str() + "_" + fname;

        FILE* fout = fopen(full_name.c_str(), "w");

        for (int icol = 0; icol < ncol; icol++) {
            fprintf(fout, "column : %4i\n", icol);
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");
            if (write_abs_only) {
                fprintf(fout, " row, absolute value\n");
            } else {
                fprintf(fout, " row, real part, imaginary part, absolute value\n");
            }
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");

            int max_row = (write_upper_only) ? std::min(icol, nrow - 1) : (nrow - 1);
            for (int j = 0; j <= max_row; j++) {
                if (write_abs_only) {
                    std::string s = "%4i  " + fmt + "\n";
                    fprintf(fout, s.c_str(), j, abs(matrix(j, icol)));
                } else {
                    fprintf(fout, "%4i  %18.12f %18.12f %18.12f\n", j, real(matrix(j, icol)), imag(matrix(j, icol)),
                            abs(matrix(j, icol)));
                }
            }
            fprintf(fout, "\n");
        }

        fclose(fout);
    }

    static void write_matrix(std::string const& fname, bool write_all, mdarray<double, 2>& matrix)
    {
        static int icount = 0;

        icount++;
        std::stringstream s;
        s << icount;
        std::string full_name = s.str() + "_" + fname;

        FILE* fout = fopen(full_name.c_str(), "w");

        for (int icol = 0; icol < (int)matrix.size(1); icol++) {
            fprintf(fout, "column : %4i\n", icol);
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");
            fprintf(fout, " row\n");
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");

            int max_row = (write_all) ? ((int)matrix.size(0) - 1) : std::min(icol, (int)matrix.size(0) - 1);
            for (int j = 0; j <= max_row; j++) {
                fprintf(fout, "%4i  %18.12f\n", j, matrix(j, icol));
            }
            fprintf(fout, "\n");
        }

        fclose(fout);
    }

    static void write_matrix(std::string const& fname, bool write_all, matrix<double_complex> const& mtrx)
    {
        static int icount = 0;

        icount++;
        std::stringstream s;
        s << icount;
        std::string full_name = s.str() + "_" + fname;

        FILE* fout = fopen(full_name.c_str(), "w");

        for (int icol = 0; icol < (int)mtrx.size(1); icol++) {
            fprintf(fout, "column : %4i\n", icol);
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");
            fprintf(fout, " row\n");
            for (int i = 0; i < 80; i++)
                fprintf(fout, "-");
            fprintf(fout, "\n");

            int max_row = (write_all) ? ((int)mtrx.size(0) - 1) : std::min(icol, (int)mtrx.size(0) - 1);
            for (int j = 0; j <= max_row; j++) {
                fprintf(fout, "%4i  %18.12f %18.12f\n", j, real(mtrx(j, icol)), imag(mtrx(j, icol)));
            }
            fprintf(fout, "\n");
        }

        fclose(fout);
    }

    template <typename T>
    static void check_hermitian(const std::string& name, matrix<T> const& mtrx, int n = -1)
    {
        assert(mtrx.size(0) == mtrx.size(1));

        double maxdiff = 0.0;
        int i0         = -1;
        int j0         = -1;

        if (n == -1) {
            n = static_cast<int>(mtrx.size(0));
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double diff = std::abs(mtrx(i, j) - type_wrapper<T>::conjugate(mtrx(j, i)));
                if (diff > maxdiff) {
                    maxdiff = diff;
                    i0      = i;
                    j0      = j;
                }
            }
        }

        if (maxdiff > 1e-10) {
            std::stringstream s;
            s << name << " is not a symmetric or hermitian matrix" << std::endl
              << "  maximum error: i, j : " << i0 << " " << j0 << " diff : " << maxdiff;

            WARNING(s);
        }
    }

    static double confined_polynomial(double r, double R, int p1, int p2, int dm)
    {
        double t = 1.0 - std::pow(r / R, 2);
        switch (dm) {
            case 0: {
                return (std::pow(r, p1) * std::pow(t, p2));
            }
            case 2: {
                return (-4 * p1 * p2 * std::pow(r, p1) * std::pow(t, p2 - 1) / std::pow(R, 2) +
                        p1 * (p1 - 1) * std::pow(r, p1 - 2) * std::pow(t, p2) +
                        std::pow(r, p1) * (4 * (p2 - 1) * p2 * std::pow(r, 2) * std::pow(t, p2 - 2) / std::pow(R, 4) -
                                           2 * p2 * std::pow(t, p2 - 1) / std::pow(R, 2)));
            }
            default: {
                TERMINATE("wrong derivative order");
                return 0.0;
            }
        }
    }

    static mdarray<int, 1> l_by_lm(int lmax)
    {
        mdarray<int, 1> v(lmmax(lmax));
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                v[lm_by_l_m(l, m)] = l;
            }
        }
        return std::move(v);
    }

    static std::vector<std::pair<int, int>> l_m_by_lm(int lmax)
    {
        std::vector<std::pair<int, int>> v(lmmax(lmax));
        for (int l = 0; l <= lmax; l++) {
            for (int m = -l; m <= l; m++) {
                int lm       = lm_by_l_m(l, m);
                v[lm].first  = l;
                v[lm].second = m;
            }
        }
        return std::move(v);
    }

    /// Read json dictionary from file or string.
    /** Terminate if file doesn't exist. */
    inline static json read_json_from_file_or_string(std::string const& str__)
    {
        json dict = {};
        if (str__.size() == 0) {
            return std::move(dict);
        }

        if (str__.find("{") == std::string::npos) { /* this is a file */
            if (utils::file_exists(str__)) {
                try {
                    std::ifstream(str__) >> dict;
                } catch(std::exception& e) {
                    std::stringstream s;
                    s << "wrong input json file" << std::endl
                      << e.what();
                    TERMINATE(s);
                }
            } 
            else {
                std::stringstream s;
                s << "file " << str__ << " doesn't exist";
                TERMINATE(s);
            }
        } else { /* this is a json string */
            try {
                std::istringstream(str__) >> dict;
            } catch (std::exception& e) {
                std::stringstream s;
                s << "wrong input json string" << std::endl
                  << e.what();
                TERMINATE(s);
            }
        }

        return std::move(dict);
    }

};

#endif
