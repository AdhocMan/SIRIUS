// Copyright (c) 2013-2017 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
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

/** \file non_local_functor.hpp
 *
 *  \brief Common operation for forces and stress tensor.
 */

#ifndef __NON_LOCAL_FUNCTOR_HPP__
#define __NON_LOCAL_FUNCTOR_HPP__

#include "periodic_function.hpp"
#include "Density/augmentation_operator.hpp"
#include "Beta_projectors/beta_projectors.hpp"

namespace sirius {

template <typename T, int N>
class Non_local_functor
{
  private:
    Simulation_context& ctx_;
    Beta_projectors_base<N>& bp_base_;

  public:

    Non_local_functor(Simulation_context& ctx__,
                      Beta_projectors_base<N>& bp_base__)
        : ctx_(ctx__)
        , bp_base_(bp_base__)
    {
    }

    /// collect summation result in an array
    void add_k_point_contribution(K_point& kpoint__, mdarray<double, 2>& collect_res__)
    {
        Unit_cell& unit_cell = ctx_.unit_cell();

        Beta_projectors& bp = kpoint__.beta_projectors();

        double main_two_factor = -2.0;

        bp_base_.prepare();
        bp.prepare();

        for (int icnk = 0; icnk < bp_base_.num_chunks(); icnk++) {
            /* generate chunk for inner product of beta */
            bp.generate(icnk);

            /* store <beta|psi> for spin up and down */
            matrix<T> beta_phi_chunks[2];

            for(int ispn = 0; ispn < ctx_.num_spins(); ispn++){
                int nbnd = kpoint__.num_occupied_bands(ispn);
                //auto beta_phi_tmp = bp.inner<T>(icnk, kpoint__.spinor_wave_functions(), ispn, 0, nbnd);
                //beta_phi_chunks[ispn] = matrix<T>(beta_phi_tmp.size(0), beta_phi_tmp.size(1)) ;
                //beta_phi_tmp >> beta_phi_chunks[ispn];
                beta_phi_chunks[ispn] = bp.inner<T>(icnk, kpoint__.spinor_wave_functions(), ispn, 0, nbnd);
            }

            for (int x = 0; x < N; x++) {
                /* generate chunk for inner product of beta gradient */
                bp_base_.generate(icnk, x);

                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    int spin_factor = (ispn == 0 ? 1 : -1);

                    int nbnd = kpoint__.num_occupied_bands(ispn);

                    /* inner product of beta gradient and WF */
                    auto bp_base_phi_chunk = bp_base_.template inner<T>(icnk, kpoint__.spinor_wave_functions(), ispn, 0, nbnd);

                    splindex<block> spl_nbnd(nbnd, kpoint__.comm().size(), kpoint__.comm().rank());

                    int nbnd_loc = spl_nbnd.local_size();

                    #pragma omp parallel for
                    for (int ia_chunk = 0; ia_chunk < bp_base_.chunk(icnk).num_atoms_; ia_chunk++) {
                        int ia   = bp_base_.chunk(icnk).desc_(beta_desc_idx::ia, ia_chunk);
                        int offs = bp_base_.chunk(icnk).desc_(beta_desc_idx::offset, ia_chunk);
                        int nbf  = bp_base_.chunk(icnk).desc_(beta_desc_idx::nbf, ia_chunk);
                        int iat  = unit_cell.atom(ia).type_id();

                        /* helper lambda to calculate for sum loop over bands for different beta_phi and dij combinations*/
                        auto for_bnd = [&](int ibf, int jbf, double_complex dij, double_complex qij, matrix<T>& beta_phi_chunk)
                        {
                            /* gather everything = - 2  Re[ occ(k,n) weight(k) beta_phi*(i,n) [ Dij - E(n)Qij] beta_base_phi(j,n) ]*/
                            for (int ibnd_loc = 0; ibnd_loc < nbnd_loc; ibnd_loc++) {
                                int ibnd = spl_nbnd[ibnd_loc];

                                double_complex scalar_part = main_two_factor * kpoint__.band_occupancy(ibnd, ispn) * kpoint__.weight() *
                                        std::conj(beta_phi_chunk(offs + jbf, ibnd)) * bp_base_phi_chunk(offs + ibf, ibnd) *
                                        (dij - kpoint__.band_energy(ibnd, ispn) * qij);

                                /* get real part and add to the result array*/
                                collect_res__(x, ia) += scalar_part.real();
                            }
                        };

                        for (int ibf = 0; ibf < nbf; ibf++) {
                            for (int jbf = 0; jbf < nbf; jbf++) {

                                /* Qij exists only in the case of ultrasoft/PAW */
                                double qij = unit_cell.atom(ia).type().augment() ? ctx_.augmentation_op(iat).q_mtrx(ibf, jbf) : 0.0;
                                double_complex dij = 0.0;

                                /* get non-magnetic or collinear spin parts of dij*/
                                switch (ctx_.num_spins()) {
                                    case 1: {
                                        dij = unit_cell.atom(ia).d_mtrx(ibf, jbf, 0);
                                        break;
                                    }

                                    case 2: {
                                        /* Dij(00) = dij + dij_Z ;  Dij(11) = dij - dij_Z*/
                                        dij =  (unit_cell.atom(ia).d_mtrx(ibf, jbf, 0) + spin_factor * unit_cell.atom(ia).d_mtrx(ibf, jbf, 1));
                                        break;
                                    }

                                    default: {
                                        TERMINATE("Error in non_local_functor, D_aug_mtrx. ");
                                        break;
                                    }
                                }

                                /* add non-magnetic or diagonal spin components ( or collinear part) */
                                for_bnd(ibf, jbf, dij, double_complex(qij, 0.0), beta_phi_chunks[ispn]);

                                /* for non-collinear case*/
                                if (ctx_.num_mag_dims() == 3) {
                                    /* Dij(10) = dij_X + i dij_Y ; Dij(01) = dij_X - i dij_Y */
                                    dij = double_complex( unit_cell.atom(ia).d_mtrx(ibf, jbf, 2), spin_factor * unit_cell.atom(ia).d_mtrx(ibf, jbf, 3));
                                    /* add non-diagonal spin components*/
                                    for_bnd(ibf, jbf, dij, double_complex(0.0, 0.0), beta_phi_chunks[ispn + spin_factor] );
                                }
                            } // jbf
                        } // ibf
                    } // ia_chunk
                } // ispn
            } // x
        }

        bp.dismiss();
        bp_base_.dismiss();
    }
};

}

#endif /* __NON_LOCAL_FUNCTOR_HPP__ */
