#ifndef _HUBBARD_HPP_
#define _HUBBARD_HPP_

#include <cstdio>
#include <cstdlib>
#include "simulation_context.h"
#include "k_point.h"
#include "wave_functions.hpp"
#include "Hamiltonian/non_local_operator.hpp"
#include "../Beta_projectors/beta_projectors.h"
#include "../Beta_projectors/beta_projectors_gradient.h"
#include "../Beta_projectors/beta_projectors_strain_deriv.h"
#include "radial_integrals.h"
#include "mixer.h"

namespace sirius {

class Hubbard_potential
{

    // Apply Hubbard correction in the colinear case
  private:
    Simulation_context& ctx_;

    Unit_cell& unit_cell_;

    int lmax_{0};

    int number_of_hubbard_orbitals_{0};

    mdarray<double_complex, 5> occupancy_number_;

    double hubbard_energy_{0.0};
    double hubbard_energy_u_{0.0};
    double hubbard_energy_dc_contribution_{0.0};
    double hubbard_energy_noflip_{0.0};
    double hubbard_energy_flip_{0.0};

    mdarray<double_complex, 5> hubbard_potential_;

    /// type of hubbard correction to be considered.  put to true if we
    /// consider a simple hubbard correction. Not valid if spin orbit
    /// coupling is included
    bool approximation_{false};

    /// orthogonalize and/or normalize the projectors
    bool orthogonalize_hubbard_orbitals_{false};

    /// by default we just normalize them
    bool normalize_orbitals_only_{false};

    /// hubbard correction with next nearest neighbors
    bool hubbard_U_plus_V_{false};

    /// hubbard projection method. By default we use the wave functions
    /// provided by the pseudo potentials.
    int projection_method_{0};

    /// Hubbard with multi channels (not implemented yet)
    bool multi_channels_{false};

    /// file containing the hubbard wave functions
    std::string wave_function_file_;

    /// pointer the radial integrals of the projectors. Only there for
    /// future use.
    ///std::unique_ptr<Radial_integrals_centered_atomic_wfc> wfc_;

public:
    std::vector<int> offset;

    void set_hubbard_U_plus_V(const bool U_plus_V_)
    {
        hubbard_U_plus_V_ = true;
    }

    void set_hubbard_simple_correction()
    {
        approximation_ = true;
    }

    inline int hubbard_lmax() const
    {
        return lmax_;
    }
    void set_orthogonalize_hubbard_orbitals(const bool test)
    {
        this->orthogonalize_hubbard_orbitals_ = test;
    }

    void set_normalize_hubbard_orbitals(const bool test)
    {
        this->normalize_orbitals_only_ = test;
    }

    double_complex U(int m1, int m2, int m3, int m4) const
    {
        return hubbard_potential_(m1, m2, m3, m4, 0);
    }

    double_complex& U(int m1, int m2, int m3, int m4)
    {
        return hubbard_potential_(m1, m2, m3, m4, 0);
    }

    double_complex U(int m1, int m2, int m3, int m4, int channel) const
    {
        return hubbard_potential_(m1, m2, m3, m4, channel);
    }

    double_complex& U(int m1, int m2, int m3, int m4, int channel)
    {
        return hubbard_potential_(m1, m2, m3, m4, channel);
    }

    const bool& orthogonalize_hubbard_orbitals() const
    {
        return this->orthogonalize_hubbard_orbitals_;
    }

    const bool& normalize_hubbard_orbitals() const
    {
        return this->normalize_orbitals_only_;
    }

    /// Apply the hubbard potential on wave functions
    void apply_hubbard_potential(K_point& kp,
                                 const int ispn_,
                                 const int idx__,
                                 const int n__,
                                 Wave_functions& phi,
                                 Wave_functions& ophi);

    /// Generate the atomic orbitals.
    void generate_atomic_orbitals(K_point& kp, Q_operator<double_complex>& q_op);
    void generate_atomic_orbitals(K_point& kp, Q_operator<double>& q_op);

    void hubbard_compute_occupation_numbers(K_point_set& kset_);

    inline void set_hubbard_occupancies_matrix(double *occ, int ld);
    inline void set_hubbard_occupancies_matrix_nc(double_complex *occ, int ld);
    inline void get_hubbard_occupancies_matrix(double *occ, int ld);
    inline void get_hubbard_occupancies_matrix_nc(double_complex *occ, int ld);

    inline void set_hubbard_potential_nc(double_complex *occ, int ld);
    inline void set_hubbard_potential(double *occ, int ld);

    void compute_occupancies_derivatives(K_point &kp,
                                         Q_operator<double_complex>& q_op,
                                         mdarray<double_complex, 6> &dn_);

    void compute_occupancies_stress_derivatives(K_point &kp,
                                                Q_operator<double_complex>& q_op, // Compensnation operator or overlap operator
                                                mdarray<double_complex, 5> &dn_); // derivative of the occupation number compared to displacement of atom aton_id


    void calculate_hubbard_potential_and_energy_colinear_case();
    void calculate_hubbard_potential_and_energy_non_colinear_case();
    void calculate_hubbard_potential_and_energy()
    {
        this->hubbard_energy_                 = 0.0;
        this->hubbard_energy_u_               = 0.0;
        this->hubbard_energy_dc_contribution_ = 0.0;
        this->hubbard_energy_noflip_          = 0.0;
        this->hubbard_energy_flip_            = 0.0;
        // the hubbard potential has the same structure than the occupation
        // numbers
        this->hubbard_potential_.zero();

        if (ctx_.num_mag_dims() != 3) {
            calculate_hubbard_potential_and_energy_colinear_case();
        } else {
            calculate_hubbard_potential_and_energy_non_colinear_case();
        }
    }

    inline double hubbard_energy() const
    {
        return this->hubbard_energy_;
    }

    inline int number_of_hubbard_orbitals() const
    {
        return number_of_hubbard_orbitals_;
    }

    Hubbard_potential(Simulation_context& ctx__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
    {
        if (!ctx_.hubbard_correction())
            return;
        this->orthogonalize_hubbard_orbitals_ = ctx_.Hubbard().orthogonalize_hubbard_orbitals_;
        this->normalize_orbitals_only_        = ctx_.Hubbard().normalize_hubbard_orbitals_;
        this->projection_method_ = ctx_.Hubbard().projection_method_;

        // if the projectors are defined externaly then we need the file
        // that contains them. All the other methods do not depend on
        // that parameter
        if(this->projection_method_ == 1) {
            this->wave_function_file_ = ctx_.Hubbard().wave_function_file_;
        }

        this->lmax_                             = -1;
        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            if (ctx__.unit_cell().atom(ia).type().hubbard_correction()) {
                this->lmax_ = std::max(this->lmax_, ctx_.unit_cell().atom(ia).type().hubbard_l());
            }
        }

        /// if spin orbit coupling or non colinear magnetisms are
        /// activated, then we consider the full spherical hubbard
        /// correction
        if ((ctx_.so_correction()) || (ctx_.num_mag_dims() == 3)) {
            approximation_ = false;
        }

        // prepare things for the multi channel case. The last index
        // indicates which channel we consider. By default we only have
        // one channel per atomic type
        occupancy_number_  = mdarray<double_complex, 5>(2 * lmax_ + 1, 2 * lmax_ + 1, 4, ctx_.unit_cell().num_atoms(), 1);
        hubbard_potential_ = mdarray<double_complex, 5>(2 * lmax_ + 1, 2 * lmax_ + 1, 4, ctx_.unit_cell().num_atoms(), 1);

        calculate_wavefunction_with_U_offset();
        calculate_initial_occupation_numbers();
        calculate_hubbard_potential_and_energy();
    }

private:
    void calculate_initial_occupation_numbers();

    void compute_occupancies(K_point& kp,
                             dmatrix<double_complex> &Phi_S_Psi,
                             dmatrix<double_complex> &dPhi_S_Psi,
                             Wave_functions& dphi,
                             mdarray<double_complex, 5>& dn_,
                             matrix<double_complex> &dm,
                             const int index);

    inline int natural_lm_to_qe(int m, int l)
    {
        return (m > 0) ? 2 * m - 1 : - 2 * m;
    }

    inline void symmetrize_occupancy_matrix_noncolinear_case();
    inline void symmetrize_occupancy_matrix();
    inline void print_occupancies();

    inline void calculate_wavefunction_with_U_offset()
    {
        offset.clear();
        offset.resize(ctx_.unit_cell().num_atoms(), -1);

        int counter = 0;
        for (auto ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);

            if (atom.type().hubbard_correction()) {
                // search for the orbital of given l corresponding to the
                // hubbard l, with strickly positive occupation
                for (int wfc = 0; wfc < atom.type().num_ps_atomic_wf(); wfc++) {
                    const int l      = std::abs(atom.type().ps_atomic_wf(wfc).first);
                    const double occ = atom.type().ps_atomic_wf_occ()[wfc];
                    if ((occ >= 0.0) && (l == atom.type().hubbard_l())) {
                        // a wave function is hubbard if and only if the occupation
                        // number is positive and l corresponds to hubbard_lmax;
                        bool hubbard_wfc = (occ > 0);
                        if (hubbard_wfc && (offset[ia] < 0)) {
                            offset[ia] = counter;
                        }

                        // the atom has spin orbit coupling so we have
                        // two wave functions with same l but different
                        // j
                        if (atom.type().spin_orbit_coupling() && hubbard_wfc) {
                            counter += (2 * l + 1);
                        } else {
                            if (hubbard_wfc && (ctx_.num_mag_dims() == 3)) {
                                // the pseudo potential does not include
                                // spin orbit coupling but we do
                                // calculation with non colinear
                                // magnetism so we still have full
                                // hubbard spinors
                                counter += 2 * (2 * l + 1);
                            }
                            if (hubbard_wfc && (ctx_.num_mag_dims() != 3)) {
                                // colinear or conventional LDA
                                counter += (2 * l + 1);
                            }
                        }
                    }
                }
            }
        }
        // compute the number of orbitals
        this->number_of_hubbard_orbitals_ = counter;
    }

    /// Compute the strain gradient of the hubbard wave functions.
    /// Unfortunately it is dependent of the pp.

    void compute_gradient_strain_wavefunctions(K_point &kp__,
                                               Wave_functions &dphi,
                                               const mdarray<double, 2> &rlm_g,
                                               const mdarray<double, 3> &rlm_dg,
                                               const int mu,
                                               const int nu);

    /// apply the S operator in the us pp case. Otherwise it makes a simple copy
    void Apply_S_operator(K_point &kp,
                          Q_operator<double_complex>& q_op,
                          Wave_functions& phi,
                          Wave_functions& ophi,
                          const int idx0,
                          const int num_phi);

    /// orthogonize (normalize) the hubbard wave functions
    void orthogonalize_atomic_orbitals(K_point& kp, Wave_functions &sphi);
};
    #include "Hubbard/hubbard_generate_atomic_orbitals.hpp"
    #include "Hubbard/hubbard_potential_energy.hpp"
    #include "Hubbard/apply_hubbard_potential.hpp"
    #include "Hubbard/hubbard_occupancy.hpp"
    #include "Hubbard/hubbard_occupancies_derivatives.hpp"
}
#endif
