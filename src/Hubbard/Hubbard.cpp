#include "hubbard.hpp"

namespace sirius {
    Hubbard::Hubbard(Simulation_context& ctx__)
        : ctx_(ctx__)
        , unit_cell_(ctx__.unit_cell())
    {
        if (!ctx_.hubbard_correction()) {
            return;
        }
        orthogonalize_hubbard_orbitals_ = ctx_.Hubbard().orthogonalize_hubbard_orbitals_;
        normalize_orbitals_only_        = ctx_.Hubbard().normalize_hubbard_orbitals_;
        projection_method_              = ctx_.Hubbard().projection_method_;

        // if the projectors are defined externaly then we need the file
        // that contains them. All the other methods do not depend on
        // that parameter
        if (this->projection_method_ == 1) {
            this->wave_function_file_ = ctx_.Hubbard().wave_function_file_;
        }

        int indexb_max = -1;

        for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
            if (ctx__.unit_cell().atom(ia).type().hubbard_correction()) {
                if (ctx__.unit_cell().atom(ia).type().spin_orbit_coupling()) {
                    indexb_max = std::max(indexb_max, ctx__.unit_cell().atom(ia).type().hubbard_indexb_wfc().size() / 2);
                } else {
                    indexb_max = std::max(indexb_max, ctx__.unit_cell().atom(ia).type().hubbard_indexb_wfc().size());
                }
            }
        }

        /* if spin orbit coupling or non colinear magnetisms are activated, then
         we consider the full spherical hubbard correction */

        if ((ctx_.so_correction()) || (ctx_.num_mag_dims() == 3)) {
            approximation_ = false;
        }

        occupancy_number_  = mdarray<double_complex, 4>(indexb_max, indexb_max, 4, ctx_.unit_cell().num_atoms());
        hubbard_potential_ = mdarray<double_complex, 4>(indexb_max, indexb_max, 4, ctx_.unit_cell().num_atoms());

        calculate_wavefunction_with_U_offset();
        calculate_initial_occupation_numbers();
        calculate_hubbard_potential_and_energy();
    }


    void Hubbard::calculate_wavefunction_with_U_offset()
    {
        offset.clear();
        offset.resize(ctx_.unit_cell().num_atoms(), -1);

        int counter = 0;

        /* we loop over atoms to check which atom has hubbard orbitals and then
         compute the number of hubbard orbitals associated to it. */

        for (auto ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            if (atom.type().hubbard_correction()) {
                offset[ia] = counter;
                counter += atom.type().hubbard_indexb_wfc().size();
                /* there is a factor two when the pseudo-potential has no SO but
                   we do full non colinear magnetism. Note that we can consider
                   now multiple orbitals calculations. The API still does not
                   support it */

                if ((ctx_.num_mag_dims() == 3) && (!atom.type().spin_orbit_coupling())) {
                    counter += atom.type().hubbard_indexb_wfc().size();
                }
            }
        }

        this->number_of_hubbard_orbitals_ = counter;
    }
}
