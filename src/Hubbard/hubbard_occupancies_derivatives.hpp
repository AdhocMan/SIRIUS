// compute the forces for the simplex LDA+U method not the fully
// rotationally invariant one. It can not be used for LDA+U+SO either

// It is based on this reference : PRB 84, 161102(R) (2011)

// gradient of beta projectors. Needed for the computations of the forces

void Hubbard_potential::compute_occupancies_derivatives(K_point& kp,
                                                        Wave_functions& phi, // hubbard derivatives
                                                        Beta_projectors_gradient& bp_grad_,
                                                        Q_operator<double_complex>& q_op, // overlap operator
                                                        mdarray<double_complex, 5>& dn_,  // derivative of the occupation number compared to displacement of atom aton_id
                                                        const int atom_id)                // Atom we shift
{
    dn_.zero();
    // check if we have a norm conserving pseudo potential only. OOnly
    // derivatives of the hubbard wave functions are needed.

    bool augment = false;

    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
        augment = ctx_.unit_cell().atom_type(ia).augment();
    }

    if ((ctx_.full_potential() || !augment) && (!ctx_.unit_cell().atom(atom_id).type().hubbard_correction())) {
        // return immediatly if the atom has no hubbard correction and is norm conserving pp.
        return;
    }

    // Compute the derivatives of the occupancies in two cases.

    //- the atom is pp norm conserving or

    // - the atom is ppus (in that case the derivative the beta projectors
    // compared to the atomic displacements gives a non zero contribution)

    // temporary wave functions
    Wave_functions dphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);

    int HowManyBands = kp.num_occupied_bands(0);
    if (ctx_.num_spins() == 2)
        HowManyBands = std::max(kp.num_occupied_bands(1), kp.num_occupied_bands(0));

    // d_phitmp contains the derivatives of the hubbard wave functions
    // corresponding to the displacement r^I_a.

    dmatrix<double_complex> dPhi_S_Psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());
    dmatrix<double_complex> Phi_S_Psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());
    matrix<double_complex> dm(this->number_of_hubbard_orbitals() * ctx_.num_spins(),
                              this->number_of_hubbard_orbitals() * ctx_.num_spins());
    Phi_S_Psi.zero();
    dphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), phi, 0, 0, 0, 0);

    // computes the S|phi^I_ia>
    if (!ctx_.full_potential() && augment) {
        // note that I do *not* update phi but dphi instead since I need
        // to have the original wavefunctions later on
        for (int i = 0; i < kp.beta_projectors().num_chunks(); i++) {
            kp.beta_projectors().generate(i);
            auto beta_phi = kp.beta_projectors().inner<double_complex>(i, phi, 0, 0, this->number_of_hubbard_orbitals());
            /* apply Q operator (diagonal in spin) */
            q_op.apply(i, 0, dphi, 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(), beta_phi);
        }
    }

    // compute <phi^I_m| S | psi_{nk}>
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        inner(ctx_.processing_unit(), ispn, kp.spinor_wave_functions(), 0, kp.num_occupied_bands(ispn), dphi, 0,
              this->number_of_hubbard_orbitals(), Phi_S_Psi, 0, ispn * this->number_of_hubbard_orbitals());
    }

    #ifdef __GPU
    if ((ctx_.processing_unit() == GPU) && !ctx_.full_potential() && augment) {
        phitmp.pw_coeffs(0).prime().allocate(memory_t::device);
    }
    #endif


    for (int dir = 0; dir < 3; dir++) {
        // reset dphi
        dphi.pw_coeffs(0).prime().zero();

        if (ctx_.unit_cell().atom(atom_id).type().hubbard_correction()) {
            // atom atom_id has hubbard correction so we need to compute the
            // derivatives of the hubbard orbitals associated to the atom
            // atom_id, the derivatives of the others hubbard orbitals been
            // zero compared to the displacement of atom atom_id

            // compute the derivative of |phi> corresponding to the
            // atom atom_id
            const int lmax_at = 2 * ctx_.unit_cell().atom(atom_id).type().hubbard_l() + 1;

            // compute the derivatives of the hubbard wave functions
            // |phi_m^J> (J = atom_id) compared to a displacement of atom J.

            kp.compute_gradient_wavefunctions(phi, this->offset[atom_id], lmax_at, dphi, this->offset[atom_id], dir);
            // For norm conserving pp, it is enough to have the derivatives
            // of |phi^J_m> (J = atom_id)

            if (!ctx_.full_potential() && augment) {
                Wave_functions phitmp(kp.gkvec_partition(), lmax_at, 1);

                phitmp.copy_from(ctx_.processing_unit(), lmax_at, dphi, 0, this->offset[atom_id], 0, 0);
                // for the ppus potential we have an additional term coming from the
                // derivatives of the overlap matrix.
                // need to apply S on dphi^I

                for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
                    kp.beta_projectors().generate(chunk__);
                    // S| dphi> for this chunk
                    const int lmax_at = 2 * ctx_.unit_cell().atom(atom_id).type().hubbard_l() + 1;
                    auto beta_phi     = kp.beta_projectors().inner<double_complex>(chunk__, phitmp, 0, 0, lmax_at);
                    /* apply Q operator (diagonal in spin) */
                    q_op.apply(chunk__, 0, dphi, this->offset[atom_id], lmax_at, kp.beta_projectors(), beta_phi);
                }
            }
        }

        // compute d S/ dr^I_a |phi> and add to dphi
        if (!ctx_.full_potential() && augment) {
            // it is equal to
            // \sum Q^I_ij <d \beta^I_i|phi> |\beta^I_j> + < \beta^I_i|phi> |d\beta^I_j>
            for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
                for (int i = 0; i < kp.beta_projectors().chunk(chunk__).num_atoms_; i++) {
                    // need to find the right atom in the chunks.
                    if (kp.beta_projectors().chunk(chunk__).desc_(beta_desc_idx::ia, i) == atom_id) {
                        kp.beta_projectors().generate(chunk__);
                        bp_grad_.generate(chunk__, dir);

                        // compute Q_ij <\beta_i|\phi> |d \beta_j> and add it to d\phi
                        {
                            // < beta | phi> for this chunk
                            auto beta_phi =
                                kp.beta_projectors().inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());
                            q_op.apply_one_atom(chunk__, 0, dphi, 0, this->number_of_hubbard_orbitals(), bp_grad_, beta_phi, i);
                        }

                        // compute Q_ij <d \beta_i|\phi> |\beta_j> and add it to d\phi
                        {
                            // < dbeta | phi> for this chunk
                            auto dbeta_phi = bp_grad_.inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());

                            /* apply Q operator (diagonal in spin) */
                            /* Effectively compute Q_ij <d beta_i| phi> |beta_j> and add it dphi */
                            q_op.apply_one_atom(chunk__, 0, dphi, 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(), dbeta_phi,
                                                i);
                        }
                    }
                }
            }
        }

        // it is actually <psi | d(S|phi>)
        dPhi_S_Psi.zero();

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            inner(ctx_.processing_unit(), ispn, kp.spinor_wave_functions(), 0, kp.num_occupied_bands(ispn),
                  dphi, //   S d |phi>
                  0, this->number_of_hubbard_orbitals(), dPhi_S_Psi, 0, ispn * this->number_of_hubbard_orbitals());
        }

        // include the occupancy directly in dPhi_S_Psi

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            for (int n_orb = 0; n_orb < this->number_of_hubbard_orbitals(); n_orb++) {
                for (int nbnd = 0; nbnd < kp.num_occupied_bands(ispn); nbnd++) {
                    dPhi_S_Psi(nbnd, ispn * this->number_of_hubbard_orbitals() + n_orb) *= kp.band_occupancy(nbnd, ispn);
                }
            }
        }

        dm.zero();

        // TODO use matrix matrix multiplication

        // compute dm = \sum_{i,j} <\phi_i|S|\psi> <\psi|d(S|\phi>)_j
        #pragma omp parallel for schedule(static)
        for (int m1 = 0; m1 < this->number_of_hubbard_orbitals() * ctx_.num_spins(); m1++) {
            for (int m2 = 0; m2 < this->number_of_hubbard_orbitals() * ctx_.num_spins(); m2++) {
                for (int nbnd = 0; nbnd < HowManyBands; nbnd++) {
                    dm(m1, m2) += Phi_S_Psi(nbnd, m1) * std::conj(dPhi_S_Psi(nbnd, m2)) +
                        dPhi_S_Psi(nbnd, m1) * std::conj(Phi_S_Psi(nbnd, m2));
                }
            }
        }

        #pragma omp parallel for schedule(static)
        for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ++ia1) {
            const auto& atom = ctx_.unit_cell().atom(ia1);
            if (atom.type().hubbard_correction()) {
                const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    const size_t ispn_offset = ispn * this->number_of_hubbard_orbitals() + this->offset[ia1];
                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        for (int m1 = 0; m1 < lmax_at; m1++) {
                            dn_(m1, m2, ispn, ia1, dir) = dm(ispn_offset + m1, ispn_offset + m2) * kp.weight();
                        }
                    }
                }
            }
        }
    }
}

void Hubbard_potential::compute_occupancies_stress_derivatives(K_point& kp,
                                                               Beta_projectors_strain_deriv& bp_grad_,
                                                               Q_operator<double_complex>& q_op, // Compensnation operator or overlap operator
                                                               mdarray<double_complex, 5>& dn_)  // derivative of the occupation number compared to displacement of atom aton_id
{
    // check if we have a norm conserving pseudo potential only. OOnly
    // derivatives of the hubbard wave functions are needed.

    Wave_functions dphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);
    Wave_functions phi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);

    dmatrix<double_complex> tmp(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals());

    const int lmax  = ctx_.unit_cell().lmax();
    const int lmmax = Utils::lmmax(lmax);

    mdarray<double, 2> rlm_g(lmmax, kp.num_gkvec_loc());
    mdarray<double, 3> rlm_dg(lmmax, 3, kp.num_gkvec_loc());

    kp.generate_atomic_centered_wavefunctions_(this->number_of_hubbard_orbitals(), phi, this->offset, true);

    bool augment = false;

    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
        augment = ctx_.unit_cell().atom_type(ia).augment();
    }

    /* array of real spherical harmonics and derivatives for each G-vector */
    #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp.num_gkvec_loc(); igkloc++) {
        auto gvc = kp.gkvec().gkvec_cart(kp.idxgk(igkloc));
        auto rtp = SHT::spherical_coordinates(gvc);

        double theta = rtp[1];
        double phi   = rtp[2];

        SHT::spherical_harmonics(lmax, theta, phi, &rlm_g(0, igkloc));
        mdarray<double, 2> rlm_dg_tmp(&rlm_dg(0, 0, igkloc), lmmax, 3);
        SHT::dRlm_dr(lmax, gvc, rlm_dg_tmp);
    }

    // Compute the derivatives of the occupancies in two cases.

    //- the atom is pp norm conserving or

    // - the atom is ppus (in that case the derivative the beta projectors
    // compared to the atomic displacements gives a non zero contribution)

    // temporary wave functions

    int HowManyBands = kp.num_occupied_bands(0);
    if (ctx_.num_spins() == 2)
        HowManyBands = std::max(kp.num_occupied_bands(1), kp.num_occupied_bands(0));

    // d_phitmp contains the derivatives of the hubbard wave functions
    // corresponding to the distortion epsilon_alphabeta.

    dmatrix<double_complex> dPhi_S_Psi(HowManyBands, this->number_of_hubbard_orbitals());
    std::vector<dmatrix<double_complex>> Phi_S_Psi(ctx_.num_spins());
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        Phi_S_Psi[ispn] = dmatrix<double_complex>(HowManyBands, this->number_of_hubbard_orbitals());
    }

    // Todo : add gpu support

    dphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), phi, 0, 0, 0, 0);

    // computes the S|phi^I_ia>
    if (!ctx_.full_potential() && augment) {
        for (int i = 0; i < kp.beta_projectors().num_chunks(); i++) {
            /* generate beta-projectors for a block of atoms */
            kp.beta_projectors().generate(i);
            /* non-collinear case */
            auto beta_phi = kp.beta_projectors().inner<double_complex>(i, phi, 0, 0, this->number_of_hubbard_orbitals());
            /* apply Q operator (diagonal in spin) */
            q_op.apply(i, 0, dphi, 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(), beta_phi);
        }
    }

    // compute <phi^I_m| S | psi_{nk}>
    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        Phi_S_Psi[ispn].zero();
        inner(ctx_.processing_unit(), ispn, kp.spinor_wave_functions(), 0, kp.num_occupied_bands(ispn), dphi, 0,
              this->number_of_hubbard_orbitals(), Phi_S_Psi[ispn], 0, 0);
    }

    for (int mu = 0; mu < 3; mu++) {
        for (int nu = 0; nu < 3; nu++) {

            // atom atom_id has hubbard correction so we need to compute the
            // derivatives of the hubbard orbitals associated to the atom
            // atom_id, the derivatives of the others hubbard orbitals been
            // zero compared to the displacement of atom atom_id

            // compute the derivatives of all hubbard wave functions
            // |phi_m^J> compared to the strain

            compute_gradient_strain_wavefunctions(kp, dphi, rlm_g, rlm_dg, mu, nu);

            // // Need to apply the overlap operator on dphi
            // if (!ctx_.full_potential() && augment) {

            //   Wave_functions phitmp(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);

            //   phitmp.copy_from(ctx_.processing_unit(),
            //                    this->number_of_hubbard_orbitals(),
            //                    dphi,
            //                    0,
            //                    0,
            //                    0,
            //                    0);
            //   // for the ppus potential we have an additional term coming from the
            //   // derivatives of the overlap matrix.
            //   // need to apply S on dphi^I

            //   for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
            //     kp.beta_projectors().generate(chunk__);
            //     // S| dphi> for this chunk
            //     auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__,
            //                                                                phitmp,
            //                                                                0,
            //                                                                0,
            //                                                                this->number_of_hubbard_orbitals());
            //     /* apply Q operator (diagonal in spin) */
            //     q_op.apply(chunk__,
            //                0,
            //                dphi,
            //                0,
            //                this->number_of_hubbard_orbitals(),
            //                kp.beta_projectors(),
            //                beta_phi);
            //   }

            //   // it is equal to
            //   // \sum Q^I_ij <d \beta^I_i|phi> \beta^I_j + < \beta^I_i|phi> |d\beta^I_j>
            //   for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
            //     kp.beta_projectors().generate(chunk__);
            //     bp_grad_.generate(chunk__, mu + 3 * nu);

            //     // < beta | phi> for this chunk
            //     auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());

            //     /* apply Q operator (diagonal in spin) */
            //     /* compute Q_ij <d beta_i|phi> |beta_j> */
            //     q_op.apply(chunk__,
            //                0,
            //                dphi,
            //                0,
            //                this->number_of_hubbard_orbitals(),
            //                bp_grad_,
            //                beta_phi);
            //   }

            //   for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
            //     kp.beta_projectors().generate(chunk__);
            //     bp_grad_.generate(chunk__, mu + 3 * nu);
            //     // < dbeta | phi> for this chunk
            //     auto dbeta_phi = bp_grad_.inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());

            //     /* apply Q operator (diagonal in spin) */
            //     q_op.apply(chunk__, 0, dphi, 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(),
            //                dbeta_phi);
            //   }
            // }

            // it is actually <psi | d(S|phi>)

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                dPhi_S_Psi.zero();

                // DO not change the order
                inner(ctx_.processing_unit(), ispn, kp.spinor_wave_functions(), 0, kp.num_occupied_bands(ispn),
                      dphi, //   S d |phi>
                      0, this->number_of_hubbard_orbitals(), dPhi_S_Psi, 0, 0);
                // apply the f_{nk} to \left<phi_i | psi_{nk}\right>

                for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
                    for (int nbnd = 0; nbnd < kp.num_occupied_bands(ispn); nbnd++) {
                        dPhi_S_Psi(nbnd, m) *= kp.band_occupancy(nbnd, ispn);
                    }
                }
                tmp.zero();

                // we calculate \sum_{n, k} \left<d\phi_i | \psi_{nk}\right> \left<\psi_[nk] | \phi_j\right> f_{nk}

                // yes it is a matrix - matrix multiplication
                // todo : we should use matrix-matrix multiplication instead

                for (int n = 0; n < this->number_of_hubbard_orbitals(); n++) {
                    for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
                        for (int nbnd = 0; nbnd < kp.num_occupied_bands(ispn); nbnd++) {
                            tmp(m, n) += std::conj(dPhi_S_Psi(nbnd, m)) * Phi_S_Psi[ispn](nbnd, n) +
                                dPhi_S_Psi(nbnd, n) * std::conj(Phi_S_Psi[ispn](nbnd, m));
                        }
                    }
                }

                // finally add the contribution to the derivative of the occupation number.
                for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ++ia1) {
                    const auto& atom = ctx_.unit_cell().atom(ia1);
                    if (atom.type().hubbard_correction()) {
                        const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                dn_(m1, m2, ispn, ia1, mu + 3 * nu) += tmp(this->offset[ia1] + m1, this->offset[ia1] + m2) * kp.weight();
                            }
                        }
                    } // hubbard correction
                }     // atom
            }         // spin
        }             // nu
    }                 // mu
}

void Hubbard_potential::compute_gradient_strain_wavefunctions(K_point& kp__,
                                                              Wave_functions& dphi,
                                                              const mdarray<double, 2>& rlm_g,
                                                              const mdarray<double, 3>& rlm_dg,
                                                              const int mu, const int nu)
{
    //  #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
        auto gvc = kp__.gkvec().gkvec_cart(kp__.idxgk(igkloc));
        /* vs = {r, theta, phi} */
        auto gvs = SHT::spherical_coordinates(gvc);
        std::vector<mdarray<double, 1>> ri_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ri_values[iat] = ctx_.atomic_wf_ri().values(iat, gvs[0]);
        }

        std::vector<mdarray<double, 1>> ridjl_values(unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            ridjl_values[iat] = ctx_.atomic_wf_djl().values(iat, gvs[0]);
        }

        // case |g+k| = 0
        auto p = (mu == nu) ? 0.5 : 0.0;
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_type = ctx_.unit_cell().atom(ia).type();
            if (atom_type.hubbard_correction()) {
                for (int i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                    auto l = std::abs(atom_type.ps_atomic_wf(i).first);
                    if (l == atom_type.hubbard_l()) {
                        auto phase        = twopi * dot(gvc, unit_cell_.atom(ia).position());
                        auto phase_factor = double_complex(std::cos(phase), -std::sin(phase));
                        auto z            = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());

                        if (gvs[0] < 1e-10) {
                            if (l == 0) {
                                auto d1 = ri_values[atom_type.id()][i] * p * y00;
                                dphi.pw_coeffs(0).prime(igkloc, this->offset[ia]) = -z * d1;
                            } else {
                                for (int m = -l; m <= l; m++) {
                                    dphi.pw_coeffs(0).prime(igkloc, this->offset[ia] + l + m) = 0.0;
                                }
                            }
                        } else {
                            for (int m = -l; m <= l; m++) {
                                int lm  = Utils::lm_by_l_m(l, m);
                                auto d1 = ri_values[atom_type.id()][i] * (gvc[mu] * rlm_dg(lm, nu, igkloc) + p * rlm_g(lm, igkloc));
                                auto d2 = ridjl_values[atom_type.id()][i] * rlm_g(lm, igkloc) * gvc[mu] * gvc[nu] / gvs[0];
                                dphi.pw_coeffs(0).prime(igkloc, this->offset[ia] + l + m) = -z * (d1 + d2) * phase_factor;
                            }
                        }
                    }
                }
            }
        }
    }
}
