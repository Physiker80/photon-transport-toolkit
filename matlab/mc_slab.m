function result = mc_slab(mua, mus, g, thickness, n_med, n_out, Nphotons, seed, Nbatches)
% MC_SLAB  Weighted-photon Monte Carlo transport in a single homogeneous
% scattering/absorbing slab, following the general MCML approach
% (Wang, Jacques & Zheng 1995): sample a free path in optical-depth
% units, advance and test for boundary escape with an angle-dependent
% Fresnel test -- resolving however many boundary reflections occur
% within a single sampled free path, which matters whenever the mean
% free path is comparable to or larger than the slab thickness --
% deposit absorbed weight, scatter via the Henyey-Greenstein phase
% function, and terminate low-weight photons by Russian roulette.
%
% This is an INDEPENDENT re-derivation from the governing equations,
% written from scratch in MATLAB/Octave -- not a translation of the
% author's existing Python implementation -- specifically so the two
% can be cross-checked against each other as independent evidence.
%
% Inputs:
%   mua, mus, g   : absorption coeff, scattering coeff, anisotropy [1/mm, 1/mm, -]
%   thickness     : slab thickness [mm]
%   n_med, n_out  : refractive index of the medium and the outside
%   Nphotons      : photons per batch
%   seed          : RNG seed
%   Nbatches      : number of independent batches, for standard-error estimation
%
% Output: struct with fields Rd, T, A (means) and Rd_se, T_se, A_se
% (standard errors across batches), plus Rsp (specular reflectance).

    if nargin < 9, Nbatches = 10; end

    mu_t = mua + mus;
    W_THRESHOLD = 1e-4;
    ROULETTE_CHANCE = 0.1;

    Rsp = fresnel_reflectance(n_out, n_med, 1.0);

    Rd_batches = zeros(Nbatches, 1);
    T_batches  = zeros(Nbatches, 1);
    A_batches  = zeros(Nbatches, 1);

    rand("state", seed);

    for b = 1:Nbatches
        Rd_acc = 0; T_acc = 0; A_acc = 0;

        for p = 1:Nphotons
            z = 0;
            ux = 0; uy = 0; uz = 1;
            w = 1 - Rsp;
            Rd_acc = Rd_acc + Rsp;   % specular reflection: deterministic, leaves immediately

            while w > 0
                xi = rand();
                if xi <= 0, xi = eps; end
                tau_left = -log(xi);   % remaining free path, optical-depth units

                photon_done_step = false;
                while ~photon_done_step
                    s_full = tau_left / mu_t;
                    z_target = z + s_full*uz;

                    if uz < 0 && z_target < 0
                        s_bnd = (0 - z) / uz;
                        tau_used = s_bnd * mu_t;
                        tau_left = tau_left - tau_used;
                        z = 0;
                        cos_i = -uz;
                        R = fresnel_reflectance(n_med, n_out, cos_i);
                        if rand() > R
                            Rd_acc = Rd_acc + w;
                            w = 0; photon_done_step = true;
                        else
                            uz = -uz;
                        end

                    elseif uz > 0 && z_target > thickness
                        s_bnd = (thickness - z) / uz;
                        tau_used = s_bnd * mu_t;
                        tau_left = tau_left - tau_used;
                        z = thickness;
                        cos_i = uz;
                        R = fresnel_reflectance(n_med, n_out, cos_i);
                        if rand() > R
                            T_acc = T_acc + w;
                            w = 0; photon_done_step = true;
                        else
                            uz = -uz;
                        end

                    else
                        z = z_target;
                        photon_done_step = true;
                    end
                end

                if w == 0
                    break;
                end

                dw = w * (mus / mu_t);
                A_acc = A_acc + (w - dw);
                w = dw;

                cos_th = sample_hg(g);
                phi = 2*pi*rand();
                [ux, uy, uz] = rotate_direction(ux, uy, uz, cos_th, phi);

                if w < W_THRESHOLD
                    if rand() < ROULETTE_CHANCE
                        w = w / ROULETTE_CHANCE;
                    else
                        w = 0;
                    end
                end
            end
        end

        Rd_batches(b) = Rd_acc / Nphotons;
        T_batches(b)  = T_acc  / Nphotons;
        A_batches(b)  = A_acc  / Nphotons;
    end

    result.Rd = mean(Rd_batches);
    result.T  = mean(T_batches);
    result.A  = mean(A_batches);
    result.Rd_se = std(Rd_batches) / sqrt(Nbatches);
    result.T_se  = std(T_batches)  / sqrt(Nbatches);
    result.A_se  = std(A_batches)  / sqrt(Nbatches);
    result.Rsp = Rsp;
end
