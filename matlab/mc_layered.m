function result = mc_layered(mua_vec, mus_vec, g_vec, thick_vec, n_vec, n_out, Nphotons, seed, Nbatches)
% MC_LAYERED  Weighted-photon Monte Carlo transport through an
% arbitrary stack of homogeneous layers.
%
% The physically essential difference from a single homogeneous slab:
% a photon's remaining free path must be carried across an internal
% boundary in OPTICAL-DEPTH (tau) units and re-expressed in the new
% layer's physical distance using that layer's own mu_t -- simply
% clamping the photon's position at the boundary and resuming with
% the old layer's mu_t would silently misrepresent the medium the
% photon is actually travelling through.
%
% This is an INDEPENDENT re-derivation from the governing transport
% equation, written from scratch in MATLAB/Octave -- not a translation
% of the author's existing Python implementation -- specifically so
% the two can be cross-checked against each other as independent
% evidence for any result obtained from either.
%
% Inputs:
%   mua_vec, mus_vec, g_vec, thick_vec, n_vec : per-layer properties,
%       one row/element per layer, layer 1 = topmost (entrance) layer
%   n_out      : refractive index outside the stack (both faces)
%   Nphotons   : photons per batch
%   seed       : RNG seed
%   Nbatches   : number of batches for standard-error estimation
%
% Output: struct with fields Rd, T, A (means) and Rd_se, T_se, A_se.

    if nargin < 9, Nbatches = 10; end

    Nlayers = numel(mua_vec);
    mut_vec = mua_vec + mus_vec;

    z_bound = zeros(Nlayers+1, 1);
    for k = 1:Nlayers
        z_bound(k+1) = z_bound(k) + thick_vec(k);
    end

    W_THRESHOLD = 1e-4;
    ROULETTE_CHANCE = 0.1;

    Rsp = fresnel_reflectance(n_out, n_vec(1), 1.0);

    Rd_batches = zeros(Nbatches, 1);
    T_batches  = zeros(Nbatches, 1);
    A_batches  = zeros(Nbatches, 1);

    rand("state", seed);

    for b = 1:Nbatches
        Rd_acc = 0; T_acc = 0; A_acc = 0;

        for p = 1:Nphotons
            z = 0; ux = 0; uy = 0; uz = 1;
            w = 1 - Rsp;
            Rd_acc = Rd_acc + Rsp;   % specular reflection: deterministic, leaves immediately
            li = 1;

            while w > 0
                xi = rand();
                if xi <= 0, xi = eps; end
                tau_left = -log(xi);

                photon_done = false;
                while ~photon_done
                    s_full = tau_left / mut_vec(li);
                    z_target = z + s_full*uz;

                    if uz > 0 && z_target >= z_bound(li+1) && li < Nlayers
                        s_bnd = (z_bound(li+1) - z) / uz;
                        tau_used = s_bnd * mut_vec(li);
                        tau_left = tau_left - tau_used;
                        z = z_bound(li+1);
                        R = fresnel_reflectance(n_vec(li), n_vec(li+1), uz);
                        if rand() > R
                            [ux, uy, uz] = refract_direction(ux, uy, uz, n_vec(li), n_vec(li+1));
                            li = li + 1;
                        else
                            uz = -uz;
                        end

                    elseif uz < 0 && z_target <= z_bound(li) && li > 1
                        s_bnd = (z_bound(li) - z) / uz;
                        tau_used = s_bnd * mut_vec(li);
                        tau_left = tau_left - tau_used;
                        z = z_bound(li);
                        R = fresnel_reflectance(n_vec(li), n_vec(li-1), -uz);
                        if rand() > R
                            [ux, uy, uz] = refract_direction(ux, uy, uz, n_vec(li), n_vec(li-1));
                            li = li - 1;
                        else
                            uz = -uz;
                        end

                    elseif uz < 0 && z_target <= z_bound(1)
                        s_bnd = (z_bound(1) - z) / uz;
                        z = z_bound(1);
                        R = fresnel_reflectance(n_vec(li), n_out, -uz);
                        if rand() > R
                            Rd_acc = Rd_acc + w;
                            w = 0; photon_done = true;
                        else
                            uz = -uz;
                            tau_used = s_bnd * mut_vec(li);
                            tau_left = tau_left - tau_used;
                        end

                    elseif uz > 0 && z_target >= z_bound(end)
                        s_bnd = (z_bound(end) - z) / uz;
                        z = z_bound(end);
                        R = fresnel_reflectance(n_vec(li), n_out, uz);
                        if rand() > R
                            T_acc = T_acc + w;
                            w = 0; photon_done = true;
                        else
                            uz = -uz;
                            tau_used = s_bnd * mut_vec(li);
                            tau_left = tau_left - tau_used;
                        end

                    else
                        z = z_target;
                        photon_done = true;
                    end
                end

                if w == 0
                    break;
                end

                dw = w * (mus_vec(li) / mut_vec(li));
                A_acc = A_acc + (w - dw);
                w = dw;

                cos_th = sample_hg(g_vec(li));
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
end
