% test_bias_direction.m
% THE CENTRAL CROSS-LANGUAGE TEST.
%
% Independently checks, in a from-scratch MATLAB/Octave implementation,
% the central finding of the Python project: that the sign of
% delta_R = R_layered - R_homogeneous(bulk-averaged) depends on WHERE a
% fixed absorption contrast sits (shallow vs. deep), not only on its
% magnitude. Same physical parameters as the Python
% examples/test_inverted_geometry.py, for a direct, independent,
% cross-language comparison.

MU_A_LOW = 0.05; MU_A_HIGH = 0.50;   % contrast = 10
SURFACE_MUS = 22.5; SURFACE_G = 0.80;
DEEP_MUS = 20.0; DEEP_G = 0.90;
DEEP_THICKNESS = 1.5;
N_INDEX = 1.4; N_OUT = 1.0;
Nphotons = 800; Nbatches = 4; seed = 5;

thicknesses = [0.12, 0.25];

function [dR, sR] = run_config(surface_mua, deep_mua, thickness, ...
        SURFACE_MUS, SURFACE_G, DEEP_MUS, DEEP_G, DEEP_THICKNESS, N_INDEX, N_OUT, Nphotons, seed, Nbatches)

    mua_v = [surface_mua, deep_mua];
    mus_v = [SURFACE_MUS, DEEP_MUS];
    g_v   = [SURFACE_G, DEEP_G];
    th_v  = [thickness, DEEP_THICKNESS];
    n_v   = [N_INDEX, N_INDEX];

    r_layered = mc_layered(mua_v, mus_v, g_v, th_v, n_v, N_OUT, Nphotons, seed, Nbatches);

    total_th = thickness + DEEP_THICKNESS;
    mua_avg = (surface_mua*thickness + deep_mua*DEEP_THICKNESS) / total_th;
    musp_avg = (SURFACE_MUS*(1-SURFACE_G)*thickness + DEEP_MUS*(1-DEEP_G)*DEEP_THICKNESS) / total_th;
    g_avg = (SURFACE_G*thickness + DEEP_G*DEEP_THICKNESS) / total_th;
    mus_avg = musp_avg / (1 - g_avg);

    r_homog = mc_slab(mua_avg, mus_avg, g_avg, total_th, N_INDEX, N_OUT, Nphotons, seed, Nbatches);

    dR = r_layered.Rd - r_homog.Rd;
    sR = sqrt(r_layered.Rd_se^2 + r_homog.Rd_se^2);
end

printf("%10s %22s %22s %10s\n", "thickness", "A: strong SHALLOW", "B: strong DEEP", "flip?");
flips = 0;
for i = 1:numel(thicknesses)
    t = thicknesses(i);
    [dR_A, sR_A] = run_config(MU_A_HIGH, MU_A_LOW, t, SURFACE_MUS, SURFACE_G, DEEP_MUS, DEEP_G, DEEP_THICKNESS, N_INDEX, N_OUT, Nphotons, seed, Nbatches);
    [dR_B, sR_B] = run_config(MU_A_LOW, MU_A_HIGH, t, SURFACE_MUS, SURFACE_G, DEEP_MUS, DEEP_G, DEEP_THICKNESS, N_INDEX, N_OUT, Nphotons, seed, Nbatches);

    same_sign = sign(dR_A) == sign(dR_B);
    if ~same_sign, flips = flips + 1; end

    printf("%10.2f  dR=%+.4f (%+.1fs)   dR=%+.4f (%+.1fs)   %s\n", ...
        t, dR_A, dR_A/sR_A, dR_B, dR_B/sR_B, ifelse(same_sign, "NO", "YES"));
end

printf("\n");
if flips == numel(thicknesses)
    printf("RESULT: sign flipped in %d/%d cases -- MATLAB independently CONFIRMS\n", flips, numel(thicknesses));
    printf("the Python finding: bias direction depends on absorber placement.\n");
else
    printf("RESULT: sign flipped in only %d/%d cases -- does NOT confirm the Python finding.\n", flips, numel(thicknesses));
end

function r = ifelse(cond, a, b)
    if cond, r = a; else, r = b; end
end
