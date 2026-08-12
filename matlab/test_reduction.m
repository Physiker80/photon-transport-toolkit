% test_reduction.m
% A stack of layers that all share identical optical properties and
% total thickness must reproduce the already-validated single-layer
% model's Rd/T/A within statistical uncertainty. This does not
% independently prove the layered algorithm correct, but proves the
% generalisation collapses correctly onto the validated special case.

mua = 0.1; mus = 6.0; g = 0.75; n = 1.4; n_out = 1.0;
total_thickness = 2.0;
Nphotons = 4000; Nbatches = 8; seed = 33;

r_homog = mc_slab(mua, mus, g, total_thickness, n, n_out, Nphotons, seed, Nbatches);

% Same medium split into 3 identical sub-layers (index-matched
% internally, so no artificial internal Fresnel reflection)
mua_v = [mua mua mua];
mus_v = [mus mus mus];
g_v   = [g g g];
th_v  = [total_thickness/3, total_thickness/3, total_thickness/3];
n_v   = [n n n];

r_layered = mc_layered(mua_v, mus_v, g_v, th_v, n_v, n_out, Nphotons, seed, Nbatches);

printf("Reduction test: 3 identical layers vs. homogeneous slab\n");
printf("  homogeneous : Rd=%.5f+/-%.5f  T=%.5f+/-%.5f  A=%.5f+/-%.5f\n", ...
    r_homog.Rd, r_homog.Rd_se, r_homog.T, r_homog.T_se, r_homog.A, r_homog.A_se);
printf("  layered(x3) : Rd=%.5f+/-%.5f  T=%.5f+/-%.5f  A=%.5f+/-%.5f\n", ...
    r_layered.Rd, r_layered.Rd_se, r_layered.T, r_layered.T_se, r_layered.A, r_layered.A_se);

sig_Rd = abs(r_homog.Rd - r_layered.Rd) / sqrt(r_homog.Rd_se^2 + r_layered.Rd_se^2);
sig_T  = abs(r_homog.T  - r_layered.T)  / sqrt(r_homog.T_se^2  + r_layered.T_se^2);
sig_A  = abs(r_homog.A  - r_layered.A)  / sqrt(r_homog.A_se^2  + r_layered.A_se^2);

printf("  deviation: Rd=%.2f sigma, T=%.2f sigma, A=%.2f sigma\n", sig_Rd, sig_T, sig_A);
if sig_Rd < 4 && sig_T < 4 && sig_A < 4
    printf("  PASS\n");
else
    printf("  FAIL\n");
end
