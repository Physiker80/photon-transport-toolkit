% test_similarity_relation.m
% Independent check: a medium with (mus, g) should be statistically
% equivalent, at fixed reduced scattering mus'=mus*(1-g), to one with
% g'=0 -- in the diffusive regime. A companion check confirms this
% breaks down for a thin, non-diffusive slab.

MUSP_FIXED = 8.0; THICKNESS = 2.0; MUA = 0.05;
N_MED = 1.4; N_OUT = 1.0;
Nphotons = 700; Nbatches = 4; seed = 3;

g_values = [0.0, 0.5, 0.8];
Rd_vals = zeros(1,3); Rd_se_vals = zeros(1,3);

printf("Similarity relation test (diffusive slab, thickness=%.1f mm)\n", THICKNESS);
for i = 1:3
    g = g_values(i);
    mus = MUSP_FIXED / (1-g);
    r = mc_slab(MUA, mus, g, THICKNESS, N_MED, N_OUT, Nphotons, seed, Nbatches);
    Rd_vals(i) = r.Rd; Rd_se_vals(i) = r.Rd_se;
    printf("  g=%.1f  mus=%.2f  Rd=%.4f +/- %.4f\n", g, mus, r.Rd, r.Rd_se);
end

tol = 6 * max(Rd_se_vals);
dev1 = abs(Rd_vals(2)-Rd_vals(1));
dev2 = abs(Rd_vals(3)-Rd_vals(1));
printf("  |Rd(g=0.5)-Rd(g=0)| = %.4f (tol %.4f)\n", dev1, tol);
printf("  |Rd(g=0.8)-Rd(g=0)| = %.4f (tol %.4f)\n", dev2, tol);
pass1 = dev1 < tol && dev2 < tol;
printf("  %s\n\n", ifelse(pass1, "PASS (g-invariant, as expected)", "FAIL"));

% Breakdown check: thin, non-diffusive slab
THIN = 0.05;
r_iso = mc_slab(MUA, MUSP_FIXED/(1-0.0), 0.0, THIN, N_MED, N_OUT, Nphotons, seed, Nbatches);
r_fwd = mc_slab(MUA, MUSP_FIXED/(1-0.9), 0.9, THIN, N_MED, N_OUT, Nphotons, seed, Nbatches);
printf("Breakdown check (thin slab, thickness=%.2f mm)\n", THIN);
printf("  g=0.0  Rd=%.4f\n", r_iso.Rd);
printf("  g=0.9  Rd=%.4f\n", r_fwd.Rd);
diff_thin = abs(r_iso.Rd - r_fwd.Rd);
printf("  difference = %.4f (expect > 0.02, i.e. similarity relation breaks down)\n", diff_thin);
pass2 = diff_thin > 0.02;
printf("  %s\n", ifelse(pass2, "PASS (correctly breaks down)", "FAIL"));

function r = ifelse(cond, a, b)
    if cond, r = a; else, r = b; end
end
