% test_energy_conservation.m
% With mua=0 (no absorption possible), every photon must eventually
% exit through either the top or bottom surface: Rd + T = 1 exactly.

mua = 0.0; mus = 8.0; g = 0.7; thickness = 2.0;
n_med = 1.4; n_out = 1.0;

r = mc_slab(mua, mus, g, thickness, n_med, n_out, 4000, 22, 8);

total = r.Rd + r.T + r.A;
se_total = sqrt(r.Rd_se^2 + r.T_se^2 + r.A_se^2);
diff_sigma = abs(total - 1.0) / se_total;

printf("Energy conservation test (mua=0)\n");
printf("  Rd=%.5f T=%.5f A=%.5f  sum=%.6f +/- %.6f\n", r.Rd, r.T, r.A, total, se_total);
printf("  A (should be 0)  = %.6f\n", r.A);
printf("  deviation from 1 = %.2f sigma\n", diff_sigma);
if diff_sigma < 4 && r.A < 1e-9
    printf("  PASS\n");
else
    printf("  FAIL\n");
end
