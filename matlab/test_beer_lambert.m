% test_beer_lambert.m
% In the no-scattering limit (mus=0) with index-matched boundaries
% (no Fresnel loss), transmittance must equal exp(-mua*thickness)
% exactly (up to Monte Carlo noise): every photon travels straight
% through and is attenuated purely by absorption.

mua = 0.8; mus = 0.0; g = 0.0; thickness = 1.5;
n_med = 1.0; n_out = 1.0;   % index-matched: no Fresnel reflection anywhere

r = mc_slab(mua, mus, g, thickness, n_med, n_out, 4000, 11, 8);

T_expected = exp(-mua*thickness);
diff_sigma = abs(r.T - T_expected) / r.T_se;

printf("Beer-Lambert test\n");
printf("  T (Monte Carlo) = %.5f +/- %.5f\n", r.T, r.T_se);
printf("  T (exp(-mua*d)) = %.5f\n", T_expected);
printf("  deviation        = %.2f sigma\n", diff_sigma);
if diff_sigma < 4
    printf("  PASS\n");
else
    printf("  FAIL\n");
end
