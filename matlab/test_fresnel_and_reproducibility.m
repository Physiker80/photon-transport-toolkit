% test_fresnel_and_reproducibility.m

% -- Fresnel at normal incidence: check the deterministic Rsp against
%    the textbook closed-form ((n1-n2)/(n1+n2))^2
n1 = 1.0; n2 = 1.4;
R_analytic = ((n1-n2)/(n1+n2))^2;
R_code = fresnel_reflectance(n1, n2, 1.0);
printf("Fresnel normal-incidence test\n");
printf("  analytic  = %.6f\n", R_analytic);
printf("  fresnel_reflectance() = %.6f\n", R_code);
if abs(R_analytic - R_code) < 1e-12
    printf("  PASS\n\n");
else
    printf("  FAIL\n\n");
end

% -- Reproducibility: identical seed must give bit-identical results
r1 = mc_slab(0.1, 5.0, 0.8, 2.0, 1.4, 1.0, 1000, 77, 3);
r2 = mc_slab(0.1, 5.0, 0.8, 2.0, 1.4, 1.0, 1000, 77, 3);
printf("Reproducibility test (fixed seed)\n");
printf("  run 1: Rd=%.8f T=%.8f A=%.8f\n", r1.Rd, r1.T, r1.A);
printf("  run 2: Rd=%.8f T=%.8f A=%.8f\n", r2.Rd, r2.T, r2.A);
if r1.Rd == r2.Rd && r1.T == r2.T && r1.A == r2.A
    printf("  PASS (bit-identical)\n");
else
    printf("  FAIL\n");
end
