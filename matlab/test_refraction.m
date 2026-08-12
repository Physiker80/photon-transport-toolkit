% test_refraction.m
% Validates the Snell's-law direction-refraction fix at internal layer
% boundaries with genuinely mismatched refractive indices -- added in
% response to an external code-review claim that this project's
% mc_layered.m lacked refraction entirely. Verified: the claim was
% correct (confirmed by code inspection before this fix existed), and
% had zero effect on any previously published result in this project,
% since every prior test/example used matched n across all layers
% (where Snell's law reduces to no bending, and fresnel_reflectance()
% correctly returns R=0).

printf("=== Test 1: Snell's law satisfied exactly ===\n");
n1 = 1.0; n2 = 1.5; theta_i = 30*pi/180;
ux = sin(theta_i); uy = 0; uz = cos(theta_i);
[ux2, uy2, uz2] = refract_direction(ux, uy, uz, n1, n2);
theta_t = acos(abs(uz2));
lhs = n1*sin(theta_i); rhs = n2*sin(theta_t);
printf("  n1*sin(theta_i)=%.6f  n2*sin(theta_t)=%.6f  (diff=%.2e)\n", lhs, rhs, abs(lhs-rhs));
printf("  unit-vector check: %.8f\n", ux2^2+uy2^2+uz2^2);
assert(abs(lhs-rhs) < 1e-9);
assert(abs(ux2^2+uy2^2+uz2^2 - 1) < 1e-9);
printf("  PASS\n\n");

printf("=== Test 2: energy conservation with mismatched n=[1.0,1.6,1.3] ===\n");
r = mc_layered([0 0 0], [8.0 6.0 8.0], [0.7 0.6 0.7], [0.3 0.5 0.3], [1.0 1.6 1.3], 1.0, 1200, 5, 6);
total = r.Rd + r.T + r.A;
se = sqrt(r.Rd_se^2 + r.T_se^2 + r.A_se^2);
printf("  Rd=%.5f T=%.5f A=%.5f sum=%.6f (%.2f sigma from 1.0)\n", r.Rd, r.T, r.A, total, abs(total-1)/se);
assert(abs(total-1)/se < 4);
printf("  PASS\n\n");
printf("All refraction tests passed.\n");
