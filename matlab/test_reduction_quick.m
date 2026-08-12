mua = 0.1; mus = 6.0; g = 0.75; n = 1.4; n_out = 1.0;
total_thickness = 2.0;
Nphotons = 400; Nbatches = 6; seed = 33;

r_homog = mc_slab(mua, mus, g, total_thickness, n, n_out, Nphotons, seed, Nbatches);
mua_v = [mua mua mua]; mus_v=[mus mus mus]; g_v=[g g g];
th_v = [total_thickness/3, total_thickness/3, total_thickness/3]; n_v=[n n n];
r_layered = mc_layered(mua_v, mus_v, g_v, th_v, n_v, n_out, Nphotons, seed, Nbatches);

sig_Rd = abs(r_homog.Rd - r_layered.Rd) / sqrt(r_homog.Rd_se^2 + r_layered.Rd_se^2);
printf("homogeneous Rd=%.5f+/-%.5f  layered Rd=%.5f+/-%.5f  (%.2f sigma)\n", ...
    r_homog.Rd, r_homog.Rd_se, r_layered.Rd, r_layered.Rd_se, sig_Rd);
if sig_Rd < 4
    printf("PASS -- reduction test still holds after the refraction fix (matched-n fast path confirmed)\n");
else
    printf("FAIL\n");
end
