% run_one_g.m
% Expects g_val set in workspace. Appends one result line to sim_results.txt.

MUSP_FIXED = 8.0; THICKNESS = 2.0; MUA = 0.05;
N_MED = 1.4; N_OUT = 1.0;
Nphotons = 500; Nbatches = 4; seed = 3;

mus = MUSP_FIXED / (1-g_val);
r = mc_slab(MUA, mus, g_val, THICKNESS, N_MED, N_OUT, Nphotons, seed, Nbatches);

fid = fopen("sim_results.txt", "a");
fprintf(fid, "g=%.1f\tmus=%.3f\tRd=%.5f\tRd_se=%.5f\n", g_val, mus, r.Rd, r.Rd_se);
fclose(fid);

printf("g=%.1f  mus=%.3f  Rd=%.5f +/- %.5f\n", g_val, mus, r.Rd, r.Rd_se);
