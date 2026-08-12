% run_thin.m
MUA = 0.05; N_MED = 1.4; N_OUT = 1.0; THIN = 0.05;
Nphotons = 500; Nbatches = 4; seed = 3;
MUSP_FIXED = 8.0;

mus = MUSP_FIXED / (1-g_val);
r = mc_slab(MUA, mus, g_val, THIN, N_MED, N_OUT, Nphotons, seed, Nbatches);

fid = fopen("sim_thin_results.txt", "a");
fprintf(fid, "g=%.1f\tRd=%.5f\n", g_val, r.Rd);
fclose(fid);
printf("thin slab, g=%.1f  Rd=%.5f\n", g_val, r.Rd);
