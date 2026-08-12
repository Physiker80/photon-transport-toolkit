% run_one_config.m
% Expects surface_mua, deep_mua, thickness, label already set in the
% workspace (via octave-cli --eval). Appends one result line to
% bias_results.txt.

SURFACE_MUS = 22.5; SURFACE_G = 0.80;
DEEP_MUS = 20.0; DEEP_G = 0.90;
DEEP_THICKNESS = 1.5;
N_INDEX = 1.4; N_OUT = 1.0;
Nphotons = 800; Nbatches = 4; seed = 5;

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

fid = fopen("bias_results.txt", "a");
fprintf(fid, "%s\tthickness=%.2f\tdR=%+.4f\tsR=%.4f\tsigma=%+.1f\n", label, thickness, dR, sR, dR/sR);
fclose(fid);

printf("%s  thickness=%.2f  dR=%+.4f +/- %.4f  (%+.1f sigma)\n", label, thickness, dR, sR, dR/sR);
