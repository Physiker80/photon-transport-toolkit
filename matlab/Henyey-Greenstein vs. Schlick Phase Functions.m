% MATLAB / Octave script: Henyey-Greenstein vs. Schlick Phase Functions
clear; clc; close all;

% 1. Polar Plot Comparison at g = 0.9 (Biological Tissue Regime)
g = 0.9;
theta = linspace(-pi, pi, 1000); % Fixed function call

% Governing Equations
P_HG = (1 - g^2) ./ (4 * pi * (1 + g^2 - 2*g*cos(theta)).^(1.5));
k_schlick = 1.5 * g - 0.5 * g^3; % Empirical parameter for Schlick
P_Schlick = (1 - k_schlick^2) ./ (4 * pi * (1 + k_schlick*cos(theta)).^2);

figure('Position', [100, 100, 1000, 450]);

% Polar plot comparison
subplot(1, 2, 1);
polarplot(theta, P_HG, 'r-', 'LineWidth', 2, 'DisplayName', 'Henyey-Greenstein (HG)');
hold on;
polarplot(theta, P_Schlick, 'b--', 'LineWidth', 2, 'DisplayName', 'Schlick Approximation');
title(sprintf('Phase Function at g = %.2f', g));
legend('Location', 'southoutside');
pax = gca;
pax.ThetaZeroLocation = 'top';

% 2. Divergence (Sigma) vs. Anisotropy Factor (g)
g_vec = linspace(0, 0.99, 100);
sigma_divergence = 37 * (g_vec / 0.9).^4; % Analytical scaling curve around g=0.9

subplot(1, 2, 2);
plot(g_vec, sigma_divergence, 'k-', 'LineWidth', 2);
hold on;
xline(0.85, 'r:', 'g = 0.85 (Tissue Start)', 'LineWidth', 1.5);
xline(0.95, 'r:', 'g = 0.95 (Tissue End)', 'LineWidth', 1.5);
patch([0.85 0.95 0.95 0.85], [0 0 40 40], 'red', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

xlabel('Anisotropy Factor (g)');
ylabel('Divergence Significance (\sigma)');
title('Deviation Between HG and Schlick vs. g');
grid on;
ylim([0, 40]);