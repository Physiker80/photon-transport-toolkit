function [ux2, uy2, uz2] = refract_direction(ux, uy, uz, n1, n2)
% REFRACT_DIRECTION  Update a photon's direction cosines upon
% transmission through a planar (z-normal) boundary between two media
% of refractive index n1 (incident side) and n2 (transmitted side),
% following Snell's law n1*sin(theta_i) = n2*sin(theta_t).
%
% Standard MCML-style direction-cosine update (Wang, Jacques & Zheng
% 1995): the transverse components scale by eta=n1/n2, and uz is
% recomputed from the transmission angle so the result stays a unit
% vector by construction:
%
%   ux2 = ux * eta
%   uy2 = uy * eta
%   uz2 = sign(uz) * sqrt(1 - eta^2*(1-uz^2))
%
% Only valid to call when transmission has already been confirmed
% (i.e. not total internal reflection) -- callers must check
% fresnel_reflectance() first.
%
% Independently derived from Snell's law in direction-cosine form; not
% a port of any existing implementation.

    if n1 == n2
        ux2 = ux; uy2 = uy; uz2 = uz;
        return;
    end

    eta = n1 / n2;
    cos_i = abs(uz);
    sin_i2 = 1 - cos_i^2;
    sin_t2 = eta^2 * sin_i2;
    sin_t2 = min(sin_t2, 1.0);  % guard against float drift at grazing incidence

    cos_t = sqrt(1 - sin_t2);

    ux2 = ux * eta;
    uy2 = uy * eta;
    uz2 = sign(uz) * cos_t;

    % renormalise against floating-point drift
    norm_val = sqrt(ux2^2 + uy2^2 + uz2^2);
    ux2 = ux2 / norm_val; uy2 = uy2 / norm_val; uz2 = uz2 / norm_val;
end
