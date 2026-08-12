function [ux2, uy2, uz2] = rotate_direction(ux, uy, uz, cos_theta, phi)
% ROTATE_DIRECTION  Rotate a unit direction vector by a scattering
% angle theta (given as cos_theta) and azimuthal angle phi, using the
% standard spherical-rotation formulas about the current direction.
%
% Handles the near-axial case (|uz| ~ 1) separately to avoid the
% 0/0 singularity in the general formula.
%
% Independently derived from the standard direction-cosine rotation
% used in radiative-transport Monte Carlo; not a port of any existing
% implementation.

    sin_theta = sqrt(max(0, 1 - cos_theta^2));
    cos_phi = cos(phi);
    sin_phi = sin(phi);

    if abs(uz) > 0.99999
        ux2 = sin_theta * cos_phi;
        uy2 = sin_theta * sin_phi;
        uz2 = cos_theta * sign(uz);
    else
        denom = sqrt(1 - uz^2);
        ux2 = sin_theta * (ux*uz*cos_phi - uy*sin_phi) / denom + ux*cos_theta;
        uy2 = sin_theta * (uy*uz*cos_phi + ux*sin_phi) / denom + uy*cos_theta;
        uz2 = -sin_theta * cos_phi * denom + uz*cos_theta;
    end

    % renormalise against floating-point drift
    norm_val = sqrt(ux2^2 + uy2^2 + uz2^2);
    ux2 = ux2/norm_val; uy2 = uy2/norm_val; uz2 = uz2/norm_val;
end
