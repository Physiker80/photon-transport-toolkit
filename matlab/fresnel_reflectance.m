function R = fresnel_reflectance(n1, n2, cos_theta_i)
% FRESNEL_REFLECTANCE  Unpolarised Fresnel reflection coefficient.
%
% R = fresnel_reflectance(n1, n2, cos_theta_i)
%
% n1, n2        : refractive indices of the incident and transmitted media
% cos_theta_i   : cosine of the angle of incidence (>= 0)
%
% Returns R = 0.5*(Rs + Rp), the average of the s- and p-polarised
% reflectances, from Snell's law and the Fresnel equations directly.
% Returns R = 1 for total internal reflection.
%
% Independently derived from the standard Fresnel equations; not a
% port of any existing implementation.

    if n1 == n2
        R = 0;
        return;
    end

    sin_theta_i2 = 1 - cos_theta_i^2;
    sin_theta_t2 = (n1/n2)^2 * sin_theta_i2;

    if sin_theta_t2 >= 1
        R = 1;  % total internal reflection
        return;
    end

    cos_theta_t = sqrt(1 - sin_theta_t2);

    rs = (n1*cos_theta_i - n2*cos_theta_t) / (n1*cos_theta_i + n2*cos_theta_t);
    rp = (n2*cos_theta_i - n1*cos_theta_t) / (n2*cos_theta_i + n1*cos_theta_t);

    R = 0.5 * (rs^2 + rp^2);
end
