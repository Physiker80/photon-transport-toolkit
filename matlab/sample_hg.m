function cos_theta = sample_hg(g)
% SAMPLE_HG  Sample cos(theta) from the Henyey-Greenstein phase function.
%
% p_HG(cos_theta) = (1-g^2) / (2*(1+g^2-2*g*cos_theta)^(3/2))
%
% Inverse-CDF sampling: for g ~= 0,
%   cos_theta = (1/(2g)) * (1+g^2 - ((1-g^2)/(1-g+2g*xi))^2)
% and for g == 0 (isotropic), cos_theta = 2*xi - 1.
%
% Independently derived from the HG phase function's known analytic
% inverse-CDF; not a port of any existing implementation.

    xi = rand();

    if abs(g) < 1e-6
        cos_theta = 2*xi - 1;
    else
        term = (1 - g^2) / (1 - g + 2*g*xi);
        cos_theta = (1/(2*g)) * (1 + g^2 - term^2);
    end

    % guard against floating-point drift outside [-1, 1]
    cos_theta = min(1, max(-1, cos_theta));
end
