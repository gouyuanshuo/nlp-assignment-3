function values = interp1q(x, y, xi)
% MATLAB's deprecated interp1q is linear interpolation with NaN outside
% the source range; Octave interp1 implements the same contract.
values = interp1(x, y, xi, 'linear');
end
