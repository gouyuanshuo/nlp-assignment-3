function [values, locations] = findpeaks(x)
% COVAREP calls MATLAB findpeaks(x) only, on signed waveforms. Octave's
% version rejects negative inputs. Use COVAREP's bundled Voicebox peak
% detector, swapping its documented [indices,values] output order.
[locations, values] = v_findpeaks(x);
end
