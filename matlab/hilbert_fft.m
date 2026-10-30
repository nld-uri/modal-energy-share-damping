function z = hilbert_fft(x)
%HILBERT_FFT  Analytic signal via FFT -- toolbox-free scipy.signal.hilbert.
%
%   Replaces scipy.signal.hilbert (used by bias_map, share_law,
%   recovery_recipe, friction_boundary, noise_sensing, bandwidth_axis).
%   MATLAB's own hilbert() is Signal Processing Toolbox, which this package
%   does not assume.
%
%   The analytic signal is ifft(fft(x) .* h) with
%
%       N even :  h(1) = 1,  h(N/2+1) = 1,  h(2:N/2)     = 2,  rest 0
%       N odd  :  h(1) = 1,                 h(2:(N+1)/2) = 2,  rest 0
%
%   The even case is the trap: the DC bin AND the Nyquist bin keep weight 1
%   while every other positive-frequency bin is doubled.  Getting that wrong
%   gives an envelope that looks plausible and is quietly wrong, which is why
%   selftest.m checks both parities against scipy's output on a chirp.
%
%   Operates on a single real (or complex) vector; returns a column.

x = x(:);
N = numel(x);
h = zeros(N,1);
if mod(N,2) == 0
    h(1) = 1;
    h(N/2+1) = 1;
    h(2:N/2) = 2;
else
    h(1) = 1;
    h(2:(N+1)/2) = 2;
end
z = ifft(fft(x).*h);
end
