function [ret] = Mask(M, Omega)
% Omega is defined as follows,
% if Omega is black then it has value = -1
% otherwise				 it has value =  1
%
% This function sets node in M to 0 if its corresponding Omega = 1;
[m, n] = size(Omega);
ret = M - M .* (Omega == ones(m,n));