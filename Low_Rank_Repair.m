function Low_Rank_Repair
%% The target of this code is to solve the following problem
% ----------------------------------------------------------
% min_A,E |A|_*+alpha*|E|_1
% subj. to B_1AB_2^T+E = D
% ----------------------------------------------------------
clear; clc;
%%
D = imread('input.png');
[m, n, r] = size(D);
D = double(D(:,:,1)); % Only consider the first channel
Dmax = max(D(:))/255;
Dmin = min(D(:))/255;
D = D ./ norm(D, 'fro');

% Decide the second input to the solver
t = 0.05; % This term determines the penalty
Omega = -ones(m,n);
mask = @(M) Mask(M, Omega);
%% algorithm
while(1)
    opts = [];
    opts.beta = .25/mean(abs(D(:)));%0.10;
    opts.tol = 7e-3;
    opts.maxit = 1000;
    opts.A0 = zeros(m,n);
    opts.B0 = zeros(m,n);
    opts.W0 = zeros(m,n);
    opts.Lam1 = zeros(m,n);
    opts.Lam2 = zeros(m,n);
    opts.print = 1;
    out = LADMM(D, mask, t/(1-t), t/(1-t), opts); % Default value

    L = abs(out.LowRank);
    Lmax = max(L(:));
    Lmin = min(L(:));
    L = imadjust(L, [Lmin; Lmax], [Dmin, Dmax]);
    
    E = abs(out.Sparse);
    Emax = max(E(:));
    Emin = min(E(:));
    E = imadjust(E, [Emin; Emax], [0, 1]);
    
    
    suppE = UpdateOmega(Omega, E .* 255, 5, 0.015);
    Omega = minusset(Omega, suppE);
    imshow(L*255);
end
% Expand output to it's original value, error term to 255

function [ret] = minusset(Omega, suppE)
[m, n] = size(Omega);
patch = (suppE == ones(m,n));
patch = patch(:);
ret = Omega;
for count = 1 : length(patch)
    if(patch(count) == 1) 
        ret(count) = 1;
    end
end

ret = reshape(ret, [n, m])';


%% End of the code
