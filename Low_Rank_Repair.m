function LADMM_Demo
%% The target of this code is to solve the following problem
% ----------------------------------------------------------
% min_A,E |A|_*+alpha*|E|_1
% subj. to B_1AB_2^T+E = D
% ----------------------------------------------------------
clear; clc;
%%
D = imread('input_old.png');
D = double(D(:,:,1)); % Only consider the first channel
[m, n, r] = size(D);
D = D ./ norm(D, 'fro');
% Decide the second input to the solver
t = 0.05; % This term determines the penalty

%% algorithm
opts = [];
opts.beta = .25/mean(abs(D(:)));%0.10;
opts.tol = 1e-6;
opts.maxit = 1000;
opts.A0 = zeros(m,n);
opts.B0 = zeros(m,n);
opts.Lam0 = zeros(m,n);
opts.print = 1;
out = LADMM_r2(D, t/(1-t), opts); % Default value



%% End of the code
