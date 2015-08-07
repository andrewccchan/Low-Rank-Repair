function Low_Rank_Repair
%%
clear; clc;
%%
D = imread('input.png');
D = double(D(:,:,1));
% D = D ./ norm(D, 'fro');
% D = D - ones(size(D,1), 1) * mean(D);
% D = D ./ (ones(size(D,1), 1) * sqrt(sum(D.^2)));
% D = [1 1 1; 1 0 1; 1 1 1];
[m, n, r] = size(D);
% Paramenter initialization
Omega = -ones(m ,n);
B1 = dctmtx(m)'; % DCT orthogonal mxm matrix
B2 = dctmtx(n)'; % DCT orthogonal nxn matrix
mask = @(M) Mask(M, Omega); % Function handle for Mask
lambda = 0.001;
alpha = 0.85;
rho = 1.9; %This parameter is not specified in the papaer, however it should  > 1
gam = 0.015;
bet = 5;

% Iterate till it gives stisfying results
D = double(D(:, :, 1));
while true
	[A, W, E] = LADMM(D, B1, B2, mask, lambda, alpha, rho, 1);
	suppE = UpdateOmega(Omega, E, bet, gam);
	Omega = minusset(Omega, suppE);
	imshow(A);
end

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

