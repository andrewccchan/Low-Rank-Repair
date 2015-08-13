function Low_Rank_Repair
%% The target of this code is to solve the following problem
% ----------------------------------------------------------
% min_A,E |A|_*+alpha*|E|_1
% subj. to B_1AB_2^T+E = D
% ----------------------------------------------------------
clear; clc;
%%
D_ori = imread('input_sim.png');
[m, n, r] = size(D_ori);
D = zeros(m ,n);
D_sum = zeros(m, n, 3);
% D = double(D(:,:,3) ./ 255); % Only consider the first channel

% D = D ./ norm(D, 'fro');

%% algorithm
for channel = 1 : 3
    % Decide the second input to the solver
    t = 0.02; % This term determines the penalty
    Omega = -ones(m,n);
    % Restrict omega region
%     for ct1 = 228 : 500
%         for ct2 = 258 : 615
%             Omega(ct1, ct2) = 1;
%         end
%     end


    D = double(D_ori(:,:,channel));
    figure; imagesc(D); pause;
%    D = D ./ norm(D, 'fro');
%     Dmax = max(D(:));
%     Dmin = min(D(:));
    
    opts = [];
    opts.beta = .25/mean(abs(D(:)));%0.10;
    opts.tol = 7e-3;
    opts.maxit = 100;
    opts.A0 = zeros(m,n);
    opts.E0 = zeros(m,n);
    opts.W0 = zeros(m,n);
    opts.Lam1 = zeros(m,n);
    opts.Lam2 = zeros(m,n);
    opts.print = 1;
    tolerance = [5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6];
    iter = 1;
while(1)
    opts.tol = tolerance(iter);
    if(iter > 6) break; end
    mask = @(M) Mask(M, Omega);
    figure; imagesc(Omega); title('Omega');
    out = LADMM(D, mask, t/(1-t), t/(1-t), opts); % Default value
   
%     L = abs(out.LowRank);
      L = out.LowRank;
%     Lmax = max(L(:));
%     Lmin = min(L(:));
%     L = imadjust(L, [Lmin; Lmax], [Dmin, Dmax]);
%     
%     E = abs(out.Sparse);
      E = out.Sparse;
%     Emax = max(E(:));
%     Emin = min(E(:));
%     E = imadjust(E, [Emin; Emax], [0, 1]); % This line is very important 
    
    
    suppE = UpdateOmega(Omega, E, 5, 0.0015);
    Omega = minusset(Omega, suppE);
    imagesc(L);
    
    % Update variables
    opts.A0 = out.A;
    opts.W0 = out.W;
    opts.E0 = out.Sparse;
    opts.Lam1 = out.Lam1;
    opts.Lam2 = out.Lam2;
    
    option = input('Press Enter to continue\n');
    if(option == 1) 
       break;
    end
    pause;
    close all;
    iter = iter + 1;
    D_sum(:,:,channel) = L;
end
end
% Expand output to it's original value, error term to 255

imwrite(D_sum/255, 'out.jpg');


function [ret] = minusset(Omega, suppE)
[m, n] = size(Omega);
patch = (suppE == ones(m,n));
patch = patch(:);
ret = Omega(:);
for count = 1 : length(patch)
    if(patch(count) == 1 && ret(count) == -1) 
        ret(count) = 1;
    end
end

ret = reshape(ret, [m n]);


%% End of the code
