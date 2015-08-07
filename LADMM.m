function [A, W, E] = LADMM(D, B1, B2, func, lambda, alpha, rho, DEBUG)

% This MATLAB code implements the LADMM solver for
% the algorithm in the paper "Repairing Sparse Low
% -Rank Texture" by Prof. Ma Yi.
%-------------------------------------------------
% min_A,W,E |A|_*+lambda*|W|_1+alpha*|E|_1
% subj. to A=W, P_Omega(B_1WB_2^T+E)=P_\Omage(D)
%-------------------------------------------------
% 
% created by Andrew Chan on 07/28/2015

% addpath PROPACK

if nargin < 8
    DEBUG = 0;
end
%% Initialize variables
tol1 = 1e-4; %tolerance of the error in constraint
tol2 = 1e-5; %tolerance of the error in the solutions
[c, r] = size(D); %size of the input image D
% opt.tol = tol2; %precision for computing the SVD
% opt.p0 = ones(n, 1);
if(DEBUG)
    maxIter = 50;
else
    maxIter = 10000;
end
max_mu = 1e10; 
% The initial value of mu will not influence the result after
% several iterations
mu = min(c,r)*tol2; % mu is the penalty in the Lagrangian
eta1 = 3;
eta2 = 3; % Use the same eta as the paper does
converged = 0;

%% Initializing optimization variables
% E = sparse(c, r);
E = zeros(c, r);
A = zeros(c, r);
W = zeros(c, r);
Y1 = zeros(c, r);
Y2 = zeros(c, r);
iter = 0;

%% Start the main loop
figure;
while iter<maxIter
    if(DEBUG)
        fprintf('Start new iteration\n');
    end
	iter = iter + 1;

	% Save the initial A, W, and E to compute the change in them later
	Ai = A;
	Wi = W;
	Ei = E;
    
	BWB = B1 * W * B2'; % To prevent from computing too many times
	% Update the variables
	% Update A, which should be a sparse image
    A = shrinkop(1/mu, W-Y1/mu);
	% Update W, which 
	W = evaluateT(lambda/mu/eta1, W - (B1' * (func(BWB...
+ E - D) + Y2/mu) * B2 + W - A  - Y1/mu)/eta1);
	% Update E
    BWB = B1 * W * B2';
	E = evaluateT(alpha/mu/eta2, E - (func(E + BWB - D)...
+ Y2/mu)/eta2);

%     if(DEBUG && mod(iter, 10) == 0)
%         
%         subplot(maxIter/10,3,1+3*((iter/10)-1)); imshow(mat2gray(A*255)); title('A');
%         subplot(maxIter/10,3,2+3*((iter/10)-1)); imshow(mat2gray(B1 * W * B2' * 255)); title('I'); 
%         subplot(maxIter/10,3,3+3*((iter/10)-1)); imshow(mat2gray(E * 255)); title('E');
%     end
    figure;
    subplot(1,4,1); imshow(mat2gray(A)); title('A');
    subplot(1,4,2); imshow(mat2gray(W)); title('W');
    subplot(1,4,3); imshow(mat2gray(B1 * W * B2')); title('I'); 
    subplot(1,4,4); imshow(mat2gray(E)); title('E');
	% Caculate the errors for evaluating convergence
	% KKT condition 1, feasibility
	BWB = B1 * W * B2'; % Need to update to the new value
	maskl = func(BWB); % Masked left value;
	maskr = func(D); % Masked right value;
	normfW = norm(W); % Norm of W
	normfD = norm(maskr, 'fro'); % Norm of D
	Errc1 = norm(A - W, 'fro')/normfW;
	Errc2 = norm(maskl - maskr, 'fro')/normfD;
	% KKT condition 2
	normmin = min(normfW, normfD);
	ChgA = norm(A - Ai, 'fro')/normmin;
	ChgW = norm(W - Wi, 'fro')/normmin;
	ChgE = norm(E - Ei, 'fro')/normmin;
    
    if(DEBUG)
        Error1 = max(Errc1, Errc2);
        Error2 = max(max(ChgA, ChgW), ChgE);
        fprintf('(Error1, Error2) = (%f, %f)\n', Error1, Error2);
    end
    
	converged = max(Errc1, Errc2) < tol1 && max(max(ChgA, ChgW), ChgE) < tol2;

	if converged
		break;
	else
		Y1 = Y1 + mu * func(BWB + E -D);
		Y2 = Y2 + mu * (A - W);
        mu = mu * rho;
% 		if( mu * max(max(ChgA, ChgW), ChgE) < tol2)
% 			mu = min(max_mu, mu*rho);
% 		end
    end
    
    %% TEST: normalized by Frobenius norm
%     if(A ~= 0) A = A ./ norm(A, 'fro'); end
%     if(A ~= 0) W = W ./ norm(W, 'fro'); end
%     if(A ~= 0) E = E ./ norm(E, 'fro'); end

end
	

function [ret] = shrinkop(epsi, M)
% Returns the result of singular value shrinkage operator
% epsi is the epsilon of the shrinkage operator
% M is the input matrix
	[U, sig, V] = svd(M); %Use full svd here
	% Use partial svd to speed up
	ret = U * (evaluateT(epsi, sig) .* eye(size(sig))) * V'; %Computation expensive, up to O(n^3)

function [ret] = evaluateT(epsi, M)
% This function evaluates the scalar shrinkage operator given input matrix M
% epsi is the epsilon of the shrinkage operator
% M is the input matrix]
	 ret = sign(M) .* max((abs(M)-epsi), 0);
	% ret = max(M - epsi, 0);


