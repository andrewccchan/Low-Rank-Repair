function out = LADMM(D, func, tau, alpha, opts)
addpath PROPACK;
global Y
%% Clarify the paramenters:
% In paper:                            In this code:
% Y1 and Y2 : the coefficients      || Lambda : Lagrangian coefficient
%    of the Lagrangian              || 
% mu : step size for Y1 and Y2      || beta
% rho : step size for mu            || (omitted)
% lambda : lambda * ||W||           || alpha : alpha * ||E||, the penalty term
% D : The original image            || D
% E : Sparse matrix                 || E
% W : low rank matrix, sparse term  || W
% A : lwo rank matix, low rank term || A
% tau : the coeff. of W
% alpha : the coeff. of E
beta = .25/mean(abs(D(:)));
tol = 1.e-6;
maxit = 1000;
print = 0;
if isfield(opts,'beta'); beta = opts.beta; end
if isfield(opts,'tol'); tol = opts.tol; end
if isfield(opts,'maxit'); maxit = opts.maxit; end
if isfield(opts,'print'); print = opts.print; end

%% initialization
[m,n] = size(D);
E = zeros(m,n);
Ap = zeros(m,n);
A = [];
A.U = zeros(m,5);
A.s = zeros(5,1);
A.V = zeros(n,5);
W = zeros(m,n);
Lambda1 = zeros(m,n);
sv = 5;
svp = sv;
svdopt = [];
svdopt.tol = 1e-8;
if isfield(opts,'E0');  E = opts.E0; end
if isfield(opts,'A0');  Ap = opts.A0; end
if isfield(opts,'W0');  W = opts.W0; end
if isfield(opts,'Lam1'); Lambda1 = opts.Lam1; end
if isfield(opts, 'Lam2'); Lambda2 = opts.Lam2; end
%% keep record
% Not used in this demo
RECORD_ERRSP = 0; RECORD_ERRLR = 0; RECORD_OBJ = 0; RECORD_RES = 0;
if isfield(opts,'Sparse'); SP = opts.Sparse; nrmSP = norm(SP,'fro'); out.errsSP = 1; RECORD_ERRSP = 1; end
if isfield(opts,'LowRank'); LR = opts.LowRank; nrmLR = norm(LR,'fro'); out.errsLR = 1; RECORD_ERRLR = 1; end
if isfield(opts,'record_obj'); RECORD_OBJ = 1; out.obj = []; end
if isfield(opts,'record_res'); RECORD_RES = 1; out.res = []; end

%---------------------------------------
% Variables here are defined by myself 
B1 = dctmtx(m)'; % DCT orthogonal mxm matrix
B2 = dctmtx(n)'; 
eta1 = 3; % Use the same parameter as the paper
eta2 = 3;
% Some reminders : 
fprintf('Beaware of the update order\n');
if(size(Lambda1, 1) < size(Lambda1,2))
    dia = size(Lambda1, 1);
else  
    dia = size(Lambda1, 2);
end
% --------------------------------------
BWB = B1 * W * B2;
% main
for iter = 1:maxit
    nrmAEW = norm([E,Ap,W],'fro');
    %% W - subproblem
    Y = W - (B1' * (func(BWB + E - D + Lambda2/beta)) * B2 + W - Ap - Lambda1/beta)/eta1;
    dW = W;
    W = sign(Y) .* max(abs(Y) - tau/beta/eta1, 0); % Equivalent to lambda/mu in the paper, eta is omitted
    dW = W - dW;    
    BWB = B1 * W * B2';
    
    %% E - subproblem
    % Procedure of this snippest of code is to min.
    % the sparse term
    Y = E - ( func(E + BWB - D) + Lambda2/beta )/eta2;
    dE = E;
    E = sign(Y) .* max(abs(Y) - alpha/beta/eta2, 0); % Equivalent to lambda/mu in the paper, eta is omitted
    dE = E - dE;
    %% B - subprolbme, in this case A (low rank)
    % Procedure of this function is to calculate the 
    % shrinkage operator

    dAp = Ap;
    Y = W - Lambda1/beta;
%     [U, S, V] = lansvd('Y', 'Yt', m, n, sv, 'L', svdopt);
    if choosvd(n, sv) == 1
%     [U, S, V] = lansvd(Y, sv, 'L');
    [U, S, V] = lansvd('Y', 'Yt', m, n, sv, 'L', svdopt);
    else 
    [U, S, V] = svd(Y, 'econ');
    end
%     [U, S, V] = svd(W - Lambda1/beta);
    S = diag(S);
    svp = length(find(S>1/(beta)));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    if svp>=1
        S = S(1:svp)-1/(beta);
    else
        svp = 1;
        S = 0;
    end
    
    A.U = U(:, 1:svp);
    A.s = S;
    A.V = V(:, 1:svp);
    Ap = A.U * diag(A.s);
    Ap = Ap * A.V';
%     A = U * (sign(sig).*max(abs(sig) - 1/beta, 0)) * VT';
    dA = Ap - dAp;
    product = B1 * A.U;
    product = product * diag(A.s);
    product = product * A.V'*B2'; % This variable is used in latter code
    %% keep record
    if RECORD_ERRSP; errSP = norm(E - SP,'fro') / (1 + nrmSP); out.errsSP = [out.errsSP; errSP]; end
    if RECORD_ERRLR; errLR = norm(Ap - LR,'fro') / (1 + nrmLR); out.errsLR = [out.errsLR; errLR]; end
    if RECORD_OBJ;   obj = alpha*norm(E(:),1) + sum(diag(D));    out.obj = [out.obj; obj];         end
    if RECORD_RES;   res = norm(E + Ap - D, 'fro');             out.res = [out.res; res];         end
    
    %% stopping criterion
    RelChg = norm([dE, dA, dW],'fro') / (1 + nrmAEW);
    if print, fprintf('Iter %d, RelChg %4.2e',iter,RelChg); end
    if print && RECORD_ERRSP && RECORD_ERRLR, fprintf(', errSP %4.2e, errLR %4.2e',errSP,errLR); end
    if print, fprintf('\n'); end
    if (RelChg < tol) break; end
    
%      if(mod(iter, 10) == 0) 
%         figure;
%         subplot(1,2,1); imagesc(E); title('E'); 
%         subplot(1,2,2); imagesc(product); title('A');
%      end
   
    %% Update Lambda, these lines are crucial to the results
    % !!!Questions
    
    Lambda2 = Lambda2 + beta * func(E + product - D);
    Lambda1 = Lambda1 + beta * (Ap - W);
%     beta = beta * 1.1;
     %% Normalization, edited by Andrew 
%     if(W ~= 0)
%         W = W ./ norm(W, 'fro');
%     end
%     if(E ~= 0) 
%         E = E ./ norm(E, 'fro');
%     end
%     if(A ~= 0)
%         A = A ./ norm(A, 'fro');
%     end
%     if(Lambda1 ~= 0)
%         Lambda1 = Lambda1 / norm(Lambda1, 'fro');
% %         beta = beta * 1.1;
%     end 
%     if(Lambda2 ~= 0)
%         Lambda2 = Lambda2 / norm(Lambda2, 'fro');
%     end
    
%     beta = beta * 1.3;
end

% output
out.Sparse = E;
out.LowRank = product;
out.W = W;
out.A = Ap;
out.Lam1 = Lambda1;
out.Lam2 = Lambda2;
out.iter = iter;
out.exit = 'Stopped by RelChg < tol';
if iter == maxit, out.exit = 'Maximum iteration reached'; end
end