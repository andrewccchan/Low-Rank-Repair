function out = LADMM(D, tau, opts)

%% Clarify the paramenters:
% In paper:                            In this code:
% Y1 and Y2 : the coefficients      || Lambda : Lagrangian coefficient
%    of the Lagrangian              || 
% mu : step size for Y1 and Y2      || beta
% rho : step size for mu            || (omitted)
% lambda : lambda * ||W||           || tau : tau * ||W||, the penalty term
% D : The original image            || C
% E : Sparse matrix                 || A
% W : low rank matrix               || B = (C - A)
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
A = zeros(m,n);
Lambda = zeros(m,n);
if isfield(opts,'E0');  E = opts.E0; end
if isfield(opts,'A0');  A = opts.A0; end
if isfield(opts,'Lam0'); Lambda = opts.Lam0; end

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
eta = 3; % Use the same parameter as the paper
% Some reminders : 
fprintf('NOTICE : eta is omitted in updating A\n');
fprintf('mexsvd is replaced by svd\n');

if(size(Lambda, 1) < size(Lambda,2))
    dia = size(Lambda, 1);
else 
    dia = size(Lambda, 2);
end
% --------------------------------------

% main
for iter = 1:maxit
    
    nrmAB = norm([E,A],'fro');
      
    %% A - subproblem, in this case, E
    % Procedure of this snippest of code is to min.
    % the sparse term
    Y = Lambda / beta + D - B1 * A * B2';
    dE = E;
    E = sign(Y) .* max(abs(Y) - tau/beta, 0); % Equivalent to lambda/mu in the paper, eta is omitted
    dE = E - dE;
    
    %% B - subprolbme, in this case A (low rank)
    % Procedure of this function is to calculate the 
    % shrinkage operator
    Y = A - (B1' * (B1 * A * B2' + E - D + Lambda/beta) * B2)/eta;
    dA = A;
    [U,sig,VT] = svd(Y);
    VT = VT';
    sig = diag(sig);
    ind = find(sig > 1/beta);
    sig = diag(sig(ind) - 1/beta);
    A = U(:,ind) * sig * VT(ind,:);
    dA = A - dA;
    product = B1 * A * B2'; % This variable is used in latter code
    
    %% keep record
    if RECORD_ERRSP; errSP = norm(E - SP,'fro') / (1 + nrmSP); out.errsSP = [out.errsSP; errSP]; end
    if RECORD_ERRLR; errLR = norm(A - LR,'fro') / (1 + nrmLR); out.errsLR = [out.errsLR; errLR]; end
    if RECORD_OBJ;   obj = tau*norm(E(:),1) + sum(diag(D));    out.obj = [out.obj; obj];         end
    if RECORD_RES;   res = norm(E + A - D, 'fro');             out.res = [out.res; res];         end
    
    %% stopping criterion
    RelChg = norm([dE, dA],'fro') / (1 + nrmAB);
    if print, fprintf('Iter %d, RelChg %4.2e',iter,RelChg); end
    if print && RECORD_ERRSP && RECORD_ERRLR, fprintf(', errSP %4.2e, errLR %4.2e',errSP,errLR); end
    if print, fprintf('\n'); end
    if RelChg < tol, break; end
    
%     if(mod(iter, 10) ~= 0) 
        figure;
        subplot(1,2,1); imagesc(E); title('E'); 
        subplot(1,2,2); imagesc(product); title('A');
%     end
   
    %% Update Lambda, this line is crucial to the whole result
    Lambda = Lambda + beta * (E + product - D);
   %% Normalization, edited by Andrew 
    if(E ~= 0) 
        E = E ./ norm(E, 'fro');
    end
    if(A ~= 0)
        A = A ./ norm(A, 'fro');
    end
    if(Lambda ~= 0)
        Lambda = Lambda / norm(Lambda, 'fro');
    end    
end

% output
out.Sparse = E;
out.LowRank = A;
out.iter = iter;
out.exit = 'Stopp;ed by RelChg < tol';
if iter == maxit, out.exit = 'Maximum iteration reached'; end
end