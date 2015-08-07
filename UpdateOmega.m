function [ret] = UpdateOmega(Omega, E, bet, gam)
% Omega should be either 1 or -1
% E is in the range between 0~255
% bet is beta
% gam is gammma
addpath Bk_matlab;

[m, n] = size(Omega); % Get the size of the image
% Create graph
Graph = Bk_Create(m*n); % Create a undirected graph with n*m vertices
% Set Unary weight
[U1, U_1] = computeU(Omega, E, bet);
Bk_SetUary(Graph, [U_1' ; U1']);
% Set P
Bk_setNeighbors(Graph, computeP(Omega, gam));
% Calculate the mincut
ret = Bk_Minimize(Graph); % ret should be a 1-D vector
ret = reshape(ret, [n, m])'; 


function [U1, U_1] = computeU(Omega, E, bet)
% Vectorize matrices
Omega = Omega(:); 
E = E(:);
bet = bet * ones(length(E), 1);
U1 = (abs(E) <= bet & Omega == 1) * (log10(bet));

U_1 = (abs(E) <= bet & Omega == -1) * (-log10(bet)) + ...
	(abs(E) > bet & Omega == -1) * (log10(bet));


function [P] = computeP(Omega, gam)

[m, n] = size(Omega);
P = zeros(m*n, m*n);
Omega = Omega(:);

for count = 1 : m*n
	row = floor(count/n) + 1;
	col = mod(count, n);
	right = (mod(count, n) == 0);
	bottom = (count-(n*(m-1)) <= n);

	if(right && ~bottom)
		P(row, col+n) = gam * Omega(count) * Omega(count+n);
	else
		if(~right && bottom)
			P(row, col+1) = gam * Omega(count) * Omega(count+1);
		else
			P(row, col+1) = gam * Omega(count) * Omega(count+1);
			P(row, col+n) = gam * Omega(count) * Omega(count+n);
		end
	end
	
end



