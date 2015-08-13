function [ret] = UpdateOmega(Omega, E, bet, gam)
% Omega should be either 1 or -1
% E is in the range between 0~255
% bet is beta
% gam is gammma
% -----------------------------------
% Clarify the parameters :
% 	This program 		||			In Paper
%			1 			||				-1
% 			2 			||				 1
% 
% SetUnary will treat nth row as label n
% SetPairwise have a different notation which uses 0,1 system, in
% which 0 means 1 (cut btw. source) and 1 means 2 (cut btw. sink).
addpath Bk_Lib

[m, n] = size(Omega); % Get the size of the image
% Create graph
Graph = BK_Create(m*n); % Create a undirected graph with n*m vertices
% Set Unary weight
[U1, U_1] = computeU(Omega, E, bet);
BK_SetUnary(Graph, [U_1' ; U1']);
% Set P
BK_SetPairwise(Graph, computeP(Omega, gam));
% Calculate the mincut
BK_Minimize(Graph) % ret should be a 1-D vector
label = BK_GetLabeling(Graph);
ret = (label == 1) .* -1 + (label == 2);
ret = reshape(ret, [n m])';
BK_Delete(Graph);

%% ComputeU  will return the U matrix
% U1 indicates that the pixel is marked 1
% U_1 means the pixel is marked -1
function [U1, U_1] = computeU(Omega, E, bet)
% Vectorize matrices
Omega = Omega';
E = E';
Omega = Omega(:); 
E = E(:);
bet = bet * ones(length(E), 1);
U1 = (abs(E) <= bet) .* (log(bet)); % Omega == 1

U_1 = (abs(E) <= bet) .* (-log(bet)) + (abs(E) > bet) .* (log(bet));

%% ComputeP. This function should return a edge# x 6 matrix
function [P] = computeP(Omega, gam)

[m, n] = size(Omega);
edgenum = (m - 1) * n + (n - 1) * m; % number of edges
P = zeros(edgenum, 6);
Omega = Omega(:);

ct2 = 1;
for count = 1 : m*n
	col = floor(count/m) + 1;
	row = mod(count, m);
	bottom = (mod(count, m) == 0);
	right = ((n*m - count) < m);

	if(right && ~bottom)
		P(ct2, :) = [count count+1 gam -gam -gam  gam];
		ct2 = ct2 + 1;
	else
		if(~right && bottom)
			P(ct2, :) = [count, count+m gam -gam -gam gam];
			ct2 = ct2 + 1;
        else 
            if(~right && ~bottom)
			P(ct2, :) = [count count+1 gam -gam -gam gam];
			P(ct2+1, :) = [count count+m gam -gam -gam gam];
			ct2 = ct2 + 2;
            end
		end
	end
	
end

if(ct2 ~= edgenum+1)
	error('ct2 and edgenum mistach');
end







