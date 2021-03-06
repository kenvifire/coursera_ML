function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
for iter_1 = 1:size(z)(1)
	for iter_2 = 1:size(z)(2)
		v = -z(iter_1,iter_2);
		g(iter_1,iter_2) = 1 / (1 + e**v);
	end
end




% =============================================================

end
