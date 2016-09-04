function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_theta = sigmoid(X*theta);
s = size(theta)(1);
J = sum((-y)' * log(h_theta) -(1-y)'*log(1-h_theta))/m + lambda/(2*m)*  sum(theta(2:s)'*theta(2:s));

%theta(0)
grad(1) = sum((h_theta-y).*X(:,1))/m;

%theta(1-->n)
for iter = 2:size(theta)(1)
	grad(iter) = sum((h_theta-y).*X(:,iter))/m + lambda* theta(iter)/m;
end




% =============================================================

end
