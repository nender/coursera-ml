function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
features = size(X, 2);
J_history = zeros(num_iters, 1);

function t = GDTerm(j)
  accumulator = 0;
  
  for i = 1:m
    xterm = X(i, :);
    hypothesis = xterm * theta;
    accumulator += (hypothesis - y(i)) * xterm(j);
  end
  
  t = (alpha / m) * accumulator;
end

for iter = 1:num_iters
    newtheta = zeros(size(theta));
    for j = 1:features
      newtheta(j) = GDTerm(j);
    end
    
    theta -= newtheta;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
