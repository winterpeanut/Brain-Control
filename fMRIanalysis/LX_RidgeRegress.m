
function beta = LX_RidgeRegress(X,Y,lambdas,nfold)


% Input data
% sx = size(x);sy = size(y);
% foldidxx = find(sx==nfold);
% foldidxy = find(sy==nfold);
fdemean=@(x) bsxfun(@minus,x,mean(x));
for il = 1:length(lambdas)
for ir = 1:6
for ifold = 1:nfold
x = squeeze(X(:,:,ifold));% your input data, should be a matrix
x=fdemean(x);
y = squeeze(Y(:,:,ifold,ir));% your target data, should be a vector

% Regularization parameter
lambda = lambdas(il);% your regularization parameter, a scalar value

% Compute X^T * X
xTx = x.' * x;

% Add regularization term to XTX
xTx_reg = xTx + lambda * eye(size(xTx));

% Compute X^T * y
XTy = x.' * y;

% Solve for w using the equation
beta(:,ifold,ir,il) = xTx_reg \ XTy;
end
end
end
