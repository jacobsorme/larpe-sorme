function prob = mixture_prob(img, K, L, mask)
%Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).
% Store all pixels for which mask=1 in a Nx3 matrix
% Randomly initialize the K components using masked pixels
% Iterate L times
%
%Expectation: Compute probabilities P_ik using masked pixels
%
%Maximization: Update weights, means and covariances using masked pixels
% Compute probabilities p(c_i) in Eq.(3) for all pixels I.

% Sample the masked pixels.
mask = reshape(mask,size(mask,1)*size(mask,2),1);
img_flat = reshape(img,size(img,1)*size(img,2),3);
pixels = single(img_flat(logical(mask),:));
n_pixels = size(pixels, 1);

% Initialize mu_k and w using K-means.
[segments, mu] = kmeans_segm(reshape(pixels, 1, n_pixels,3), K, 100, 2345);
w = histcounts(segments, 1:K+1) / n_pixels;

% Estimate Sigma from the segmentation.
S = cell(K,1);
for k = 1:K
    S{k} = cov(pixels(segments'==k,:));
end

G = zeros(n_pixels, K);

for l = 1:L
    %%%% Step 1: Expectation %%%%
    % Calculate g_k for all c_i.
    for k = 1:K
        G(:, k) = mvnpdf(pixels, mu(k), S{k});
        %c = pixels - mu(k);
        %G(:, k) = (1/sqrt((2*pi)^3 * norm(S{k}))) * exp(-.5*(c' * inv(S{k}) * c));
    end
    % Probability that every row (pixel) belongs to k.
    P = w .* G;
    % Normalize each row so p_i sums up to 1.
    s = sum(P, 2);
    P = P ./ (s + (s == 0));

    %%%% Step 2: Maximization %%%%
    pixel_sums = sum(P, 1);
    w = (1 / n_pixels) * pixel_sums;
    mu = (P' * pixels) ./ (pixel_sums' + (pixel_sums'==0));
    for k = 1:K
        c = pixels - mu(k);
        S{k} = cov(c);
    end
    
end

% Calculate G and P for every pixel in image, using our mixture model.
G = zeros(size(img_flat,1), K);
for k = 1:K
    G(:, k) = mvnpdf(single(img_flat), mu(k), S{k});
end
prob = reshape(G * w', [size(img,1), size(img,2)]);

end