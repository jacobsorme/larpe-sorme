function [segmentation, centers] = kmeans_segm(image, K, L, seed)

I = double(reshape(image, size(image,1) * size(image,2), 3));

% Randomly initialize centers by picking existing pixels.
s = RandStream('mt19937ar', 'Seed', seed);
centers = datasample(s, unique(I, 'rows'), K, 1, 'Replace', false);

for l = 1:L
    % Get distance from every center to every pixel.
    D = pdist2(centers, I, 'squaredeuclidean');
    [~, segmentation] = min(D, [], 1);
    
    centers_old = centers;
    
    % Update center positions (their means).
    for k = 1:K
        ind = segmentation == k;
        centers(k, :) = mean(I(ind(:)', :));
    end
    
    % Check for convergence.
    if all(centers_old - centers == 0)
        it_to_convergence = l
        break
    end
end

% Reshape into 2D image.
segmentation = uint8(reshape(segmentation, size(image,1), size(image,2), 1));

end