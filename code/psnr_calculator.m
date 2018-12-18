function [psnr] = psnr_calculator(image_original, image_decoded)
img_orig = imread(image_original);
img_new = imread(image_decoded);
img_orig = double(img_orig);
img_new = double(img_new);

mse = 0;
[row, col] = size(img_orig);
for i = 1:row
    for j = 1:col
        error = abs((img_orig(i,j) - img_new(i,j))^2);
        mse = mse + error;
    end
end
mse = mse / numel(img_orig);

psnr = 10 * log10((255^2)/mse);
end

