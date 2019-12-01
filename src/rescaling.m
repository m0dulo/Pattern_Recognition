function[dst] = rescaling(src)

row_nums = size(src, 1);
column_nums = size(src, 2);

dst = zeros(row_nums, column_nums);

for i = 1 : column_nums
    min_num = min(src(:, i));
    max_num = max(src(:, i));
    dst(:, i) = (src(:, i) - min_num) / (max_num - min_num);
end
end