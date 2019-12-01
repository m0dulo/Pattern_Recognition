function[energy,contrast,correlation,entropy, variance, deficit] = get_features(src, level)

energy = 0;
entropy = 0;
contrast = 0;
correlation = 0;
variance = 0;
deficit = 0;

sum = 0;
mean = 0;
mean_x = 0;
mean_y = 0;
variance_x = 0;
variance_y = 0;
Exy = 0;

for i = 1 : level
    for j = 1 : level
        energy = energy + src(i, j) * src(i, j);
        if src(i, j) ~= 0
            entropy = entropy - src(i, j) * log(src(i, j));
        end
        contrast = ((i - j)^2) * src(i, j) + contrast;
        mean_x = i * src(i, j) + mean_x;
        mean_y = j * src(i, j) + mean_y;
        sum = sum + src(i, j);
    end
end

mean = sum / (level * level);

for i = 1 : level
    for j = 1: level
        variance_x = (i - mean_x) ^ 2 * src(i, j) + variance_x;
        variance_y = (j - mean_y) ^ 2 * src(i, j) + variance_y;
        Exy = i * j * src(i, j) + Exy;
        deficit = src(i, j) / (1 + (i - j) ^ 2) + deficit;
        variance = variance + ((i - mean) ^ 2) * src(i, j); 
    end
end

correlation = (Exy - mean_x * mean_y) / (variance_x * variance_y);

end