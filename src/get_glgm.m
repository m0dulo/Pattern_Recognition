function [dst] = get_glgm(src, L, W, level, angle)

dst = zeros(level, level);

switch angle
    case 0
        for i = 1: L
            for j = 1 : W - 1
                rows = src(i, j);
                cols = src(i, j + 1);
                dst(rows + 1, cols + 1) = dst(rows + 1, cols + 1) + 1;
            end
        end
    case 45
        for i = 1: L - 1
            for j = 1 : W - 1
                rows = src(i, j);
                cols = src(i + 1, j + 1);
                dst(rows + 1, cols + 1) = dst(rows + 1, cols + 1) + 1;
            end
        end
    case 90
        for i = 1: L - 1
            for j = 1 : W
                rows = src(i, j);
                cols = src(i + 1, j);
                dst(rows + 1, cols + 1) = dst(rows + 1, cols + 1) + 1;
            end
        end
end

end

