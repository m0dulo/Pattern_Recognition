function[src] = glgm_norm(src)

src(:, :)=src(:, :)/ sum(sum(src(:, :)));

end