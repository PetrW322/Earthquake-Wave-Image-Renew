function mse_value = calculate_mse(YT, XT)
% 均方误差 YT 原图 XT 现图

mse_value = mean((YT(:) - XT(:)).^2);
end