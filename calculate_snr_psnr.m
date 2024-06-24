%该函数用于计算SNR信噪比和PSNR峰值信噪比
%信噪比与峰值信噪比的值越大代表图像的质量越好。
%I表示原始图像
%J表示恢复后的图像
function [snr,psnr] = calculate_snr_psnr(YT,XT)
%如果是I灰度图像只有二维，如果I是彩色图像将会有三维
dim = length(size(YT));%保存的是I的维度
M = size(YT,1);
N = size(YT,2);
dif = (YT-XT).^2;
I_2 = YT.^2;
if dim == 2
    val1 = sum(sum(dif));
    val2 = sum(sum(I_2));
else
    val1 = sum(sum(sum(dif)));
    val2 = sum(sum(sum(I_2)));
end
snr = 10*log10(val2/val1);
psnr = 10*log10((255*255*M*N)/val1);
end
