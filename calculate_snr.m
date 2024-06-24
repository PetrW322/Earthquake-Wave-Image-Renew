
function snr_value = calculate_snr(YT, XT)
% YT 原图 XT 现图
noise = YT - XT;
signal_power = sum(YT(:).^2) / numel(YT);
noise_power = sum(noise(:).^2) / numel(noise);
snr_value = 10 * log10(signal_power / noise_power);

end