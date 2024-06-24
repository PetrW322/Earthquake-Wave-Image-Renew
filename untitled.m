% parallel-beam test problem
% From ARTdemo (script) Demonstrates the use of the ART methods.
%
% This script illustrates the use of the randomized Surrounding method,
% restarted determinstic surrounding method, and restarted randomized surrounding method.
%
% The script creates a 2D seismic travel-time tomographytest problem, adds noise, and solves
% the problems with the above three methods.  The exact solution and the results
% from the methods are shown.
%

close all
fprintf(1,'\nStarting ARTdemo:\n\n');

% Set the parameters for the test problem.
N = 50;           % The discretization points.
eta = 0.01;       % Relative noise level.

fprintf(1,'Creating a 2D seismic travel-time tomography test problem\n');

% Create the test problem.
[A,b_ex,x_ex] = seismictomo(N);

% Noise level.
delta = eta*norm(b_ex);

% Add noise to the rhs.
randn('state',0);
e = randn(size(b_ex));
e = delta*e/norm(e); %生成噪声
b = b_ex + e;   %有噪声
% b = b_ex;        %无噪声

YT = reshape(x_ex,N,N);
c = caxis;

%{
% Show the exact solution.
subplot(3,2,1)
imagesc(reshape(x_ex,N,N)), colormap gray,
axis image off
c = caxis;
title('Exact phantom')
%}

% No. of iterations.
[m,n]=size(A);
k = 100;
x0=zeros(n,1);
tol=1e-3;

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with Pure CGLS method.',k);
fprintf(1,'\nThis takes a moment ...');



%% 纯CGLS，无嵌入，不优化
% 单纯的cgls方法
% input   A        系数矩阵
%         x0       初始估计向量
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_cgls        解向量
%         Res_cgls      相对残差向量
%         iter_cgls     总迭代步数
%         t_it_cgls     迭代总时间
%         ATrk_cgls     储存A^T r_k 范数的向量
%         Time_cgls     时间向量
start_time_cgls = cputime;
timer1 = tic;

Res_cgls=zeros(1,k);
Time_cgls=zeros(1,k);
ATrk_cgls=zeros(1,k);
x_cgls=x0;
rk_cgls=b-A*x_cgls;
beta_cgls=norm(rk_cgls);
AT=A';
pk_cgls=AT*rk_cgls;
Ark_cgls=pk_cgls;
nmz_cgls=norm(Ark_cgls);
tic
for iter_cgls = 1:k 
    res_cgls=norm(rk_cgls)/beta_cgls;
    Res(iter_cgls)=res_cgls;
    Time(iter_cgls)=toc;
    ATrk(iter_cgls) = norm(Ark_cgls)/nmz_cgls;
    if (res_cgls <= tol)    %判断是否满足停机准则
        break;
    end
    Ark1_cgls=AT*rk_cgls;
    Apk_cgls=A*pk_cgls;
    nArk1_cgls=Ark1_cgls'*Ark1_cgls;
    alpha_cgls=nArk1_cgls/(Apk_cgls'*Apk_cgls);
    x_cgls=x_cgls+alpha_cgls*pk_cgls;
    rk_cgls=rk_cgls-alpha_cgls*Apk_cgls;
    Ark_cgls=AT*rk_cgls;
    betak_cgls=(Ark_cgls'*Ark_cgls)/nArk1_cgls;
    pk_cgls=Ark_cgls+betak_cgls*pk_cgls;
end
t_it_cgls = toc; %迭代总时间
elapsed_time_cgls = toc(timer1); %算法总运行时间
end_time_cgls = cputime; 
disp(iter_cgls);  %迭代次数
total_time_cgls = end_time_cgls - start_time_cgls; %CPU时间


% Show the CGLS pure method solution.
subplot(2,2,1)
imagesc(reshape(x_cgls,N,N)), colormap gray,
axis image off
caxis(c);
title('Pure CGLS reconstruction')

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with Gaussian Embedding applied in CGLS method.',k);
fprintf(1,'\nThis takes a moment ...\n');

XT = reshape(x_cgls,N,N);
[snr_p,psnr_p] = calculate_snr_psnr(YT,XT);
snr_value_p = calculate_snr(YT, XT);
mse_value_p = calculate_mse(YT, XT);

%% CGLS方法带高斯嵌入
% cgls方法带高斯嵌入
% input   A        系数矩阵
%         S        Sketch矩阵
%         x0       初始估计向量
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_cggaussian        解向量
%         Res_cggaussian      相对残差向量
%         iter_cggaussian     总迭代步数
%         t_pre_cggaussian    QR分解预处理时间
%         t_it_cggaussian     迭代时间
%         time_cggaussian     总时间, 即t_pre+t_it
%         ATrk_cggaussian     A^T r_k 的范数
%         Time_cggaussian     时间向量
% 格式：[x_cggaussian,Res_cggaussian,iter_cggausian,t_pre_cggaussian,t_it_cggaussian,time_cggaussian,ATrk_cggaussian,Time_cggaussian] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_cggaussian = cputime;
timer2 = tic;
d=10*n;  %sketching 维数
S=randn(d,m)/sqrt(d); %Gaussian嵌入

Res_cggaussian=zeros(1,k);
Time_cggaussian=zeros(1,k);
ATrk_cggaussian=zeros(1,k);
x_cggaussian=x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_cggaussian=toc;
r_cggaussian = b - A*x_cggaussian; 
z_cggaussian = A'*r_cggaussian;
nmz_cggaussian=norm(z_cggaussian);
%s_cggaussian = R\((R')\z_cggaussian);
E1 = lsqminnorm(R', z_cggaussian);
s_cggaussian = lsqminnorm(R, E1);
p_cggaussian = s_cggaussian; 
beta_cggaussian = norm(r_cggaussian);
tic
for iter_cggaussian = 1:k
    res_cggaussian = norm(r_cggaussian)/beta_cggaussian;
    Res(iter_cggaussian) = res_cggaussian;
    ATrk(iter_cggaussian) = norm(z_cggaussian)/nmz_cggaussian;
    Time(iter_cggaussian)=toc;
    if (res_cggaussian <= tol)    %判断是否满足停机准则
        break;
    end
    % update x
    w_cggaussian = A*p_cggaussian;
    alpha_cggaussian = dot(z_cggaussian,s_cggaussian)/(norm(w_cggaussian)^2);
    x_cggaussian = x_cggaussian + alpha_cggaussian * p_cggaussian;
    r_cggaussian = r_cggaussian - alpha_cggaussian * w_cggaussian;
    z1_cggaussian = A'*r_cggaussian;
    %s1_cggaussian = R\((R')\z1_cggaussian);
    F1 = lsqminnorm(R', z1_cggaussian);
    s1_cggaussian = lsqminnorm(R, F1);
    betak_cggaussian = dot(z1_cggaussian,s1_cggaussian)/dot(z_cggaussian,s_cggaussian);
    z_cggaussian = z1_cggaussian; 
    s_cggaussian = s1_cggaussian;
    p_cggaussian = s_cggaussian + betak_cggaussian*p_cggaussian;
end
t_it_cggaussian=toc;
time_cggaussian=t_pre_cggaussian+t_it_cggaussian;
elapsed_time_cggaussian = toc(timer2); %算法总运行时间
end_time_cggaussian = cputime; 
disp(iter_cggaussian);  %迭代次数
total_time_cggaussian = end_time_cggaussian - start_time_cggaussian; %CPU时间

% Show the Gaussing Embedding solution.
subplot(2,2,2)
imagesc(reshape(x_cggaussian,N,N)), colormap gray,
axis image off
caxis(c);
title('CGLS Restruction with Gaussian Embedding')

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with CountSketch applied in CGLS method.',k);
fprintf(1,'\nThis takes a moment ...\n');

XT = reshape(x_cggaussian,N,N);
[snr_g,psnr_g] = calculate_snr_psnr(YT,XT);
snr_value_g = calculate_snr(YT, XT);
mse_value_g = calculate_mse(YT, XT);

%% CGLS方法带CountSketch变换
% cgls方法带CountSketch变换
% input   A        系数矩阵
%         S        Sketch矩阵
%         x0       初始估计向量
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_cgcountsketch        解向量
%         Res_cgcountsketch      相对残差向量
%         iter_cgcountsketch     总迭代步数
%         t_pre_cgcountsketch    QR分解预处理时间
%         t_it_cgcountsketch     迭代时间
%         time_cgcountsketch     总时间, 即t_pre+t_it
%         ATrk_cgcountsketch     A^T r_k 的范数
%         Time_cgcountsketch     时间向量
% 格式：[x_cgcountsketch,Res_cgcountsketch,iter_cgcountsketch,t_pre_cgcountsketch,t_it_cgcountsketch,time_cgcountsketch,ATrk_cgcountsketch,Time_cgcountsketch] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_cgcountsketch = cputime;
timer3 = tic;
d =round(d);
sgn = 2 * (randi(2, [1, m]) - 1.5);
% A=bsxfun(@times,A,sgn');
% b=bsxfun(@times,b,sgn');
B=randsample(d,m,true);
S=sparse(B,1:m,sgn,d,m); %CountSketch嵌入

Res_cgcountsketch=zeros(1,k);
Time_cgcountsketch=zeros(1,k);
ATrk_cgcountsketch=zeros(1,k);
x_cgcountsketch=x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_cgcountsketch=toc;
r_cgcountsketch = b - A*x_cgcountsketch; 
z_cgcountsketch = A'*r_cgcountsketch;
nmz_cgcountsketch=norm(z_cgcountsketch);
%s_cgcountsketch = R\((R')\z_cgcountsketch);
E1 = lsqminnorm(R', z_cgcountsketch);
s_cgcountsketch = lsqminnorm(R, E1);
p_cgcountsketch = s_cgcountsketch; 
beta_cgcountsketch = norm(r_cgcountsketch);
tic
for iter_cgcountsketch = 1:k
    res_cgcountsketch = norm(r_cgcountsketch)/beta_cgcountsketch;
    Res(iter_cgcountsketch) = res_cgcountsketch;
    ATrk(iter_cgcountsketch) = norm(z_cgcountsketch)/nmz_cgcountsketch;
    Time(iter_cgcountsketch)=toc;
    if (res_cgcountsketch <= tol)    %判断是否满足停机准则
        break;
    end
    % update x
    w_cgcountsketch = A*p_cgcountsketch;
    alpha_cgcountsketch = dot(z_cgcountsketch,s_cgcountsketch)/(norm(w_cgcountsketch)^2);
    x_cgcountsketch = x_cgcountsketch + alpha_cgcountsketch * p_cgcountsketch;
    r_cgcountsketch = r_cgcountsketch - alpha_cgcountsketch * w_cgcountsketch;
    z1_cgcountsketch = A'*r_cgcountsketch;
    %s1_cgcountsketch = R\((R')\z1_cgcountsketch);
    F1 = lsqminnorm(R', z1_cgcountsketch);
    s1_cgcountsketch = lsqminnorm(R, F1);
    betak_cgcountsketch = dot(z1_cgcountsketch,s1_cgcountsketch)/dot(z_cgcountsketch,s_cgcountsketch);
    z_cgcountsketch = z1_cgcountsketch; 
    s_cgcountsketch = s1_cgcountsketch;
    p_cgcountsketch = s_cgcountsketch + betak_cgcountsketch*p_cgcountsketch;
end
t_it_cgcountsketch=toc;
time_cgcountsketch=t_pre_cgcountsketch+t_it_cgcountsketch;
elapsed_time_cgcountsketch = toc(timer3); %算法总运行时间
end_time_cgcountsketch = cputime; 
disp(iter_cgcountsketch);  %迭代次数
total_time_cgcountsketch = end_time_cgcountsketch - start_time_cgcountsketch; %CPU时间

% Show the CountSketch Transformation applied in CGLS solution.
subplot(2,2,3)
imagesc(reshape(x_cgcountsketch,N,N)), colormap gray,
axis image off
caxis(c);
title('CGLS Restruction with CountSketch')

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with SRTT applied in CGLS method.',k);
fprintf(1,'\nThis takes a moment ...\n');

XT = reshape(x_cgcountsketch,N,N);
[snr_c,psnr_c] = calculate_snr_psnr(YT,XT);
snr_value_c = calculate_snr(YT, XT);
mse_value_c = calculate_mse(YT, XT);

%% CGLS方法带SRTT变换
% cgls方法带SRTT变换
% input   A        系数矩阵
%         S        Sketch矩阵
%         x0       初始估计向量
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_cgsrtt        解向量
%         Res_cgsrtt      相对残差向量
%         iter_cgsrtt     总迭代步数
%         t_pre_cgsrtt    QR分解预处理时间
%         t_it_cgsrtt     迭代时间
%         time_cgsrtt     总时间, 即t_pre+t_it
%         ATrk_cgsrtt     A^T r_k 的范数
%         Time_cgsrtt     时间向量
% 格式：[x_cgsrtt,Res_cgsrtt,iter_cgsrtt,t_pre_cgsrtt,t_it_cgsrtt,time_cgsrtt,ATrk_cgsrtt,Time_cgsrtt] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_cgsrtt = cputime;
timer4 = tic;
rowIndices = 1:d;
colIndices = randi(m, 1, d);
values = ones(1, d);
P = sparse(rowIndices, colIndices, values, d, m);
D=diag((2*randi(2,m,1) - 3));
S=sqrt(m/d)*P*dct(D); %SRTT变换

Res_cgsrtt=zeros(1,k);
Time_cgsrtt=zeros(1,k);
ATrk_cgsrtt=zeros(1,k);
x_cgsrtt=x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_cgsrtt=toc;
r_cgsrtt = b - A*x_cgsrtt; 
z_cgsrtt = A'*r_cgsrtt;
nmz_cgsrtt=norm(z_cgsrtt);
%s_cgcountsketch = R\((R')\z_cgcountsketch);
E1 = lsqminnorm(R', z_cgsrtt);
s_cgsrtt = lsqminnorm(R, E1);
p_cgsrtt = s_cgsrtt; 
beta_cgsrtt = norm(r_cgsrtt);
tic
for iter_cgsrtt = 1:k
    res_cgsrtt = norm(r_cgsrtt)/beta_cgsrtt;
    Res(iter_cgsrtt) = res_cgsrtt;
    ATrk(iter_cgsrtt) = norm(z_cgsrtt)/nmz_cgsrtt;
    Time(iter_cgsrtt)=toc;
    if (res_cgsrtt <= tol)    %判断是否满足停机准则
        break;
    end
    % update x
    w_cgsrtt = A*p_cgsrtt;
    alpha_cgsrtt = dot(z_cgsrtt,s_cgsrtt)/(norm(w_cgsrtt)^2);
    x_cgsrtt = x_cgsrtt + alpha_cgsrtt * p_cgsrtt;
    r_cgsrtt = r_cgsrtt - alpha_cgsrtt * w_cgsrtt;
    z1_cgsrtt = A'*r_cgsrtt;
    %s1_cgsrtt = R\((R')\z1_cgsrtt);
    F1 = lsqminnorm(R', z1_cgsrtt);
    s1_cgsrtt = lsqminnorm(R, F1);
    betak_cgsrtt = dot(z1_cgsrtt,s1_cgsrtt)/dot(z_cgsrtt,s_cgsrtt);
    z_cgsrtt = z1_cgsrtt; 
    s_cgsrtt = s1_cgsrtt;
    p_cgsrtt = s_cgsrtt + betak_cgsrtt*p_cgsrtt;
end
t_it_cgsrtt=toc;
time_cgsrtt=t_pre_cgsrtt+t_it_cgsrtt;
elapsed_time_cgsrtt = toc(timer4); %算法总运行时间
end_time_cgsrtt = cputime; 
disp(iter_cgsrtt);  %迭代次数
total_time_cgsrtt = end_time_cgsrtt - start_time_cgsrtt; %CPU时间

% Show the SRTT Transformation applied in CGLS solution.
subplot(2,2,4)
imagesc(reshape(x_cgsrtt,N,N)), colormap gray,
axis image off
caxis(c);
title('CGLS Restruction with SRTT')

fprintf(1,'\n\n');
fprintf(1,'\nALL DONE!\n');
fprintf(1,'\nHAPPY NEW YEAR!\n');

XT = reshape(x_cgsrtt,N,N);
[snr_s,psnr_s] = calculate_snr_psnr(YT,XT);
snr_value_s = calculate_snr(YT, XT);
mse_value_s = calculate_mse(YT, XT);

