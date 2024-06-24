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
eta = 0.1;       % Relative noise level.

fprintf(1,'Creating a 2D seismic travel-time tomography test problem\n');

% Create the test problem.
[A,b_ex,x_ex] = seismictomo(N);

% Noise level.
delta = eta*norm(b_ex);

% Add noise to the rhs.
randn('state',0);
e = randn(size(b_ex));
e = delta*e/norm(e); %生成噪声
%b = b_ex + e;   %有噪声
b = b_ex;        %无噪声

YT = reshape(x_ex,N,N);

%{
% Show the exact solution.
subplot(3,2,5)
imagesc(reshape(x_ex,N,N)), colormap gray,
axis image off
c = caxis;
title('Exact phantom')
%}
c = caxis;

% No. of iterations.
[m,n]=size(A);
k = 1000;
x0=zeros(n,1);
tol=1e-3;

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with Pure RK method.',k);
fprintf(1,'\nThis takes a moment ...');

%% 纯RK，无嵌入，不优化
% 单纯的RK方法
%随机Kaczmarz方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_rk        解向量
%         Err_rk      相对误差
%         iter_rk     总迭代步数
%         t_it_rk     迭代过程运行时间
start_time_rk = cputime;
timer1 = tic;

[m,~]=size(A);
norm2_row=sum(abs(A).^2,2); %储存每行行范数的平方
Af=norm(A,'fro');     %计算矩阵A的Frobenius范数
x_rk =x0;
Err_rk=zeros(k,1);  %为相对误差向量分配储存空间
norm_x_ex=norm(x_ex);
prob=norm2_row/(Af^2);  %定义概率准则
alphabet=1:m;
uk=randsrc(k,1,[alphabet; prob']);  %依概率生成行指标

tic
for iter_rk=1:k
    err_rk=norm(x_rk-x_ex)/norm_x_ex;   %计算相对误差
    Err(iter_rk)=err_rk;
    if (err_rk <= tol)    %判断是否满足停机准则
        break;
    end 
    o =uk(iter_rk);
    x_rk =x_rk +(b(o)-A(o,:)*x_rk)/norm2_row(o)*A(o,:)';  %得到下一步迭代 
end
t_it_rk = toc; %迭代总时间
elapsed_time_rk = toc(timer1); %算法总运行时间
end_time_rk = cputime; 
disp(iter_rk);  %迭代次数
total_time_rk = end_time_rk - start_time_rk; %CPU时间

% Show the RK pure method solution.
subplot(2,2,1)
imagesc(reshape(x_rk,N,N)), colormap gray,
axis image off
caxis(c);
title('Pure RK reconstruction')

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with Gaussian Embedding applied in RK method.',k);
fprintf(1,'\nThis takes a moment ...\n');

XT= reshape(x_rk,N,N);
[snr_p,psnr_p] = calculate_snr_psnr(YT,XT);
%% RK方法带高斯嵌入
% rk方法带高斯嵌入
%随机Kaczmarz方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_rkgaussian        解向量
%         Err_rkgaussian      相对残差向量
%         iter_rkgaussian     总迭代步数
%         t_pre_rkgaussian    QR分解预处理时间
%         t_it_rkgaussian     迭代时间
%         time_rkgaussian     总时间, 即t_pre+t_it
%         Time_rkgaussian     时间向量
% 格式：[x_rkgaussian,Err_rkgaussian,iter_rkgaussian,t_pre_rkgaussian,t_it_rkgaussian,time_rkgaussian,Time_rkgaussian] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_rkgaussian = cputime;
timer2 = tic;
d=10*n;  %sketching 维数
S=randn(d,m)/sqrt(d); %Gaussian嵌入

[m,~]=size(A);
norm2_row=sum(abs(A).^2,2); %储存每行行范数的平方
Af=norm(A,'fro');     %计算矩阵A的Frobenius范数
x_rkgaussian=x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_rkgaussian =toc;  %预处理
P = R;
Aa = A * P;
norm2_row1=sum(abs(Aa).^2,2); %储存每行行范数的平方
y_rkgaussian = triu(pinv(R)) *x_rkgaussian;
Err=zeros(k,1);  %为相对误差向量分配储存空间
norm_x_ex=norm(x_ex);
Af1=norm(Aa,"fro");
prob=norm2_row1/(Af1^2);  %定义概率准则
alphabet=1:m;
uk=randsrc(k,1,[alphabet; prob']);  %依概率生成行指标
tic
for iter_rkgaussian=1:k
    err_rkgaussian=norm(P* y_rkgaussian-x_ex)/norm_x_ex;   %计算相对误差
    Err(iter_rkgaussian)=err_rkgaussian;
    Time(iter_rkgaussian)=toc;
    if (err_rkgaussian <= tol)    %判断是否满足停机准则
        break;
    end 
    %update y
    o =uk(iter_rkgaussian); %获取行指标为o
    y_rkgaussian =y_rkgaussian +(b(o)-(Aa(o,:))*y_rkgaussian)/norm2_row1(o)*Aa(o,:)';  %得到下一步迭代 
end
x_rkgaussian = P * y_rkgaussian; %update x
t_it_rkgaussian = toc; %迭代总时间
time_rkgaussian=t_pre_rkgaussian+t_it_rkgaussian;
elapsed_time_rkgaussian = toc(timer2); %算法总运行时间
end_time_rkgaussian = cputime; 
disp(iter_rkgaussian);  %迭代次数
total_time_rkgaussian = end_time_rkgaussian - start_time_rkgaussian; %CPU时间

% Show the Gaussian Embedding applied in RK method solution.
subplot(2,2,2)
imagesc(reshape(x_rkgaussian,N,N)), colormap gray,
axis image off
caxis(c);
title('RK reconstruction with Gaussian Embedding')

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with CountSketch applied in RK method.',k);
fprintf(1,'\nThis takes a moment ...\n');

XT= reshape(x_rkgaussian,N,N);
[snr_g,psnr_g] = calculate_snr_psnr(YT,XT);
%% RK方法带CountSketch变换
% rk方法带countsktech变换
%随机Kaczmarz方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_rkcountsketch        解向量
%         Err_rkcountsketch      相对残差向量
%         iter_rkcountsketch     总迭代步数
%         t_pre_rkcountsketch    QR分解预处理时间
%         t_it_rkcountsketch     迭代时间
%         time_rkcountsketch     总时间, 即t_pre+t_it
%         Time_rkcountsketch     时间向量
% 格式：[x_rkcountsketch,Err_rkcountsketch,iter_rkcountsketch,t_pre_rkcountsketch,t_it_rkcountsketch,time_rkcountsketch,Time_rkcountsketch] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_rkcountsketch = cputime;
timer3 = tic;
d =round(d);
sgn = 2 * (randi(2, [1, m]) - 1.5);
% A=bsxfun(@times,A,sgn');
% b=bsxfun(@times,b,sgn');
B=randsample(d,m,true);
S=sparse(B,1:m,sgn,d,m); %CountSketch嵌入

[m,~]=size(A);
norm2_row=sum(abs(A).^2,2); %储存每行行范数的平方
Af=norm(A,'fro');     %计算矩阵A的Frobenius范数
x_rkcountsketch=x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_rkcountsketch =toc;
P = R;
%I= eye(2500);
%P = lsqminnorm(R, I);
Aa = A * P;
I = eye(2500);
R1 = lsqminnorm(R, I);
y_rkcountsketch = R1 *x_rkcountsketch;
Err=zeros(k,1);  %为相对误差向量分配储存空间
norm_x_ex=norm(x_ex);
norm2_row1=sum(abs(Aa).^2,2);
Af1=norm(Aa,'fro');
prob=norm2_row1/(Af1^2);  %定义概率准则
alphabet=1:m;
uk=randsrc(k,1,[alphabet; prob']);  %依概率生成行指标


tic
for iter_rkcountsketch=1:k
    err_rkcountsketch=norm(P*y_rkcountsketch-x_ex)/norm_x_ex;   %计算相对误差
    Err(iter_rkcountsketch)=err_rkcountsketch;
    Time(iter_rkcountsketch)=toc;
    if (err_rkcountsketch <= tol)    %判断是否满足停机准则
        break;
    end 
    %update y
    o =uk(iter_rkcountsketch);
    y_rkcountsketch =y_rkcountsketch +(b(o)-Aa(o,:)*y_rkcountsketch)/norm2_row1(o)*Aa(o,:)';  %得到下一步迭代 
end
x_rkcountsketch = P* y_rkcountsketch;%update x
t_it_rkcountsketch = toc; %迭代总时间
time_rkcountsketch=t_pre_rkcountsketch+t_it_rkcountsketch;
elapsed_time_rkcountsketch = toc(timer3); %算法总运行时间
end_time_rkcountsketch = cputime; 
disp(iter_rkcountsketch);  %迭代次数
total_time_rkcountsketch = end_time_rkcountsketch - start_time_rkcountsketch; %CPU时间

% Show the CountSketch applied in RK method solution.
subplot(2,2,3)
imagesc(reshape(x_rkcountsketch,N,N)), colormap gray,
axis image off
caxis(c);
title('RK reconstruction with CountSketch')


fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with SRTT applied in RK method.',k);
fprintf(1,'\nThis takes a moment ...\n');

XT= reshape(x_rkcountsketch,N,N);
[snr_c,psnr_c] = calculate_snr_psnr(YT,XT);

%% RK方法带SRTT变换
% rk方法带SRTT变换
%随机Kaczmarz方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_rksrtt        解向量
%         Err_rksrtt      相对残差向量
%         iter_rksrtt     总迭代步数
%         t_pre_rksrtt    QR分解预处理时间
%         t_it_rksrtt     迭代时间
%         time_rksrtt     总时间, 即t_pre+t_it
%         Time_rksrtt     时间向量
% 格式：[x_rksrtt,Err_rksrtt,iter_rksrtt,t_pre_rksrtt,t_it_rksrtt,time_rksrtt,Time_rksrtt] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_rksrtt = cputime;
timer4 = tic;
rowIndices = 1:d;
colIndices = randi(m, 1, d);
values = ones(1, d);
P = sparse(rowIndices, colIndices, values, d, m);
D=diag((2*randi(2,m,1) - 3));
S=sqrt(m/d)*P*dct(D); %SRTT变换

[m,~]=size(A);
norm2_row=sum(abs(A).^2,2); %储存每行行范数的平方
Af=norm(A,'fro');     %计算矩阵A的Frobenius范数
x_rksrtt =x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_rksrtt =toc;
U = R;
Aa = A * U;
norm2_row1=sum(abs(Aa).^2,2); %储存每行行范数的平方
y_rksrtt = triu(pinv(R)) *x_rksrtt;
Err=zeros(k,1);  %为相对误差向量分配储存空间
norm_x_ex=norm(x_ex);
%prob=norm2_row/(Af^2);  %定义概率准则
Af1=norm(Aa,"fro");
prob=norm2_row1/(Af1^2);  %定义概率准则
alphabet=1:m;
uk=randsrc(k,1,[alphabet; prob']);  %依概率生成行指标
tic
for iter_rksrtt=1:k
    err_rksrtt=norm(U* y_rksrtt-x_ex)/norm_x_ex;   %计算相对误差
    Err(iter_rksrtt)=err_rksrtt;
    Time(iter_rksrtt)=toc;
    if (err_rksrtt <= tol)    %判断是否满足停机准则
        break;
    end 
    %update y 
    o =uk(iter_rksrtt);
    y_rksrtt =y_rksrtt +(b(o)-Aa(o,:)*y_rksrtt)/norm2_row1(o)*Aa(o,:)';  %得到下一步迭代 
end
x_rksrtt = U * y_rksrtt; %update x
t_it_rksrtt = toc; %迭代总时间
time_rksrtt = t_pre_rksrtt +t_it_rksrtt;
elapsed_time_rksrtt = toc(timer4); %算法总运行时间
end_time_rksrtt = cputime; 
disp(iter_rksrtt);  %迭代次数
total_time_rksrtt = end_time_rksrtt - start_time_rksrtt; %CPU时间

% Show the SRTT applied in RK method solution.
subplot(2,2,4)
imagesc(reshape(x_rksrtt,N,N)), colormap gray,
axis image off
caxis(c);
title('RK reconstruction with SRTT')

XT = reshape(x_rksrtt,N,N);
[snr_s,psnr_s] = calculate_snr_psnr(YT,XT);
