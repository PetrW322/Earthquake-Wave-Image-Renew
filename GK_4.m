%%
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
% b = b_ex;      %无噪声

YT= reshape(x_ex,N,N);
%{
% Show the exact solution.
subplot(3,2,1)
imagesc(reshape(x_ex,N,N)), colormap gray,
axis image off
c = caxis;
title('Exact phantom')
%}

c = caxis;

% No. of iterations.
[m,n]=size(A);
k = 100;  %可设置迭代次数
x0=zeros(n,1);
tol=1e-3;  %可设置误差准则

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with Pure GK method.',k);
fprintf(1,'\nThis takes a moment ...\n');


%% 纯贪婪Kaczmarz，无嵌入
% 单纯的GK方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_gk        解向量
%         Err_gk      相对误差
%         iter_gk     总迭代步数
%         t_it_gk     迭代过程运行时间
start_time_gk = cputime;
timer1 = tic;

norm2_row=sum(abs(A).^2,2);  %储存矩阵A的行范数
Err=zeros(k,1);
x_gk=x0;
nxstar=norm(x_ex);

tic
for iter_gk = 1:k
    err_gk=norm(x_gk-x_ex)/nxstar;   %计算相对误差
    Err(iter_gk)=err_gk;
    if (err_gk <= tol)     %判断是否满足停机准则
        break;
    end
    re=b-A*x_gk;
    e1=(re).^2./norm2_row;
    [~,o]= max(e1);
    x_gk=x_gk+(re(o))/norm2_row(o)*A(o,:)';
end

t_it_gk = toc; %迭代总时间
elapsed_time_gk = toc(timer1); %算法总时间
end_time_gk = cputime;
disp(iter_gk); %迭代次数
total_time_gk = end_time_gk - start_time_gk; %CPU时间


% Show the greedy Kaczmarz solution.
subplot(2,2,1)
imagesc(reshape(x_gk,N,N)), colormap gray,
axis image off
caxis(c);
title('Pure GK reconstruction')
XT = reshape(x_gk,N,N);

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with Gaussian Embedding applied in GK method.',k);
fprintf(1,'\nThis takes a moment ...\n');

[snr_p,psnr_p] = calculate_snr_psnr(YT,XT);

%% 贪婪Kaczmarz方法带高斯嵌入
% gk方法带高斯嵌入
%贪婪Kaczmarz方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_gkgaussian        解向量
%         Err_gkgaussian      相对残差向量
%         iter_gkgaussian     总迭代步数
%         t_pre_gkgaussian    QR分解预处理时间
%         t_it_gkgaussian     迭代时间
%         time_gkgaussian     总时间, 即t_pre+t_it
%         Time_gkgaussian     时间向量
% 格式：[x_gkgaussian,Err_gkgaussian,iter_gkgaussian,t_pre_gkgaussian,t_it_gkgaussian,time_gkgaussian,Time_gkgaussian] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_gkgaussian = cputime;
timer2 = tic;
d=10*n;  %sketching 维数
S=randn(d,m)/sqrt(d); %Gaussian嵌入


norm2_row=sum(abs(A).^2,2); %储存每行行范数的平方
x_gkgaussian=x0;
tic
A_=S*A;
[~,R] = qr(A_,0);
t_pre_gkgaussian =toc;  %预处理
P =  R;
Aa = A * P;
norm2_row1=sum(abs(Aa).^2,2); %储存每行行范数的平方
y_gkgaussian = triu(pinv(R))*x_gkgaussian;
Err=zeros(k,1);  %为相对误差向量分配储存空间
norm_x_ex=norm(x_ex);

tic
for iter_gkgaussian = 1:k
    err_gkgaussian=norm(P* y_gkgaussian-x_ex)/norm_x_ex;   %计算相对误差
    Err(iter_gkgaussian)=err_gkgaussian;
    Time(iter_gkgaussian)=toc;
    if (err_gkgaussian <= tol)     %判断是否满足停机准则
        break;
    end
    %update y
    re=b-Aa*y_gkgaussian;
    e1=(re).^2./norm2_row1;
    [~,o]= max(e1);
    y_gkgaussian=y_gkgaussian+(re(o))/norm2_row1(o)*Aa(o,:)';
end
x_gkgaussian = P * y_gkgaussian; %update x
t_it_gkgaussian = toc; %迭代总时间
time_gkgaussian=t_pre_gkgaussian+t_it_gkgaussian;
elapsed_time_gkgaussian = toc(timer2); %算法总运行时间
end_time_gkgaussian = cputime; 
disp(iter_gkgaussian);  %迭代次数
total_time_gkgaussian = end_time_gkgaussian - start_time_gkgaussian; %CPU时间


% Show the greedy Kaczmarz with Gaussian embedding solution.
subplot(2,2,2)
imagesc(reshape(x_gkgaussian,N,N)), colormap gray,
axis image off
caxis(c);
title('GK reconstruction with Gaussian Embedding')
XT = reshape(x_gkgaussian,N,N);

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with CountSketch applied in GK method.',k);
fprintf(1,'\nThis takes a moment ...\n');

[snr_g,psnr_g] = calculate_snr_psnr(YT,XT);


%% GK方法带CountSketch变换
% gk方法带countsktech变换
%贪婪Kaczmarz方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_gkcountsketch        解向量
%         Err_gkcountsketch      相对残差向量
%         iter_gkcountsketch     总迭代步数
%         t_pre_gkcountsketch    QR分解预处理时间
%         t_it_gkcountsketch     迭代时间
%         time_gkcountsketch     总时间, 即t_pre+t_it
%         Time_gkcountsketch     时间向量
% 格式：[x_gkcountsketch,Err_gkcountsketch,iter_gkcountsketch,t_pre_gkcountsketch,t_it_gkcountsketch,time_gkcountsketch,Time_gkcountsketch] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_gkcountsketch = cputime;
timer3 = tic;
d =round(d);
sgn = 2 * (randi(2, [1, m]) - 1.5);
% A=bsxfun(@times,A,sgn');
% b=bsxfun(@times,b,sgn');
B=randsample(d,m,true);
S=sparse(B,1:m,sgn,d,m); %CountSketch嵌入

norm2_row=sum(abs(A).^2,2); %储存每行行范数的平方
x_gkcountsketch=x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_gkcountsketch =toc;  %预处理
P = R;
Aa = A * P;
I = eye(2500);
R1 = lsqminnorm(R, I);
y_gkcountsketch = R1 *x_gkcountsketch;
Err=zeros(k,1);  %为相对误差向量分配储存空间
norm_x_ex=norm(x_ex);
norm2_row1=sum(abs(Aa).^2,2); %储存每行行范数的平方

tic
for iter_gkcountsketch = 1:k
    err_gkcountsketch=norm(P* y_gkcountsketch-x_ex)/norm_x_ex;   %计算相对误差
    Err(iter_gkcountsketch)=err_gkcountsketch;
    Time(iter_gkcountsketch)=toc;
    if (err_gkcountsketch <= tol)     %判断是否满足停机准则
        break;
    end
    %update y
    re=b-Aa*y_gkcountsketch;
    e1=(re).^2./norm2_row1;
    [~,o]= max(e1);
    y_gkcountsketch = y_gkcountsketch + (re(o))/norm2_row1(o)*Aa(o,:)';   %得到下一步迭代 
end

x_gkcountsketch = P* y_gkcountsketch;   %update x
t_it_gkcountsketch = toc; %迭代总时间
time_gkcountsketch=t_pre_gkcountsketch+t_it_gkcountsketch;
elapsed_time_rkcountsketch = toc(timer3); %算法总运行时间
end_time_gkcountsketch = cputime; 
disp(iter_gkcountsketch);  %迭代次数
total_time_gkcountsketch = end_time_gkcountsketch - start_time_gkcountsketch; %CPU时间

% Show the CountSketch applied in GK method solution.
subplot(2,2,3)
imagesc(reshape(x_gkcountsketch,N,N)), colormap gray,
axis image off
caxis(c);
title('GK reconstruction with CountSketch')
XT = reshape(x_gkcountsketch,N,N);

fprintf(1,'\n\n');
fprintf(1,'Perform k = %2.0f iterations with SRTT applied in GK method.',k);
fprintf(1,'\nThis takes a moment ...\n');

[snr_c,psnr_c] = calculate_snr_psnr(YT,XT);


%% GK方法带SRTT变换
% gk方法带SRTT变换
%贪婪Kaczmarz方法
% input   A        系数矩阵
%         x0       初始估计向量
%         x_ex     方程组的最小范数解
%         b        右端向量
%         k        最大迭代步数
%         tol      精度
%
% output  x_gksrtt        解向量
%         Err_gksrtt      相对残差向量
%         iter_gksrtt     总迭代步数
%         t_pre_gksrtt    QR分解预处理时间
%         t_it_gksrtt     迭代时间
%         time_gksrtt     总时间, 即t_pre+t_it
%         Time_fksrtt     时间向量
% 格式：[x_gksrtt,Err_gksrtt,iter_gksrtt,t_pre_gksrtt,t_it_gksrtt,time_gksrtt,Time_gksrtt] = cgls_sqr(A,S,x0,b,k,tol);
% 预处理
start_time_gksrtt = cputime;
timer4 = tic;
rowIndices = 1:d;
colIndices = randi(m, 1, d);
values = ones(1, d);
P = sparse(rowIndices, colIndices, values, d, m);
D=diag((2*randi(2,m,1) - 3));
S=sqrt(m/d)*P*dct(D); %SRTT变换


norm2_row=sum(abs(A).^2,2); %储存每行行范数的平方
x_gksrtt =x0;
tic
A_=S*A;
[~,R]=qr(A_,0);
t_pre_gksrtt =toc;
U = R;
Aa = A * U;
norm2_row1=sum(abs(Aa).^2,2); %储存每行行范数的平方
y_gksrtt = triu(pinv(R)) *x_gksrtt;
Err=zeros(k,1);  %为相对误差向量分配储存空间
norm_x_ex=norm(x_ex);

tic
for iter_gksrtt=1:k
    err_gksrtt=norm(U* y_gksrtt-x_ex)/norm_x_ex;   %计算相对误差
    Err(iter_gksrtt)=err_gksrtt;
    Time(iter_gksrtt)=toc;
    if (err_gksrtt <= tol)    %判断是否满足停机准则
        break;
    end 
    %update y 
    re=b-Aa*y_gksrtt;
    e1=(re).^2./norm2_row1;
    [~,o]= max(e1);
    y_gksrtt=y_gksrtt+(re(o))/norm2_row1(o)*Aa(o,:)';
end
x_gksrtt = U * y_gksrtt; %update x
t_it_gksrtt = toc; %迭代总时间
time_gksrtt = t_pre_gksrtt +t_it_gksrtt;
elapsed_time_gksrtt = toc(timer4); %算法总运行时间
end_time_gksrtt = cputime; 
disp(iter_gksrtt);  %迭代次数
total_time_gksrtt = end_time_gksrtt - start_time_gksrtt; %CPU时间

% Show the SRTT applied in GK method solution.
subplot(2,2,4)
imagesc(reshape(x_gksrtt,N,N)), colormap gray,
axis image off
caxis(c);
title('RK reconstruction with SRTT')
XT = reshape(x_gksrtt,N,N);

[snr_s,psnr_s] = calculate_snr_psnr(YT,XT);

