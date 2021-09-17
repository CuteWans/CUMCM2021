T0=99588; % 初始温度
T1=1e-3;% 终止温度
l=2; % 各温度下的迭代次数
q=0.993;%降温速率
Time = ceil(log(T1 / T0) / log(0.9)); %提前估计迭代次数
point1=[150; 1; 2.5; 1.68; 325];
f = zeros(Time,1);%存储退火过程中的函数值
f0=func(point1);%计算初始值
idc=0;%初始化计数值
 
while T0>T1 && idc <= 100
    idc =idc+1;
    point2=new_point(point1, T0);
    % Metropolis法则判断是否接受新解
    [point1,R] = Metropolis(point1,point2,T0);
    f(idc) = R;           
    end
    T0 = q * T0; 
end

figure
x = 1:idc
plot(x,f(x))
xlabel('迭代次数')
ylabel('函数值')
title('优化过程')
 
disp('最优解:')
f(idc)
