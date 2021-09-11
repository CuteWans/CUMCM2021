function [S,R] = Metropolis(S1,S2,T)
 
% S1：  当前解
% S2:   新解
% D:    距离矩阵（点的函数值）
% T:    当前温度
% S：   下一个当前解
% R：   下一个当前解的函数值

R1 = func(S1);  
R2 = func(S2);  
dC = R2 - R1;   %计算函数值之差
if dC > 0       %如果函数值增加 接受新点
    S = S2;
    R = R2;
elseif exp(dC/T)>= rand   %以exp(-dC/T)概率接受新点
    S = S2;
    R = R2;
else        
    S = S1;
    R = R1;
end

