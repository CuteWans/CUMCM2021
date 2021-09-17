clc
clear
close all
[input_data] = textread('input.txt') ;
[output_data] = textread('output2.txt');
input = input_data;
output = output_data;

c = randperm(74);
A = input, B = output;
for i = 1:4
    A(c(i)) = input(i);
    B(c(i)) = output(i);
end
input = A, output = B;

input_train = input(1:74,:)';
output_train = output(1:74,:)';
input_test = input(1:20,:)';
output_test = output(1:20,:)';

%训练数据归一化
[inputn, inputps] = mapminmax(input_train);
[outputn, outputps] = mapminmax(output_train);
net = newff(inputn,outputn,90);

%参数设置
net.trainParam.epochs=1000; %迭代次数
net.trainParam.lr=0.3; %学习率
net.trainParam.goal=0.0000000001; %收敛目标

%神经网络训练
net = train(net, inputn, outputn);
load('func2', 'net');
save('func2', 'net', 'inputps', 'outputps');
%训练数据归一化
inputn_test = mapminmax('apply', input_test, inputps);
%神经网络测试输出
an = sim(net, inputn_test);
pred_output = mapminmax('reverse',an,outputps);

%可视化处理
figure(1)
plot(pred_output(1,:),'r')
hold on
plot(output_test(1,:),'b.');
legend('模拟值（C4烯烃选择性）','原始值（C4烯烃选择性）')
err = abs(pred_output - output_test);
err_mean = mean(err);
title('原始值与模拟值')
xlabel('样本')
ylabel('C4烯烃选择性')

figure(2)
plot(err,'-*')
err_mean
title('测试误差')
ylabel('平均误差')
xlabel('样本')