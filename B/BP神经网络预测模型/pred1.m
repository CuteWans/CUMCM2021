clc
clear
close all
[num_input]=textread('input.txt') ;
[num_output]=textread('output1.txt');
input = num_input;
output = num_output;

c = randperm(74);
A = input, B = output;
for i = 1:4
    A(c(i)) = input(i);
    B(c(i)) = output(i);
end
input = A, output = B;

data_train_input = input(1:74,:);
data_train_output = output(1:74,:);
data_test_input = input(1:20,:);
data_test_output = output(1:20,:);
input_train = data_train_input';
output_train = data_train_output';
input_test = data_test_input';
output_test = data_test_output';
%训练数据归一化
[inputn,inputps] = mapminmax(input_train);
[outputn,outputps] = mapminmax(output_train);
net = newff(inputn,outputn,90);
%参数设置
net.trainParam.epochs=1000;%迭代次数

net.trainParam.lr=0.3;%学习率
net.trainParam.goal=0.0000000001;%收敛目标
%神经网络训练
net = train(net,inputn,outputn);
load('func1', 'net');
save('func1', 'net', 'inputps', 'outputps');
%训练数据归一化
inputn_test = mapminmax('apply',input_test,inputps);
%神经网络测试输出
an = sim(net,inputn_test);
BPoutput = mapminmax('reverse',an,outputps);

%数据可视化
figure(1)
plot(BPoutput(1,:),'r') %红
hold on
plot(output_test(1,:),'b.');
legend('模拟值（乙醇转化率）','原始值（乙醇转化率）')
err = abs(BPoutput - output_test);
err_mean = mean(err);

figure(2)
plot(err,'-*')
err_mean
title('测试误差')
ylabel('平均误差')
xlabel('样本')