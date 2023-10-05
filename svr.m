close all;
clc
clear
%% 下载数据
load('p_train.mat');
load('p_test.mat');
load('t_train.mat');
load('t_test.mat');
%% 数据归一化
%输入样本归一化
[pn_train,ps1] = mapminmax(p_train');
pn_train = pn_train';
pn_test = mapminmax('apply',p_test',ps1);
pn_test = pn_test';
%输出样本归一化
[tn_train,ps2] = mapminmax(t_train');
tn_train = tn_train';
tn_test = mapminmax('apply',t_test',ps2);
tn_test = tn_test';
%% SVR模型创建/训练
% 寻找最佳c参数/g参数——交叉验证方法
% SVM模型有两个非常重要的参数C与gamma。
% 其中 C是惩罚系数，即对误差的宽容度。
% c越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差
% gamma是选择RBF函数作为kernel后，该函数自带的一个参数。隐含地决定了数据映射到新的特征空间后的分布，
% gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
[c,g] = meshgrid(-10:0.5:10,-10:0.5:10);
[m,n] = size(c);
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 0;
bestg = 0;
error = Inf;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j) ),' -s 3 -p 0.1'];
        cg(i,j) = libsvmtrain(tn_train,pn_train,cmd);
        if cg(i,j) < error
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
        if abs(cg(i,j) - error) <= eps && bestc > 2^c(i,j)
            error = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end
    end
end
% 创建/训练SVR 
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];
model = libsvmtrain(tn_train,pn_train,cmd);
%保存训练出的模型
save('model.mat', 'model');

%% SVR仿真预测
% [Predict_1,error_1,dec_values_1] = libsvmpredict(tn_train,pn_train,model);
[Predict_2,error_2,dec_values_2] = libsvmpredict(tn_test,pn_test,model);
% 反归一化
% predict_1 = mapminmax('reverse',Predict_1,ps2);
predict_2 = mapminmax('reverse',Predict_2,ps2);
% 将预测数据转为0或1
for i = 1:length(predict_2)
    if abs(predict_2(i)-0) <= abs(predict_2(i)-1)
        predict_2(i) = 0;
    else
        predict_2(i) = 1;
    end
end
%% 计算正确率
[len,~]=size(predict_2);
correct = sum(predict_2 == t_test);
accuracy = correct / len * 100;
disp(['正确率为:', num2str(accuracy), '%'])
figure(1)
plot(1:length(t_test),t_test,'r-*',1:length(t_test),predict_2,'b:o')
grid on
legend('真实值','预测值')
xlabel('样本编号')
ylabel('值')
string_2 = {'测试集预测结果对比'};
title(string_2)


