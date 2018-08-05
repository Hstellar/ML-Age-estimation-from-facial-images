clear all
clc

% Load data.
m = matfile('F:\New_Downloads\ML\high_res.mat');
data=m.data;

m1= matfile('F:\New_Downloads\ML\test_res.mat');
data1=m1.data1;

%Set feature value as input
X(:,1)=data(:,4);
X(:,2)=data(:,5);

X1(:,1)=data1(:,4);
X1(:,2)=data1(:,5);
%Set age range as target
Y = data(:,1)>30;
Y1=data1(:,1)>30;

%For split data as test and train data
%[trainInd,valInd,testInd] = dividerand(size(X,1),0.7,0.005,0.3);

%Xtrain=X(trainInd,:);
%Xtest=X(testInd,:);

%Ytrain=Y(trainInd,:);
%Ytest=Y(testInd,:);
Xtrain=X(:,:);
Xtest=X1(:,:);

Ytrain=Y(:,:);
Ytest=Y1(:,:);
%Train svm classifier
svmStruct = svmtrain(Xtrain,Ytrain,'ShowPlot',true);

%Predict age group for test data
Group = svmclassify(svmStruct,Xtest);
disp(Group);

%Analysis of Prediction of age group
comp = (Group==Ytest);
acc = sum(comp)/size(Ytest,1);

acc_cl=sum(~Group&~Ytest)/sum(~Ytest);
disp(acc_cl);
acc_c2=sum(Group&Ytest)/sum(Ytest);
disp(acc_c2);
