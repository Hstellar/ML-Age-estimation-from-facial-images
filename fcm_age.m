clear all
close all
clc


% Load data.
m = matfile('F:\New_Downloads\ML\high_res.mat');
data=m.data;

m1= matfile('F:\New_Downloads\ML\test_res.mat');
data1=m1.data1;

%Set feature value as input
X(:,1)=data(:,6);
X(:,2)=data(:,5);

X1(:,1)=data1(:,4);
X1(:,2)=data1(:,5);
%Set age as target
Y=data(:,1);
Y1=data1(:,1);
%For split data as test and train data
% [trainInd,valInd,testInd] = dividerand(size(X,1),0.7,0.005,0.3);
% 
% Xtrain=X(trainInd,:);
% Xtest=X(testInd,:);
% 
% Ytrain=Y(trainInd,:);
% Ytest=Y(testInd,:);
Xtrain=X(:,:);
Xtest=X1(:,:);

Ytrain=Y(:,:);
Ytest=Y1(:,:);
% Find 4 clusters using fuzzy c-means clustering.
[centers,U] = fcm(Xtrain,4);

% Classify each data point into the cluster with the largest membership value.
maxU = max(U);
index1 = find(U(1,:) == maxU);
index2 = find(U(2,:) == maxU);
index3 = find(U(3,:) == maxU);
index4 = find(U(4,:) == maxU);



% Find mean age of each cluster
mean1=sum(Ytrain(index1))/numel(index1);
mean2=sum(Ytrain(index2))/numel(index2);
mean3=sum(Ytrain(index3))/numel(index3);
mean4=sum(Ytrain(index4))/numel(index4);


% Estimate age of Train data
Y2=mean1*U(1,:)+mean2*U(2,:)+mean3*U(3,:)+mean4*U(4,:);
Y2=round(Y2).';

% Analysis prediction for Train data
diff=abs(Ytrain-Y2);
mae=sum(diff)/numel(diff);
pred=sum(diff<8)/numel(diff<8);


% Estimate age of Test data
for i=1:size(Xtest,1)
    total=0;
    for j=1:size(centers,1)
        dis(j,i)=norm(Xtest(i,:)-centers(j,:));
        total=total+dis(j,i);
    end
    for j=1:size(centers,1)
        U2(j,i)=dis(j,i)/total;
    end
end

Y2=mean1*U2(1,:)+mean2*U2(2,:)+mean3*U2(3,:)+mean4*U2(4,:);
Y2=round(Y2).';
disp(Y2);
% Analysis age of Test data
diff=abs(Ytest-Y2);
mae2=sum(diff)/numel(diff);
pred2=sum(diff<8)/numel(diff<8);
disp(mae2);
disp(pred2);
