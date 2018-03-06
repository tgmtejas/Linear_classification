close all;
clear all;
addpath export_fig

% Choose which dataset to use (choices wine, wallpaper, taiji) :
 dataset = 'wine';
 [train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);
 
 

%% Least Square Classifier for wine dataset

%Training a model 

W0 = ones(size(train_labels));
X_train= [W0 train_featureVector];
[W_Matrix, numClasses, categories_Class] = TrainingModel(X_train, train_labels);


%Predicting new training labels for wine data set
[train_pred] = Prediction (W_Matrix, X_train, numClasses, categories_Class); 


%Predicting new testing labels for wine data set
W0 = ones(size(test_labels));
X_test= [W0 test_featureVector];
[test_pred] = Prediction (W_Matrix, X_test, numClasses, categories_Class); 


%Confusion Matrix for Training and Testing
Leastsqr_train_ConfMat = confusionmat(train_labels,categorical(train_pred));
Leastsqr_test_ConfMat = confusionmat(test_labels,categorical(test_pred));

% Classification matrix for Training and Testing(rows should sum to 1)

Leastsqr_train_ClassMat = Leastsqr_train_ConfMat./(meshgrid(countcats(train_labels))');
Leastsqr_test_ClassMat = Leastsqr_test_ConfMat./(meshgrid(countcats(test_labels))');


% mean group accuracy and std for training
Leastsqr_train_accuracy = mean(diag(Leastsqr_train_ClassMat));
Leastsqr_train_stdDev = std(diag(Leastsqr_train_ClassMat));


% mean group accuracy and std for testing
Leastsqr_test_accuracy = mean(diag(Leastsqr_test_ClassMat));
Leastsqr_test_stdDev = std(diag(Leastsqr_test_ClassMat));

%Display
display(Leastsqr_train_accuracy)
display(Leastsqr_test_accuracy)
display(Leastsqr_train_stdDev)
display(Leastsqr_test_stdDev)

%% Plotting
% categories_Class = categories(train_labels);
% numClasses= length(categories_Class);

tf2 = strcmpi(dataset, 'wine');

if (tf2 == 1)
    W0 = ones(length(train_labels),1);
    featureA = 1;
    featureB = 7;
    feature_idx = [featureA,featureB];

     X_train_MLD = train_featureVector(:,feature_idx);
     X_test_MLD = test_featureVector(:,feature_idx);


     X_MdlLinear= [W0 X_train_MLD];
     %MdlLinear = fitcdiscr(X_MdlLinear,train_labels);
     [W_Mld, numClasses, categories_Class] = TrainingModel(X_MdlLinear, train_labels);
   
     % Display the linear discriminants and a set of features in two of the feature dimensions
     figure(1)
     visualizeBoundaries(W_Mld,X_test_MLD,test_labels,1,2)
     title('{\bf Linear Discriminant Classification}')
     export_fig linear_discriminant_wine -png -transparent
end






 %% Fisher Discriminator

categories_Class = categories(train_labels);
%numClasses= length(categories_Class);

[W_fisher, Sw, Sb] = fisherDiscriminator_X(train_featureVector, train_labels, test_featureVector, test_labels, categories_Class);
%W_fisher_Wine = abs(W_fisher_Wine);

%Getting new dimention features
test_fisher = test_featureVector * W_fisher;
train_fisher = train_featureVector * W_fisher;

% KNN Classification
%
[Y_knn_train, Y_train] = KNNClaassifier_X(train_fisher,train_fisher, 9, train_labels, categories_Class );
[Y_knn_test, Y_test] = KNNClaassifier_X(test_fisher,train_fisher, 9, train_labels, categories_Class );


% Create confusion matrix for Fisher_KNN_Wine
%test_labels = double(test_labels);
fisher_train_ConfMat = confusionmat(train_labels,Y_knn_train);
fisher_test_ConfMat = confusionmat(test_labels,Y_knn_test);
[A,B] = size(fisher_train_ConfMat);
tf = strcmpi(dataset, 'taiji');
if (tf == 0)

  fisher_train_ConfMat = fisher_train_ConfMat(1:(A-1), 1:(B-1));

end

[A,B] = size(fisher_test_ConfMat);
tf = strcmpi(dataset, 'taiji');
if (tf == 0)

  fisher_test_ConfMat = fisher_test_ConfMat(1:(A-1), 1:(B-1));

end

% Create classification matrix (rows should sum to 1)
fisher_train_ClassMat = fisher_train_ConfMat./(meshgrid(countcats(train_labels))');
fisher_test_ClassMat = fisher_test_ConfMat./(meshgrid(countcats(test_labels))');

% mean group accuracy and std
fisher_train_accuracy = mean(diag(fisher_train_ClassMat));
fisher_train_stdDev = std(diag(fisher_train_ClassMat));

fisher_test_accuracy = mean(diag(fisher_test_ClassMat));
fisher_test_stdDev = std(diag(fisher_test_ClassMat));


%Display accuracy and StdDev for fisher
display(fisher_train_accuracy)
display(fisher_test_accuracy)
display(fisher_train_stdDev)
display(fisher_test_stdDev)



