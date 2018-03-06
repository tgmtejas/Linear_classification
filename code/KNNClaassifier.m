
function [Y_KNN_test] = KNNClaassifier(X_test,X_train,knn, Y_train, categories_Class)

numClasses= length(categories_Class);
%Y_train = double(Y_train);

[A, B] = size(X_test); % 90 x 2
[C, D] = size(X_train); % 88 x 2
EuclidDist = zeros(A,C);
Y_KNN_test =zeros(A,knn);
Y_KNN_test = categorical(Y_KNN_test);

%Find Euclidean disance: We will find Euclidean distance from 
%                        every point of test set from every pt in train set
for t = 1 : A
    Xt = X_test(t,:); 
    for r = 1 : C
        Xr = X_train(r, :);
        EuclidDist(t,r) = sqrt(sum((Xr - Xt).^2));
    end
end

KNN_matrix = [];
KNN_Indice = [];
for t = 1:A
    [Asc_order, index] = sort(EuclidDist(t,:));
    KNN_matrix(t, :) = Asc_order(1, 1:knn);
    KNN_Indice(t, :) = index(1, 1:knn);
end
        
for t = 1 : A
    for k = 1 : knn
            z = KNN_Indice(t, k); 
            y = Y_train(z,1);
            Y_KNN_test(t,k) = categories_Class(y,1);
            
    end
end

 Y_KNN_test  = mode(Y_KNN_test,2);

end