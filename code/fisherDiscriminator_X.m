function [W_fisher_Wine, Sw, Sb] = fisherDiscriminator_X(train_featureVector, train_labels, test_featureVector, test_labels,categories_Class)

numClasses= length(categories_Class);
[N,M] = size(train_featureVector);
Mean = zeros(numClasses, M);
Sk = zeros(M,M);
Sw = zeros(M,M);

Datapoints_class= [];
%Finding mean of every class and stacking it to Mean vector
for i= 1 : numClasses
    z= categories_Class(i,1);
    X= [];
    for j =1 : N  % Finding the Category matrix X
        if(train_labels(j,1) == z)
            X = [X; train_featureVector(j, :)];
                        
        end
    end
    
    [P, y] = size(X);
    Mean(i,:) = mean(X);
    
    for k = 1 : P
        Sk = Sk + (X(k, :))' * X(k, :);
    end
    
    Datapoints_class = [Datapoints_class P];
   
    Sw  = Sw + Sk;
            
    
end

%Finding mean of total dataset
m = mean(train_featureVector);


%Finding the Sb:
Sb= zeros(M, M);
%I1 = ones(M, M);
%Nk = [];
for i = 1 : numClasses
    Sb = Sb + Datapoints_class(1,i) *(Mean(i, : ) - m)' * (Mean(i, : ) - m) ;
%     Nk =  I1 ;
%     Sb = Sb + (Nk * ab);
end

%Find Eigen vector
[V,D] = eig(Sw\Sb);
%V=V';
W_fisher_Wine = V(:,1:numClasses-1);


end