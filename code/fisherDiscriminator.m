function [W_fisher_Wine, Sw, Sb] = fisherDiscriminator(train_featureVector, train_labels, test_featureVector, test_labels,categories_Class)

numClasses= length(categories_Class);
[N,M] = size(train_featureVector);
Mean = [];

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
    
    Datapoints_class = [Datapoints_class P];
   
    
    Mean(i,:) = mean(X);
        
    
end

Sk = zeros(M,M);
Sw = zeros(M,M);

for i =1 : numClasses
    z= categories_Class(i,1);
     X= [];
    for j =1 : N  % Finding the Category matrix X
        if(train_labels(j,1) == z)
            X = [X; train_featureVector(j, :)];
        end
    end
     
    for k = 1: numClasses
        [P, y] = size(X);
        I1 = ones(P,1);
        I2 = Mean(k,:);
        mk = I1 * I2 ;
        Sk=Sk + (X - mk)'*(X - mk);
    end
    
    Sw = Sw + Sk;
  
end
 
 
%Finding mean of total dataset
m = mean(train_featureVector);


%Finding the Sb:
Sb= zeros(M, M);
I1 = ones(M, M);
Nk = [];
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