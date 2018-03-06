function [W, numClasses, categories_Class] = TrainingModel (featureVector, labels)

X= featureVector;
%labels = double(labels);
categories_Class = categories(labels);
numClasses= length(categories_Class);

[N, M] = size(labels);
Y= zeros(N, numClasses);

% 1 to k 
for i=1 : N
    
         y= labels(i,1);
         for j = 1: numClasses
             if (categories_Class(j,1) == y)
                 Y(i, j)=1;
             end
         end
             
    
end

% Weight matrix
W = (X'*X)\(X'*Y);%W optimal calculation



end