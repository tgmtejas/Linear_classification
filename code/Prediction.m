function [Y_pred] = Prediction (W,X, numClasses, categories_Class)

Y_values = X * W;

%New predict of training label 
A= [];
Y_pred=[];
Y_pred = categorical(Y_pred);
val =[];
[N, M] = size(Y_values);

for i = 1 : N
   
   for j= 1 :  numClasses
       A(1,j) = Y_values(i,j);
   end
   
   [val(i,1), z] = max(A);

   %Y_pred(i,1) = z;
   
   Y_pred(i,1) = categories_Class(z,1);
   
end


end