function [obj] = CalcuObjhoTucker(TX, A, B, C, G, L,lambda,beta)
    sum1 = sum(sum((C' * L) .* C')); %tr(F'LF)+tr[(F-A)'S(F-A)]
    dC = C'*C-eye(size(C,2));
    sum2=sum(sum(dC.^2));
    obj_Lap = lambda*sum1 + beta* sum2;
    GABC = ttm(G, {A,B,C}, [1,2,3]);
    dX = TX - GABC;
    dX1 = tenmat(dX,1);
    dX1=dX1.data;
    obj_tucker = sum(sum(dX1.^2));
    obj = obj_tucker + obj_Lap;
end
