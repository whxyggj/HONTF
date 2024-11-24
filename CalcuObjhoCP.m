function [obj] = CalcuObjhoCP(T1, A, B, C, L,lambda,beta)
    sum1 = sum(sum((C' * L) .* C')); %tr(C'LC)
    dC = C'*C-eye(size(C,2));
    sum2=sum(sum(dC.^2));
    obj_Lap = lambda*sum1 + beta* sum2;
    CkhaB = khatrirao(C,B);
    dX = T1 - A * CkhaB';
    obj_cp = sum(sum(dX.^2));
    obj = obj_cp + obj_Lap;
end
