function [A, B, C, nIter, obj_all] = hocp(Xl, Xu, Yl, Yu,r ,n1, n2,options)
% ||X-[[A,B,C]]||_{F}^{2}+\lambda*tr(C'*L*C)+\alpha||C'_{u}C_{u}-I/2||_{F}^{2}
%
% X ... (mFea x nSmp)data matrix 
%       mFea  ... number of dimensions 
%       nSmp  ... number of samples
% T ... (n1,n2,nSmp)tensor data 
%       mfea=n1*n2
% r...  number of cp rank
% A ... (n1 x r) factor matrix
% B ... (n2 x r) factor matrix
% C ... (nSmp x r) represenatation matrix
% Cl ... (nl x r) labeled represenatation matrix
%          nl ... number of labeled samples
% Cu ... (nu x r) unlabeled represenatation matrix
%           nu ... number of unlabeled samples


maxIter = options.maxIter;
lambda = options.lambda;
beta = options.beta;



X=[Xl,Xu];
nl=length(Yl);
nu=length(Yu);


[mFea,nSmp]=size(X);


T=reshape(X,[n1,n2,nSmp]);
T1= (double(tenmat(T, 1)));
T2= (double(tenmat(T, 2)));
T3= (double(tenmat(T, 3)));

Tu=reshape(Xu,[n1,n2,nu]);
Tu3=(double(tenmat(Tu, 3)));


param.k = 5;
HG = gsp_nn_hypergraph(X', param); 
% L = Dv - S
L = HG.L;
D = diag(HG.dv);
S = D - L;
Su =S(nl+1:end,:);
Du = D(nl+1:end,:);

%% Initialization   
Y = [Yl;Yu];     
nCluster = length(unique(Y));
YY = [];
for i = reshape(unique(Y),1,nCluster)
    YY = [YY,Y==i];
end
YYl = YY(1:nl,:);
Cl = YYl;
Cl = Cl*diag((1./sqrt(2*sum(Cl.^2))));


A = abs(rand(n1, r));
B = abs(rand(n2, r));
Cu = abs(rand(nu, r));


obj_all=[];
tol=1e-5;
nIter = 0;
obj_old = CalcuObjhoCP(T1, A, B, [Cl;Cu], L,lambda,beta);

while nIter < maxIter
        % ===================== update A ========================
        C = [Cl;Cu];
        CkhaB = khatrirao(C,B);
        UPA = T1 * CkhaB;
        DownA = A * (CkhaB' * CkhaB);
        A = A .* (UPA ./ max(DownA,1e-10));
        
       % ===================== update B ========================
       CkhaA = khatrirao(C,A);
       UPB = T2 * CkhaA;
       DownB = B * (CkhaA' * CkhaA);
       B = B .* (UPB ./ max(DownB,1e-10));
       
       % ===================== update Cu ========================
       BkhaA = khatrirao(B,A);
       UPC = Tu3 * BkhaA + lambda *  Su * C+ beta* Cu;
       Cu3 = (Cu * Cu') * Cu;
       DownC =  Cu * (BkhaA' * BkhaA) + lambda * Du * C + 2 * beta * Cu3;
       Cu = Cu .* (UPC ./ max(DownC,1e-10));
        
        
       obj=CalcuObjhoCP(T1, A, B, C, L,lambda,beta);
       obj_all=[obj_all,obj];
       err=abs((obj_old-obj)/obj);
       fprintf('obj=%f,err=%f\n',obj,err);
       if err < tol
           break;
       end
       nIter = nIter+1;
       obj_old=obj;



end
end
