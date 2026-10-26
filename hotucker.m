function [A, B, C, G,nIter, obj_all] = hotucker(Xl, Xu, Yl, Yu,n1, n2,m1,m2,m3,options)


maxIter = options.maxIter;
lambda = options.lambda;
beta = options.beta;



X=[Xl,Xu];
nl=length(Yl);
nu=length(Yu);


[mFea,nSmp]=size(X);


TX=reshape(X,[n1,n2,nSmp]);
TX=tensor(TX);

TXu=reshape(Xu,[n1,n2,nu]);
TXu=tensor(TXu);

param.k = 5;
HG = gsp_nn_hypergraph(X', param); % 输入是n*m
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


A=nvecs(TX,1,m1);
B=nvecs(TX,2,m2);
C=nvecs(TX,3,m3);
G=ttm(TX,{A',B',C'});
A(A<0)=0;
B(B<0)=0;
C(C<0)=0;
G=double(G);
G(G<0)=0;
G=tensor(G);

Cu = nvecs(TXu,3,m3);
Cu(Cu<0)=0;
% A = abs(rand(n1, r));
% B = abs(rand(n2, r));
% Cu = abs(rand(nu, r));


obj_all=[];
tol=1e-5;
nIter = 0;
obj_old = CalcuObjhoTucker(TX, A, B, [Cl;Cu],G, L,lambda,beta);

while nIter < maxIter
        % ===================== update A ========================
        C = [Cl;Cu];
        G1 = tenmat(G,1);
        G1 = G1.data;
        TXBC = ttm(TX, {B',C'}, [2,3]);
        TXBC1 = tenmat(TXBC,1);
        TXBC1 = TXBC1.data;
        UPA = TXBC1 * G1';

        TGABBCC = ttm(G,{A, B'*B,C'*C},[1,2,3]);
        TGABBCC1 = tenmat(TGABBCC, 1);
        TGABBCC1 = TGABBCC1.data;
        DownA = TGABBCC1 * G1';

        A = A .* (UPA ./ max(DownA,1e-10));

       % ===================== update B ========================
        G2 = tenmat(G,2);
        G2= G2.data;
        TXAC = ttm(TX, {A',C'}, [1,3]);
        TXAC2 = tenmat(TXAC,2);
        TXAC2 = TXAC2.data;
        UPB = TXAC2 * G2';

        TGAABCC = ttm(G,{A'*A, B,C'*C},[1,2,3]);
        TGAABCC2 = tenmat(TGAABCC, 2);
        TGAABCC2 = TGAABCC2.data;
        DownB  = TGAABCC2 * G2';

        B = B .* (UPB ./ max(DownB,1e-10));

       % ===================== update Cu ========================
        G3 = tenmat(G,3);
        G3= G3.data;
        TXuAB = ttm(TXu, {A',B'}, [1,2]);
        TXuAB3 = tenmat(TXuAB,3);
        TXuAB3 = TXuAB3.data;
        TXuABG = TXuAB3 * G3';
        UPC = TXuABG + lambda *  Su * C+ beta* Cu;

        TGAABBC = ttm(G,{A'*A, B'*B,Cu},[1,2,3]);
        TGAABBC3 = tenmat(TGAABBC, 3);
        TGAABBC3 = TGAABBC3.data;
        TGAABBCG = TGAABBC3 * G3';
        Cu3 = (Cu * Cu') * Cu;
        DownC =  TGAABBCG  + lambda * Du * C + 2 * beta * Cu3;

        Cu = Cu .* (UPC ./ max(DownC,1e-10));


       % ==================== update G ==========================
        TXABC = ttm(TX, {A',B',C'}, [1,2,3]);
        TXABC = TXABC.data;

        TGAABBCC = ttm(G, {A'*A,B'*B,C'*C}, [1,2,3]);
        TGAABBCC = TGAABBCC.data;

        GD = G.data;
        GD = GD.* (TXABC ./ max(TGAABBCC, 1e-10));
        G = tensor(GD);


       obj=CalcuObjhoTucker(TX, A, B, [Cl;Cu],G, L,lambda,beta);
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
