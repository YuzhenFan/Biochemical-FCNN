function dydt = multi_UD_Sig_jug(t,y,X,p,al,rate)
row=3;col=size(X,2); %样本和batch size的设置
o=32;k1=0.01;k2=0.001;k3=0.01;k4=0.001;kf=10;%初始化参数
% G=[-3,0,3,0;0,-3,0,3;-1,-1,1,1;1,0,-1,0;0,1,0,-1];
kk=[k1;k3;k2;k4];K=diag(kk,0);%初始化\gamma,K矩阵
kk1=8;kk2=1;kk3=2;kk4=0.399375;%双稳定速率常数0.45(实际误差限在0.45-0.5之间)
% kk1=2;kk2=0.5;kk3=2;kk4=0.045;%双稳定速率常数0.1(实际误差限在?之间)
% kk1=4;kk2=0.25;kk3=2;kk4=0.045;%双稳定速率常数0.2
for i=1:o
    x(i)=y(i);%o
end
%S矩阵
for j=1:row % dimension+1,3
for k=1:p % batch size
    S(j,k)=y(k+o+(j-1)*p);%p*row                                                                                                                   
end
end
%C矩阵
for i=1:col % dimension+1,3 按行写矩阵
  for j=1:p % batch size 
    C(i,j)=y(j+o+row*p+(i-1)*p);%p*col
  end
end
%C~矩阵
for i=1:col % dimension+1,3 按行写矩阵
  for j=1:p % batch size 
    CC(i,j)=y(j+o+row*p+p*col+(i-1)*p);%p*col
  end
end
%输入-隐层权重矩阵（positive）
for i=1:2
    for j=1:3
        W1p(i,j)=y(j+o+row*p+2*p*col+(i-1)*3);%6
    end
end
%输入-隐层权重矩阵（positive）
for i=1:2
    for j=1:3
        W1n(i,j)=y(j+o+row*p+2*p*col+6+(i-1)*3);%6
    end
end
%隐层输入positive矩阵
for i=1:2
    for j=1:p
        Np(i,j)=y(j+o+row*p+2*p*col+12+(i-1)*p);%2*p
    end
end
%隐层输入negative矩阵
for i=1:2
    for j=1:p
        Nn(i,j)=y(j+o+row*p+2*p*col+12+2*p+(i-1)*p);%2*p
    end
end
%Sigmoid_Hp
for i=1:2
    for j=1:p
        Hp(i,j)=y(j+o+row*p+2*p*col+12+2*2*p+(i-1)*p);%2*p
    end
end
%Sigmoid_Hn
for i=1:2
    for j=1:p
        Hn(i,j)=y(j+o+row*p+2*p*col+12+3*2*p+(i-1)*p);%2*p
    end
end
%Sigmoid_H
for i=1:2
    for j=1:p
        H(i,j)=y(j+o+row*p+2*p*col+12+4*2*p+(i-1)*p);%2*p
    end
end
%隐层-输出向量(p)
for i=1:3
    W2p(i)=y(i+o+row*p+2*p*col+12+5*2*p);%3
end
%隐层-输出向量(p)
for i=1:3
    W2n(i)=y(i+o+row*p+2*p*col+12+5*2*p+3);%3
end
%输出输入positive向量
for i=1:p
    N3p(i)=y(i+o+row*p+2*p*col+12+5*2*p+6);%p
end
%输出输入negative向量
for i=1:p
    N3n(i)=y(i+o+row*p+2*p*col+12+5*2*p+6+p);%p
end
%Sigmoid_Yp.Yn,Y
for i=1:p
    Yp(i)=y(i+o+row*p+2*p*col+12+5*2*p+6+2*p);%p
end
for i=1:p
    Yn(i)=y(i+o+row*p+2*p*col+12+5*2*p+6+3*p);%p
end
for i=1:p
    Y(i)=y(i+o+row*p+2*p*col+12+5*2*p+6+4*p);%p
end
%Pre_computing
for i=1:7
    for j=1:p
    P(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+(i-1)*p);%7*p
    end
end
for i=1:4
    for j=1:p
    PH1(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+7*p+(i-1)*p);%4*p
    end
end
for i=1:4
    for j=1:p
    PH2(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+7*p+4*p+(i-1)*p);%4*p
    end
end
for i=1:2
e(i)=y(i+o+row*p+2*p*col+12+5*2*p+6+5*p+7*p+2*4*p);%1
end
%Judgement
for i=1:4
    J(i)=y(i+2+o+row*p+2*p*col+12+5*2*p+6+5*p+7*p+2*4*p);%2
end
%UD
for i=1:9
    for j=1:p
        UB(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+7*p+2*4*p+6+(i-1)*p);%9*p
    end
end
for i=1:12
    for j=1:p
        UT(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+(i-1)*p);%12*p
    end
end
for i=1:14
    for j=1:p
        DEL(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+12*p+(i-1)*p);%14*p
    end
end
for i=1:3
    for j=1:3
        Parp(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+12*p+14*p+(i-1)*3);%9
    end
end
for i=1:3
    for j=1:3
        Parn(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+12*p+14*p+9+(i-1)*3);%9
    end
end
for i=1:3
    for j=1:3
        Gammap(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+12*p+14*p+2*9+(i-1)*3);%9
    end
end
for i=1:3
    for j=1:3
        Gamman(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+12*p+14*p+3*9+(i-1)*3);%9
    end
end
for i=1:3
    for j=1:3
        UP(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+12*p+14*p+4*9+(i-1)*3);%9
    end
end
for i=1:3
    for j=1:3
        UN(i,j)=y(j+o+row*p+2*p*col+12+5*2*p+6+5*p+...
                 7*p+2*4*p+6+9*p+12*p+14*p+5*9+(i-1)*3);%9
    end
end
%pc_matrix
A1 =[-1,0,0;1,-1,0;1,0,-1;1,0,0;0,-1,0;0,0,-1];
GP=[A1,zeros(6,6);zeros(3,3),eye(3),-eye(3)];
V23=P(1:2,:).*[S(3,:);P(4,:)];
Vp=kf*[Y;V23;S(3,:);P(1,:);P(4:7,:)];%P速率函数
Pdot=GP*Vp;
A2=[-1,0;1,-1;1,0;0,-1];
GH1=[A2,zeros(4,2);zeros(1,2),1,-1];
VV1=PH1(1,:).*PH1(3,:);
Vph1=kf*[H(1,:);VV1;PH1(3:4,:)];%PH1速率函数
PH1dot=GH1*Vph1;
VV2=PH2(1,:).*PH2(3,:);
Vph2=kf*[H(2,:);VV2;PH2(3:4,:)];%PH2速率函数
PH2dot=GH1*Vph2;
%oscillator
xdot(1)=rate*(-x(1)*x(2)+x(1)*x(32));%assignment
xdot(2)=rate*(x(1)*x(2)-x(2)*x(3));
xdot(3)=rate*(x(2)*x(3)-x(3)*x(4));%控制样本在每次循环开始时发生变化
xdot(4)=rate*(x(3)*x(4)-x(4)*x(5));
xdot(5)=rate*(x(4)*x(5)-x(5)*x(6));%控制样本在每次循环开始时发生变化
xdot(6)=rate*(x(5)*x(6)-x(6)*x(7));
xdot(7)=rate*(x(6)*x(7)-x(7)*x(8));%linear weighted sum
xdot(8)=rate*(x(7)*x(8)-x(8)*x(9));
xdot(9)=rate*(x(8)*x(9)-x(9)*x(10));
xdot(10)=rate*(x(9)*x(10)-x(10)*x(11));
xdot(11)=rate*(x(10)*x(11)-x(11)*x(12));
xdot(12)=rate*(x(11)*x(12)-x(12)*x(13));
xdot(13)=rate*(x(12)*x(13)-x(13)*x(14));
xdot(14)=rate*(x(13)*x(14)-x(14)*x(15));
xdot(15)=rate*(x(14)*x(15)-x(15)*x(16));
xdot(16)=rate*(x(15)*x(16)-x(16)*x(17));
xdot(17)=rate*(x(16)*x(17)-x(17)*x(18));
xdot(18)=rate*(x(17)*x(18)-x(18)*x(19));
xdot(19)=rate*(x(18)*x(19)-x(19)*x(20));
xdot(20)=rate*(x(19)*x(20)-x(20)*x(21));
xdot(21)=rate*(x(20)*x(21)-x(21)*x(22));
xdot(22)=rate*(x(21)*x(22)-x(22)*x(23));
xdot(23)=rate*(x(22)*x(23)-x(23)*x(24));
xdot(24)=rate*(x(23)*x(24)-x(24)*x(25));
xdot(25)=rate*(x(24)*x(25)-x(25)*x(26));
xdot(26)=rate*(x(25)*x(26)-x(26)*x(27));
xdot(27)=rate*(x(26)*x(27)-x(27)*x(28));
xdot(28)=rate*(x(27)*x(28)-x(28)*x(29));
xdot(29)=rate*(x(28)*x(29)-x(29)*x(30));
xdot(30)=rate*(x(29)*x(30)-x(30)*x(31));
xdot(31)=rate*(x(30)*x(31)-x(31)*x(32));
xdot(32)=rate*(x(31)*x(32)-x(32)*x(1));
%Assignment
Sdot=(X*C-S)*x(1)+x(23)*[zeros(2,p);Pdot(5,:)];
Sdot=Sdot';sdot=Sdot(:);
for l=1:p
   for j=l+p:p:(col-p+l)
    Cdot(j,l)= -x(3)*C(j,l)*5+x(5)*CC(j-p,l)*5;
    CCdot(j,l)=x(3)*C(j,l)*5-x(5)*CC(j,l)*5;
   end
    Cdot(l,l)= -x(3)*C(l,l)*5+x(5)*CC(col-p+l,l)*5;
    CCdot(l,l)=x(3)*C(l,l)*5-x(5)*CC(l,l)*5;
end
Cdot=Cdot';cdot=Cdot(:);
CCdot=CCdot';ccdot=CCdot(:);
%Linear weight summation(1) & NAE(1)
W1pdot=x(7)*zeros(2,3)+(J(4)+J(2))*x(27)*(UP(1:2,:)+Gammap(1:2,:)-W1p);
W1ndot=x(7)*zeros(2,3)+(J(4)+J(2))*x(27)*(UN(1:2,:)+Gamman(1:2,:)-W1n);
G12pdot=(x(7)*(W1p-Gammap(1:2,:)))';g12pdot=G12pdot(:);
G12ndot=(x(7)*(W1n-Gamman(1:2,:)))';g12ndot=G12ndot(:);
SS=[S(1,:);S(2,:);ones(1,p)];
%N1
Npdot=(W1p*SS-Np)*x(7)+x(9)*(-Np.*Nn)*10+x(13)*(-Np);
Nndot=(W1n*SS-Nn)*x(7)+x(9)*(-Np.*Nn)*10+x(13)*(-Nn);
W1pdot=W1pdot';w1pdot=W1pdot(:);W1ndot=W1ndot';w1ndot=W1ndot(:);
Npdot=Npdot';npdot=Npdot(:);Nndot=Nndot';nndot=Nndot(:);
O1dot=[w1pdot;w1ndot;npdot;nndot];
HT=repmat(1/2,[2 p]);
Hpdot=(x(11)*(Np.*(HT-Hp)*5)+x(13)*(Np.*(Hp-Hp.^2)))'+x(29)*(-Hp)';hpdot=Hpdot(:);
Hndot=(x(11)*(Nn.*(HT-Hn)*5)+x(13)*(Nn.*(Hn.^2-Hn)))'+x(29)*(-Hn)';hndot=Hndot(:);
Hdot=(x(13)*(Hp+Hn-H)+x(23)*[PH1dot(1,:);PH2dot(1,:)])';hdot=Hdot(:);
O2dot=[hpdot;hndot;hdot];
%LMS(2)，output
W2pdot=x(15)*zeros(1,3)+(J(4)+J(2))*x(27)*(UP(3,:)+Gammap(3,:)-W2p);
W2ndot=x(15)*zeros(1,3)+(J(4)+J(2))*x(27)*(UN(3,:)+Gamman(3,:)-W2n);
G3pdot=(x(15)*(W2p-Gammap(3,:)))';g3pdot=G3pdot(:);
G3ndot=(x(15)*(W2n-Gamman(3,:)))';g3ndot=G3ndot(:);
W2pdot=W2pdot';w2pdot=W2pdot(:);W2ndot=W2ndot';w2ndot=W2ndot(:);
HH=[H;ones(1,p)];
%N3
N3pdot=x(15)*(W2p*HH-N3p)+x(17)*(-N3p.*N3n)*10+x(21)*(-N3p);
N3ndot=x(15)*(W2n*HH-N3n)+x(17)*(-N3p.*N3n)*10+x(21)*(-N3n);
n3pdot=N3pdot';n3ndot=N3ndot';
O3dot=[w2pdot;w2ndot;n3pdot;n3ndot];
%Y
YT=repmat(1/2,[1 p]);
Ypdot=(x(19)*(N3p.*(YT-Yp)*5)+x(21)*(N3p.*(Yp-Yp.^2)))'+x(29)*(-Yp)';ypdot=Ypdot(:);
Yndot=(x(19)*(N3n.*(YT-Yn)*5)+x(21)*(N3n.*(Yn.^2-Yn)))'+x(29)*(-Yn)';yndot=Yndot(:);
Ydot=(x(21)*(Yp+Yn-Y)+x(23)*Pdot(1,:))';ydot=Ydot(:);
O4dot=[ypdot;yndot;ydot];
%J_matrix
AJ=[2,-1,-1,-1;-1,1,0,0];
GJ=[AJ,zeros(2,2);zeros(1,4),5,-1];%5是为了放大原来的催化剂浓度1.8
kkj=[kk1;kk2;kk3;kk4;20;20];%20是为了放大E赋值给C的反应速率
KKJ=diag(kkj,0);
VJ1=KKJ*[J(1);[e(1);e(1)].*[e(1);J(1)];e(1);e(1);J(2)];
J1dot=GJ*VJ1;
VJ2=KKJ*[J(3);[e(2);e(2)].*[e(2);J(3)];e(2);e(2);J(4)];
J2dot=GJ*VJ2;
%PC
PPdot=(x(23)*[Pdot(2:4,:);Pdot(6:9,:)]+...
       10*x(29)*[-P(1:3,:);ones(1,p)-P(4,:);-P(5:7,:)])';
ppdot=PPdot(:);
E=P(5:6,:);edot=(E(1,:)+E(2,:)-e)*x(23)+50*x(25)*[J1dot(1),J2dot(1)]+10*x(29)*(-e);
PPH1dot=(x(23)*PH1dot(2:5,:)+10*x(29)*[-PH1(1:2,:);ones(1,p)-PH1(3,:);-PH1(4,:)])';
PPH2dot=(x(23)*PH2dot(2:5,:)+10*x(29)*[-PH2(1:2,:);ones(1,p)-PH2(3,:);-PH2(4,:)])';
pph1dot=PPH1dot(:);pph2dot=PPH2dot(:);
O5dot=[ppdot;pph1dot;pph2dot;edot'];
%J
jdot=x(25)*[J1dot(2:3);J2dot(2:3)]*50+10*x(29)*(-J');
%UD
UB1=[E;E;P(3,:)].*[PH1(2,:);PH1(2,:);PH2(2,:);PH2(2,:);P(7,:)];
UB2=[W2p(1)*PH1(4,:);W2n(1)*PH1(4,:);...
    W2p(2)*PH2(4,:);W2n(2)*PH2(4,:)];
UBdot=(([UB1;UB2]-UB)*x(27)*(J(4)+J(2)))';ubdot=UBdot(:);
UT1=[UB(1:4,:);UB(6:7,:);UB(6:7,:);UB(8:9,:);UB(8:9,:)].*...
   [UB(5,:);UB(5,:);UB(5,:);UB(5,:);S(1,:);S(1,:);S(2,:);S(2,:);...
   S(1,:);S(1,:);S(2,:);S(2,:)];
UTdot=((UT1-UT)*x(27)*(J(4)+J(2)))';utdot=UTdot(:);
DEL1=[UT(1:4,:);UT(1:4,:)].*[UT(5,:);UT(5,:);UT(9,:);UT(9,:);...
                             UT(7,:);UT(7,:);UT(11,:);UT(11,:)];
DEL2=[UT(2,:);UT(1,:);UT(4,:);UT(3,:);UT(2,:);UT(1,:);UT(4,:);UT(3,:);].*...
     [UT(6,:);UT(6,:);UT(10,:);UT(10,:);UT(8,:);UT(8,:);UT(10,:);UT(10,:)];
DEL3=[UT(1,:);UT(1,:);UT(3,:);UT(3,:)].*UB(6:9,:);
DEL4=[UT(2,:);UT(2,:);UT(4,:);UT(4,:)].*[UB(7,:);UB(6,:);UB(9,:);UB(8,:)];
LEFT=[DEL1;DEL3;E.*[UB(5,:);UB(5,:)]];RIGHT=[DEL2;DEL4;zeros(2,p)];
DELdot=((LEFT+RIGHT-DEL)*x(27)*(J(4)+J(2)))';deldot=DELdot(:);
Parpp=[sum(DEL(1,:)),sum(DEL(5,:)),sum(DEL(9,:));...
       sum(DEL(3,:)),sum(DEL(7,:)),sum(DEL(11,:));...
       sum(UT(1,:)),sum(UT(3,:)),sum(DEL(13,:))];
Parnn=[sum(DEL(2,:)),sum(DEL(6,:)),sum(DEL(10,:));...
       sum(DEL(4,:)),sum(DEL(8,:)),sum(DEL(12,:));...
       sum(UT(2,:)),sum(UT(4,:)),sum(DEL(14,:))];
Parpdot=((Parpp-Parp)*(J(4)+J(2))*x(27))';Parndot=((Parnn-Parn)*(J(4)+J(2))*x(27))';
pardot=[Parpdot(:);Parndot(:)];
Gdot=[g12pdot;g3pdot;g12ndot;g3ndot];
O6dot=[ubdot;utdot;deldot;pardot;Gdot];
UPdot=((J(4)+J(2))*x(27)*(al*Parp-UP))';updot=UPdot(:);
UNdot=((J(4)+J(2))*x(27)*(al*Parn-UN))';undot=UNdot(:);
O7dot=[updot;undot];
dydt=[xdot';sdot;cdot;ccdot;O1dot;O2dot;O3dot;...
     O4dot;O5dot;jdot;O6dot;O7dot];