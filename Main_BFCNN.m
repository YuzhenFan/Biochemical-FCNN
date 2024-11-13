%%
%batch size = 2
load('name of the dataset.mat')
X=[train_data';train_labels'];
X=double(X);
tspan=[0 9000];
p=2;col=size(X,2);al=0.4;rate=1;
x0=[0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;1;1];
S0=zeros(3,p)';S0=S0(:);
C0=[eye(p);zeros(col-p,p)]';C0=C0(:);
CC0=zeros(col,p)';CC0=CC0(:);
W1p0=W1p';W1p0=W1p0(:);%W1p, W1n are found in the initial concentration mat file 
W1n0=W1n';W1n0=W1n0(:);
N1p0=zeros(p,2);N1n0=zeros(p,2);
N1p0=N1p0(:);N1n0=N1n0(:);
O10=[W1p0;W1n0;N1p0;N1n0];
Hp0=zeros(p,2);Hp0=Hp0(:);
Hn0=zeros(p,2);Hn0=Hn0(:);
H0=zeros(p,2);H0=H0(:);
O20=[Hp0;Hn0;H0];
W2p0=W2p';%W2p, W2n are found in the initial concentration mat file 
W2n0=W2n';
N3p0=zeros(p,1);N3n0=zeros(p,1);
O30=[W2p0;W2n0;N3p0;N3n0];
Yp0=zeros(p,1);
Yn0=zeros(p,1);
Y0=zeros(p,1);
O40=[Yp0;Yn0;Y0];
P0=[zeros(3,p);ones(1,p);zeros(3,p)]';
PH10=[zeros(2,p);ones(1,p);zeros(1,p)]';
PH20=[zeros(2,p);ones(1,p);zeros(1,p)]';
P0=P0(:);PH10=PH10(:);PH20=PH20(:);
e0=zeros(p,1);
O50=[P0;PH10;PH20;e0];
J0=zeros(2*p,1);
UB0=zeros(9,p)';UB0=UB0(:);
UT0=zeros(12,p)';UT0=UT0(:);
DEL0=zeros(14,p)';DEL0=DEL0(:);
Parp0=zeros(3,3)';Parp0=Parp0(:);
Parn0=zeros(3,3)';Parn0=Parn0(:);
Gammap0=zeros(3,3)';Gammap0=Gammap0(:);
Gamman0=zeros(3,3)';Gamman0=Gamman0(:);
UP0=zeros(3,3)';UP0=UP0(:);
UN0=zeros(3,3)';UN0=UN0(:);
O60=[UB0;UT0;DEL0;Parp0;Parn0;Gammap0;Gamman0;UP0;UN0];
y0=[x0;S0;C0;CC0;O10;O20;O30;O40;O50;J0;O60];
y0=double(y0);
options = odeset('RelTol', 1e-2, 'AbsTol', 1e-5);
[t, y] = ode45(@(t,y) multi_UD_Sig_jug(t,y,X,p,al,rate), tspan, y0, options);
%%
%batch size =4
load('name of the dataset.mat')
X=[train_data';train_labels'];
X=double(X);
tspan=[0 10000];
p=4;col=size(X,2);al=0.4;rate=1;
x0=[0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;1;1];
S0=zeros(3,p)';S0=S0(:);
C0=[eye(p);zeros(col-p,p)]';C0=C0(:);
CC0=zeros(col,p)';CC0=CC0(:);
W1p0=W1p';W1p0=W1p0(:);
W1n0=W1n';W1n0=W1n0(:);
N1p0=zeros(p,2);N1n0=zeros(p,2);
N1p0=N1p0(:);N1n0=N1n0(:);
O10=[W1p0;W1n0;N1p0;N1n0];
Hp0=zeros(p,2);Hp0=Hp0(:);
Hn0=zeros(p,2);Hn0=Hn0(:);
H0=zeros(p,2);H0=H0(:);
O20=[Hp0;Hn0;H0];
W2p0=W2p';
W2n0=W2n';
N3p0=zeros(p,1);N3n0=zeros(p,1);
O30=[W2p0;W2n0;N3p0;N3n0];
Yp0=zeros(p,1);
Yn0=zeros(p,1);
Y0=zeros(p,1);
O40=[Yp0;Yn0;Y0];
P0=[zeros(3,p);ones(1,p);zeros(3,p)]';%PC
PH10=[zeros(2,p);ones(1,p);zeros(1,p)]';
PH20=[zeros(2,p);ones(1,p);zeros(1,p)]';
P0=P0(:);PH10=PH10(:);PH20=PH20(:);
e0=zeros(p,1);
O50=[P0;PH10;PH20;e0];
J0=zeros(2*p,1);
UB0=zeros(9,p)';UB0=UB0(:);
UT0=zeros(12,p)';UT0=UT0(:);
DEL0=zeros(14,p)';DEL0=DEL0(:);
Parp0=zeros(3,3)';Parp0=Parp0(:);
Parn0=zeros(3,3)';Parn0=Parn0(:);
Gammap0=zeros(3,3)';Gammap0=Gammap0(:);
Gamman0=zeros(3,3)';Gamman0=Gamman0(:);
UP0=zeros(3,3)';UP0=UP0(:);
UN0=zeros(3,3)';UN0=UN0(:);
O60=[UB0;UT0;DEL0;Parp0;Parn0;Gammap0;Gamman0;UP0;UN0];
y0=[x0;S0;C0;CC0;O10;O20;O30;O40;O50;J0;O60];
y0=double(y0);
options = odeset('RelTol', 1e-2, 'AbsTol', 1e-5);
[t, y] = ode45(@(t,y) multi_UD_Sig_jug_changep(t,y,X,p,al,rate), tspan, y0, options);
%%
%feedforward for test output
load('name of the dataset.mat')
X=[test_data';test_labels'];
tspan=[0 200];
p=12;col=size(test_data',2);al=0.4;
x0=[0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    0.000001;0.000001;0.000001;0.000001;...
    1;1];
S0=zeros(3,p)';S0=S0(:);
C0=[eye(p);zeros(col-p,p)]';C0=C0(:);
CC0=zeros(col,p)';CC0=CC0(:);
W1p0=W1p_final';W1p0=W1p0(:);%Wp_final can be found in the supplementary material of the paper
W1n0=W1n_final';W1n0=W1n0(:);
N1p0=zeros(p,2);N1n0=zeros(p,2);
N1p0=N1p0(:);N1n0=N1n0(:);
O10=[W1p0;W1n0;N1p0;N1n0];
Hp0=zeros(p,2);Hp0=Hp0(:);
Hn0=zeros(p,2);Hn0=Hn0(:);
H0=zeros(p,2);H0=H0(:);
O20=[Hp0;Hn0;H0];
W2p0=W2p_final';
W2n0=W2n_final';
N3p0=zeros(p,1);N3n0=zeros(p,1);
O30=[W2p0;W2n0;N3p0;N3n0];
Yp0=zeros(p,1);
Yn0=zeros(p,1);
Y0=zeros(p,1);
O40=[Yp0;Yn0;Y0];
y0=[x0;S0;C0;CC0;O10;O20;O30;O40];
y0=double(y0);
[t,y]=ode45(@(t,y) multi_UD_Sig_jug_feedf(t,y,X,p),tspan,y0);