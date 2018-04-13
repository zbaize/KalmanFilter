% Kalman Filter

% Prediction
% X(t) = AX(t-1)+BU(t)+W(t), A=1, BU(t)=0, W(t)~N(0,Q)
clear;
N = 200;
% Q = (0.001)^2
W = randn(1,N)*0.001;
VarW = std(W).^2;
X(1) = 0;
A = 1;
for t = 2:N
    X(t) = A*X(t-1)+W(t);
end

% Observation
% Z(t) = CX(t)+V(t), C=0.2, V(t)~N(0,R)
% R = (0.1)^2
V = randn(1,N)*0.1;
C = 0.2;
Z = C*X+V;
VarV = std(V).^2;

% Integration(Corrction)
VarP(1) = 0;
Com(1) = 0;
for t = 2:N
    VarPT(t) = A*VarP(t-1)+VarW;
    Gain(t) = VarPT(t)*C/(C.^2*VarPT(t)+VarV);
    Com(t) = Com(t-1)+Gain(t)*(Z(t)-C*Com(t-1));
    VarP(t) = VarPT(t)*(1-C*Gain(t));
end

% Plot
t = 1:N;
plot(t,Com,'r',t,Z,'g',t,X,'b');
legend('Combination:Com','Measurement:Z','SysEstimate:X');