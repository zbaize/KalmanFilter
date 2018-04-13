# Kalman filter intro pdf demo in Python  
 
import numpy  
import pylab  

# XT(k) = A*X(k-1)+B*U(k-1)+W(k), A=1, U(k)=0, W(k)~N(0,Q)
# Z(k) = H*X(k)+V(k), H=1, V(k)~N(0,R)
# PT(k) = A*PT(t-1)*A'+Q
# K(k) = PT(k)*H'/(H*PT(k)*H'+Q)
# X(K) = XT(k)+K(k)*(Z(k)-H*XT(k))
# P(k) = (1-H*K(k))*PT(k)

# ------------------------------------- #
# ***Prediction***                      |
# XT(k) = ComX(k-1)                     |
# PT(k) = P(k-1)+Q                      |
#                                       |
# ***Observation***                     |
# Z(k) = ComX(k)+V(k)                   |
#                                       |
# ***Combination(Correction)***         |
# K(k) = PT(k)/(PT(k)+R)                |
# ComX(k) = XT(k)+K(k)*(Z(k)-XT(k))     |
# P(k) = (1-K(k))*PT(k)                 |
# ------------------------------------- #
  
# Parameter Init
n_iter = 50
sz = (n_iter,) 
# True Value x
x = -0.37727  
# Observation Z(k)~N(x,0.01) R=0.1**2
Z = numpy.random.normal(x,0.1,size=sz) 
# Process Variance 
Q = 1e-5  
# Estimate of measurement variance, change to see effect  
R = 0.1**2

# Init Memory space 
X=numpy.zeros(sz)           # a posteri estimate of x  
P=numpy.zeros(sz)           # a posteri error estimate
XT=numpy.zeros(sz)          # a priori estimate of x 
PT=numpy.zeros(sz)          # a priori error estimate
K = numpy.zeros(sz)         # gain or blending factor  

  
# Intial guesses  
X[0] = 0.0  
P[0] = 1.0  
  
for k in range(1,n_iter):  
    # Prediction  
    XT[k] = X[k-1]              # XT(k) = ComX(k-1)
    PT[k] = P[k-1]+Q            # PT(k) = P(k-1)+Q

    # Update:Correction/Combination
    K[k] = PT[k]/(PT[k]+R)              # K(k) = PT(k)/(PT(k)+R)
    X[k] = XT[k]+K[k]*(Z[k]-XT[k])      # ComX(k) = XT(k)+K(k)(Z(k)-XT(k))
    P[k] = (1-K[k])*PT[k]               # P(k) = (1-K(k))*PT(k)  

pylab.figure()  
pylab.plot(Z,'k+',label='Observation by measurement')
pylab.plot(X,'b-',label='Estimente by filter') 
pylab.axhline(x,color='g',label='True Value')
pylab.legend()  
pylab.xlabel('Iteration')  
pylab.ylabel('Voltage')  
  
pylab.figure()  
valid_iter = range(1,n_iter) # PT not valid at step 0  
pylab.plot(valid_iter,PT[valid_iter],label='Priori Error Estimate')  
pylab.xlabel('Iteration')  
pylab.ylabel('$(Voltage)^2$')  
pylab.setp(pylab.gca(),'ylim',[0,.01])  
pylab.show()  

# http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html  
