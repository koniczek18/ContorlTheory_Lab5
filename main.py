import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

L=0.1
Vin=8
R=5
c=0.01
Vd=5

A=np.array([[0,1],[-1/(c*L),-1/(c*R)]])
B=np.array([[0],[Vin/(c*L)]])
C=np.array([1,0])
D=0

def model(x,t):
    u=1
    x=np.array([x]).T
    dx=A@x+B*u
    return np.array([dx[0,0],dx[1,0]])

#
# Układ ten jest stabilny i posiada oscylacje, ponieważ jego wartości własne znajdują się
# w lewej półpłaszczyźnie i są one zespolone
#

t=np.linspace(0,20,20001)
sim=odeint(model,y0=[0,0],t=t)
plt.figure('Model')
plt.plot(t,sim[:,0],label='x1')
plt.legend()

def modelFeedback(x,t,vd,omega):
    x = np.array([x]).T
    x_d = np.array([[vd],[0]])
    e = x_d - x
    k_1 = (omega / 8000) - (1 / 8)
    k_2 = (omega / 4000) - (1 / 400)
    K = np.array([k_1, k_2])
    u_D = vd / 8
    u_FB = K @ e
    u = u_FB + u_D
    dx = A @ x + B *u
    return np.array([dx[0,0],dx[1,0]])

sim1a=odeint(modelFeedback,y0=[0,0],t=t,args=(10,5))
sim1b=odeint(modelFeedback,y0=[0,0],t=t,args=(10,10))
sim1c=odeint(modelFeedback,y0=[0,0],t=t,args=(10,25))
sim2a=odeint(modelFeedback,y0=[3,4],t=t,args=(10,5))
sim2b=odeint(modelFeedback,y0=[3,4],t=t,args=(10,10))
sim2c=odeint(modelFeedback,y0=[3,4],t=t,args=(10,25))
plt.figure('Feedback Model - y0=[0,0]')
plt.plot(t,sim1a[:,0],label='w=5')
plt.plot(t,sim1b[:,0],label='w=10')
plt.plot(t,sim1c[:,0],label='w=25')
plt.legend()
plt.figure('Feedback Model - y0=[3,4]')
plt.plot(t,sim2a[:,0],label='w=5')
plt.plot(t,sim2b[:,0],label='w=10')
plt.plot(t,sim2c[:,0],label='w=25')
plt.legend()

#
# Omegę należy dobierać w taki sposób, by sprzężenie od stanu było szybsze od sprzężenia zwrotnego
# Sterownik nie zadziała dla ujemnych wartości w_c
# Dobór w_c wpływa na szybkość stabilizacji układu
#

def modelFeedbackLuenberger(x_in, t, vd, omega):
    x = np.array([[x_in[0]], [x_in[1]]])
    x2 = np.array([[x_in[2]], [x_in[3]]])
    x_d = np.array([[vd],[0]])
    e = x_d - x2
    k1 = omega / 8000 - 1 / 8
    k2 = omega / 4000 - 1 / 400
    K = np.array([k1, k2])
    u_D = vd / 8
    u_FB = K @ e
    u = u_FB + u_D
    laambda = -omega
    l_1 = -2 * laambda - 20
    l_2 = laambda ** 2 - 1000 - 20 * l_1
    L = np.array([[l_1], [l_2]])
    temp = C @ x - C @ x2
    dx = A @ x + B * u
    dx2 = A @ x2 + B * u + L * temp
    return np.array([dx[0, 0], dx[1, 0], dx2[0, 0], dx2[1, 0]])

sim3a=odeint(modelFeedbackLuenberger,y0=[0,0,0,0],t=t,args=(10,5))
sim3b=odeint(modelFeedbackLuenberger,y0=[0,0,0,0],t=t,args=(10,10))
sim3c=odeint(modelFeedbackLuenberger,y0=[0,0,0,0],t=t,args=(10,25))
sim4a=odeint(modelFeedbackLuenberger,y0=[3,4,0,0],t=t,args=(10,5))
sim4b=odeint(modelFeedbackLuenberger,y0=[3,4,0,0],t=t,args=(10,10))
sim4c=odeint(modelFeedbackLuenberger,y0=[3,4,0,0],t=t,args=(10,25))

plt.figure('Feedback Model Luenberger - y0=[0,0]')
plt.plot(t,sim3a[:,0],label='w=5')
plt.plot(t,sim3b[:,0],label='w=10')
plt.plot(t,sim3c[:,0],label='w=25')
plt.plot(t,sim3a[:,2],label='w=5 estim.')
plt.plot(t,sim3b[:,2],label='w=10 estim.')
plt.plot(t,sim3c[:,2],label='w=25 estim.')
plt.legend()
plt.figure('Feedback Model Luenberger - y0=[3,4]')
plt.plot(t,sim4a[:,0],label='w=5')
plt.plot(t,sim4b[:,0],label='w=10')
plt.plot(t,sim4c[:,0],label='w=25')
plt.plot(t,sim4a[:,2],label='w=5 estim.')
plt.plot(t,sim4b[:,2],label='w=10 estim.')
plt.plot(t,sim4c[:,2],label='w=25 estim.')
plt.legend()

plt.show()