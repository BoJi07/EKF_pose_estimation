#-------------------------------------------------------------------------------
# Name:        unscented_kalman_filter
# Purpose:     estimate robot pose from laser reading in a unknown map
#
# Author:      Bo Ji
#
# Created:     25/10/2019

#-------------------------------------------------------------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt
# global variable
std_a = 0.01
std_w = 0.01
var_r = 0.01
var_phi = 10
n = 3
n_aug = 5
alpha = 3-n_aug
def getData(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)

    t = data['t']  # time stamp
    x_init = data['x_init'] #init x
    y_init = data['y_init'] #init_y
    yaw_init = data['th_init'] #initial heading angle

    v = data['v'] #traslation velcoty of robot
    om = data['om']  # rotational velocity input [rad/s]
    b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
    r = data['r']  # range measurements [m]
    l = data['l']  # x,y positions of landmarks [m]
    d = data['d']  # distance between robot center and laser rangefinder [m]

    return t,x_init,y_init,yaw_init,v,om,b,r,l,d

def generateSigmaPoints(p,x):
    # generate augment covariance
    aug_p = np.zeros((5,5))
    aug_p[0:3,0:3] = p
    aug_p[3,3] = std_a*std_a
    aug_p[4,4] = std_w*std_w
    #cholesky decomposition
    L = np.linalg.cholesky(aug_p)
    m = aug_p.shape[0]
    x_sig = np.zeros((2*n_aug+1,m))
    # x dim 5*1 n = 5
    x_sig[0] = x
    w = np.sqrt(alpha+n_aug)
    for i in range(n_aug):
        x_sig[i+1] = x + w*L[:,i].T
        x_sig[i+1+n_aug] = x-w*L[:,i].T

    return x_sig

def compute_mean_covariance(x_pred,dim):
    k = x_pred.shape[0]
    weight = np.zeros(k)
    weight[0] = alpha/(n_aug+alpha)
    for i in range(1,k):
        weight[i] = 1/(2*(alpha+n_aug))
    new_mean = np.zeros(dim)
    new_covariance = np.zeros((dim,dim))
    for i in range(k):
        new_mean += weight[i]*x_pred[i]
    if dim==3:
        new_mean[2]  =wraptopi(new_mean[2])
    else:
        new_mean[1] = wraptopi(new_mean[1])
    for i in range(k):
        x_diff = x_pred[i]-new_mean
        if dim==3:
            x_diff[2] = wraptopi(x_diff[2])
        else:
            x_diff[1] = wraptopi(x_diff[1])
        new_covariance += weight[i]*(np.mat(x_diff).T.dot(np.mat(x_diff)))

    return new_mean, new_covariance

def wraptopi(x):
    while x > np.pi:
        x = x - 2* np.pi
    while x < -np.pi:
        x = x + 2 * np.pi
    return x

def predict_motion(x_sig,dt,v,w):
    k = x_sig.shape[0]
    x_pred = np.zeros((2*n_aug+1,3))
    for i in range(k):
        std_a = x_sig[i,3]
        std_w = x_sig[i,4]
        x_pred[i,0] = x_sig[i,0]+dt*np.cos(x_sig[i,2])*(v+0.5*std_a*dt**2)
        x_pred[i,1] = x_sig[i,1]+dt*np.sin(x_sig[i,2])*(v+0.5*std_w*dt**2)
        x_pred[i,2] = x_sig[i,2]+dt*(w+0.5*std_w*dt**2)
        x_pred[i,2] = wraptopi(x_pred[i,2])

    return x_pred

def measurement_update(x_pred,landmark,dist,new_mean,new_covariance,r,b):
    k = x_pred.shape[0]
    z_sig = np.zeros((k,2))
    for i in range(k):
        z_sig[i,0] = np.sqrt((landmark[0]-new_mean[0]-dist*np.cos(new_mean[2]))**2 +(landmark[1]-new_mean[1]-dist*np.sin(new_mean[2]))**2)
        z_sig[i,1] = np.arctan2(landmark[1]-new_mean[1]-dist*np.sin(new_mean[2]),landmark[0]-new_mean[0]-dist*np.cos(new_mean[2]))-new_mean[2]
        z_sig[i,1] = wraptopi(z_sig[i,1])
    new_z,S = compute_mean_covariance(z_sig,2)
    new_z[1] = wraptopi(new_z[1])
    R = np.identity(2)
    R[0,0] = var_r
    R[1,1] = var_phi
    S = S + R
    #cross-correlation matrix
    weight = np.zeros(k)
    weight[0] = alpha/(n_aug+alpha)
    T = np.zeros((3,2))
    for i in range(1,k):
        weight[i] = 1/(2*(n_aug+alpha))
    for i in range(k):
        T += weight[i]*np.mat(x_pred[i]-new_mean).T.dot(np.mat(z_sig[i]-new_z))
    K_k = T.dot(np.linalg.inv(S))
    z = np.array([r,wraptopi(b)])
    x_est = new_mean + K_k.dot(z-new_z)
    x_est[2] = wraptopi(x_est[2])
    covariance_est = new_covariance - K_k.dot(S).dot(K_k.T)
    return x_est,covariance_est

def main():
    path = 'data.pickle'
    time_stamp,x_init,y_init,yaw_init,vel,om,bearing,rang,landmark,dist = getData(path)
    p = 0.01*np.identity(3)
    x = np.array([x_init,y_init,yaw_init])
    x_est = np.zeros([len(time_stamp),3])
    p_est = np.zeros([len(time_stamp),3,3])
    x_est[0] = x
    p_est[0] = p
    for i in range(1,len(time_stamp)):
        dt = time_stamp[i]- time_stamp[i-1]
        # step 1 generate sigma points
        x_aug = np.zeros(5)
        x[2] = wraptopi(x[2])
        x_aug[0:3] = x;
        x_sig = generateSigmaPoints(p,x_aug)
        # step 2 use sigma point to predict motion
        x_pred = predict_motion(x_sig,dt,vel[i-1],om[i-1])
        # step 3 compute predicted mean and covarinace
        x,p = compute_mean_covariance(x_pred,n)
        x[2] = wraptopi(x[2])
        # step 4 correction with measurement
        for k in range(len(rang[i])):
            x,p = measurement_update(x_pred,landmark[k],dist,x,p,rang[i,k],bearing[i,k])

        x_est[i] = x
        p_est[i] = p

    e_fig = plt.figure()
    ax = e_fig.add_subplot(111)
    ax.plot(x_est[:, 0], x_est[:, 1])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Estimated trajectory')
    plt.show()

    e_fig = plt.figure()
    ax = e_fig.add_subplot(111)
    ax.plot(time_stamp[:], x_est[:, 2])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('theta [rad]')
    ax.set_title('Estimated trajectory')
    plt.show()

if __name__ == '__main__':
    main()
