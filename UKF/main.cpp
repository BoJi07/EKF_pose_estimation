#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <math.h>
#include "matplotlibcpp.h"

using std::vector;
using std::string;
using std::ifstream;
using std::stringstream;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::cout;
using std::cerr;
using std::endl;
namespace plt = matplotlibcpp;

#define n 3  // number of state
#define n_aug 5  //number of augmented state
#define m  2


vector<double> readData1D(string path){
    ifstream input;
    double val;
    vector<double> data;
    input.open(path);
    if(!input){
        cerr<<"can't open the file !";
        exit(1);
    }
    while(input>>val){
        data.push_back(val);
    }   
    input.close();
    return data;
}

vector<vector<double>> readData2D(string path){
    ifstream input;
    double val;
    string line;
    vector<vector<double>> data;
    input.open(path);
    if(!input){
        cerr<<"can't open the file!";
        exit(1);
    }
    while(std::getline(input,line)){
        stringstream ss(line);
        vector<double> temp;
        while(ss>>val){
            temp.push_back(val);
        }
        data.push_back(temp);
    }
    input.close();
    return data;
}
// keep yaw between -pi to pi
double wraptopi(double x){
    while(x>M_PI){
        x -= 2*M_PI;
    }
    while(x<-M_PI){
        x += 2*M_PI;
    }
    return x;
}

MatrixXd generateSigmaPoints(VectorXd x, MatrixXd p, double std_v, double std_w){
    MatrixXd x_sig = MatrixXd::Zero(n_aug , 2*n_aug+1);
    MatrixXd p_aug = MatrixXd::Zero(n_aug,n_aug);
    p_aug.topLeftCorner(n,n) = p;
    p_aug(n,n) = std_v*std_v;
    p_aug(n+1,n+1) = std_w*std_w;
    MatrixXd L = p_aug.llt().matrixL();
    x_sig.col(0) = x;
    int lambda = 3-n_aug;
    double alpha = sqrt(lambda+n_aug);
    for(int i = 0; i<n_aug; i++){
        x_sig.col(i+1) = x + (alpha*L.col(i));
        x_sig(2,i+1) = wraptopi(x_sig(2,i+1));
        x_sig.col(i+1+n_aug) = x- (alpha*L.col(i));
        x_sig(2,i+1+n_aug) = wraptopi(x_sig(2,i+1+n_aug));
    }
    return x_sig;
}

MatrixXd prediction(MatrixXd x_sig,double dt,double v, double w){
    MatrixXd x_pred = MatrixXd(n,n_aug*2+1);
    for(int i = 0;i<2*n_aug+1; i++){
        double p_x = x_sig(0,i);
        double p_y = x_sig(1,i);
        double p_yaw = x_sig(2,i);

        double std_a = x_sig(3,i);
        double std_w = x_sig(4,i);

        double p_x_pred = p_x + dt*cos(p_yaw)*v + 0.5* std_a*dt*dt*cos(p_yaw);
        double p_y_pred = p_y + dt*sin(p_yaw)*v + 0.5* std_a*dt*dt*sin(p_yaw);
        double p_yaw_pred = p_yaw + dt*w + 0.5*std_w*dt*dt;

        x_pred(0,i) = p_x_pred;
        x_pred(1,i) = p_y_pred;
        x_pred(2,i) = p_yaw_pred;
        x_pred(2,i) = wraptopi(x_pred(2,i));
    }
    return x_pred;
}

VectorXd computeMean(MatrixXd x_pred){
    VectorXd weights = VectorXd::Zero(2*n_aug+1);
    int lambda = 3-n_aug;
    weights(0) = lambda/(n_aug+lambda);
    for(int i = 1;i<2*n_aug+1; i++){
        weights(i) = 0.5/(n_aug+lambda);
    }
    VectorXd x_mean = VectorXd::Zero(n);
    for(int i = 0; i<2*n_aug+1; i++){
        x_mean  = x_mean + (weights(i) * x_pred.col(i));
    }
    x_mean(2) = wraptopi(x_mean(2));
    return x_mean;
}

MatrixXd computeCovariance(MatrixXd x_pred, VectorXd x_mean){
    MatrixXd covariance = MatrixXd::Zero(n,n);
    int lambda = 3-n_aug;
    VectorXd weights = VectorXd::Zero(2*n_aug+1);
    weights(0) = lambda/(lambda+n_aug);
    for(int i = 1;i<2*n_aug+1; i++){
        weights(i) = 0.5/(n_aug+lambda);
    }

    for(int i = 0; i<2*n_aug+1; i++){
        VectorXd x_diff = x_pred.col(i)-x_mean;
        x_diff(2) = wraptopi(x_diff(2));
        covariance = covariance + (weights(i)*x_diff*x_diff.transpose());
    }
    
    return covariance;
}

void measurementUpdate(MatrixXd& p, VectorXd& x, MatrixXd x_pred,VectorXd x_mean,vector<double> landmark, double range, double bearing, double dist,double var_r,double var_b){
    double xl = landmark[0];
    double yl = landmark[1];
    MatrixXd z_pred = MatrixXd::Zero(m,2*n_aug+1);
    for(int i = 0; i<2*n_aug+1; i++){
        double dx = xl-x_pred(0,i)-dist*cos(x_pred(2,i));
        double dy = yl-x_pred(1,i)-dist*sin(x_pred(2,i));
        z_pred(0,i) = sqrt(dx*dx+dy*dy);
        z_pred(1,i) = atan2(dy,dx)-x_pred(2,i);
        z_pred(1,i) =  wraptopi(z_pred(1,i));
    }
    int lambda = 3-n_aug;
    VectorXd weights = VectorXd::Zero(n_aug*2+1);
    weights(0) = lambda/(lambda+n_aug);
    for(int i = 1; i<2*n_aug+1; i++){
        weights(i) = 0.5/(lambda+n_aug);
    }
    VectorXd z_mean = VectorXd::Zero(m);
    MatrixXd S = MatrixXd::Zero(m,m);
    MatrixXd R = MatrixXd::Zero(m,m);
    MatrixXd T = MatrixXd::Zero(n,m);
    VectorXd z_mea = VectorXd::Zero(m);
    R(0,0) = var_r;
    R(1,1) = var_b;
    z_mea(0) = range;
    z_mea(1) = bearing;
    z_mea(1) = wraptopi(z_mea(1));
    for(int i = 0; i<2*n_aug+1; i++){
        z_mean = z_mean + (weights(i)*z_pred.col(i));
    }
    for(int i = 0; i<2*n_aug+1; i++){
        VectorXd z_diff = z_pred.col(i)-z_mean;
        z_diff(1) = wraptopi(z_diff(1));
        S = S + (weights(i)*z_diff*z_diff.transpose());
    }
    S = S + R;
    for(int i = 0; i<2*n_aug+1; i++){
        VectorXd z_diff = z_pred.col(i)-z_mean;
        z_diff(1) = wraptopi(z_diff(1));
        VectorXd x_diff = x_pred.col(i)-x_mean;
        x_diff(2) = wraptopi(x_diff(2));
        T = T + (weights(i)*x_diff*z_diff.transpose());
    }
    MatrixXd K = T*S.inverse();
    VectorXd y = z_mea -z_mean;
    y(1) = wraptopi(y(1));
    x = x + (K*y);
    x(2) = wraptopi(x(2));
    p = p - (K*S*K.transpose());
    p = 0.5*(p+p.transpose());
}

int main(){
    // path for the input data
    string path_time = "data_ukf/time_stamp.txt";
    string path_init = "data_ukf/init_pose.txt";
    string path_range = "data_ukf/range.txt";
    string path_bearing = "data_ukf/bearing.txt";
    string path_vel = "data_ukf/velocity.txt";
    string path_rot = "data_ukf/rotation_speed.txt";
    string path_landmark = "data_ukf/landmark.txt";
    string path_dist = "data_ukf/dist.txt";
    //retrieve the data
    //time stamp
    vector<double> time_stamp = readData1D(path_time);
    //init pose for the robot
    vector<double> init_pose = readData1D(path_init);
    //range reading from laser scan
    vector<vector<double>> range = readData2D(path_range);
    //bearing reading from laser scan
    vector<vector<double>> bearing = readData2D(path_bearing);
    //input velocity for the robot
    vector<double> vel = readData1D(path_vel);
    //input rotation speed for the robot
    vector<double> rot = readData1D(path_rot);
    //landmark pos in the known map (fixed)
    vector<vector<double>> landmark = readData2D(path_landmark);
    //dist between laser corrdinate frame to the center of the robot
    vector<double> dist = readData1D(path_dist);
    double distance  = dist[0];

    //variance and uncertainty
    //input noise
    double std_a = 0.1;
    double std_w = 0.1;
    //measurement noise
    double var_r = 0.01;
    double var_b = 10;

    //init_condition   
    VectorXd x = VectorXd(3);
    x(0) = init_pose[0];
    x(1) = init_pose[1];
    x(2) = init_pose[2];
    //covariance matrix
    MatrixXd p = MatrixXd::Zero(n,n);

    for(int i = 1; i<time_stamp.size(); i++){
        double dt = time_stamp[i]-time_stamp[i-1];
        //step 1 generate sigma points
        VectorXd x_aug = VectorXd::Zero(5);
        x_aug.head(n) = x;
        MatrixXd x_sig = generateSigmaPoints(x_aug,p,std_a,std_w);
        //step 2 predict motion
        MatrixXd x_pred = prediction(x_sig,dt,vel[i-1],rot[i-1]);
        //step3 compute new state and covariance
        x = computeMean(x_pred);
        
        p = computeCovariance(x_pred,x);
        //step4 measurement correction
        
        for(int j = 0; j<landmark.size(); j++){
            measurementUpdate(p,x,x_pred,x,landmark[j],range[i][j],bearing[i][j],distance,var_r,var_b);
        }
        
    }

    //

}