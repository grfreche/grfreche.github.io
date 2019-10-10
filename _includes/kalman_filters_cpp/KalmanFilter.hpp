//
//  KalmanFilter.hpp
//  Kalman_filter
//
//  Created by Guillaume Frèche on 30/09/2019.
//  Copyright © 2019 Guillaume Frèche. All rights reserved.
//

#ifndef KalmanFilter_hpp
#define KalmanFilter_hpp

#include <iostream>
#include "Matrix.hpp"

class KalmanFilter{
private:
    int state_dim;
    int obs_dim;
    Matrix F;
    Matrix Rv;
    Matrix H;
    Matrix Ru;
    
public:
    // Kalman filter constructor
    KalmanFilter(int m, int n, Matrix matF, Matrix matRv, Matrix matH, Matrix matRu) :
    state_dim(m), obs_dim(n), F(matF), Rv(matRv), H(matH), Ru(matRu) {}
    // Kalman filter state estimator
    Matrix estimate_state(vector<double> observ);
};

#endif /* KalmanFilter_hpp */
