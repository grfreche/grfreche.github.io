//
//  KalmanFilter.cpp
//  Kalman_filter
//
//  Created by Guillaume Frèche on 30/09/2019.
//  Copyright © 2019 Guillaume Frèche. All rights reserved.
//

#include "KalmanFilter.hpp"

Matrix KalmanFilter::estimate_state(vector<double> observ){
    Matrix estim_state = Matrix::zeros(state_dim, (int) observ.size());
    Matrix prior_innov_cov;
    Matrix posterior_innov_cov = Matrix::zeros(state_dim, state_dim);
    Matrix kalman_gain;
    Matrix prior_estim_state;
    Matrix posterior_estim_state;
    
    
    for (int k=1; k<observ.size(); k++){
        // Prior innovation covariance matrix update
        prior_innov_cov = (F*posterior_innov_cov*(F.transpose())) + Rv;
        // Kalman gain update
        double denom = ((H*prior_innov_cov*(H.transpose()))+Ru).to_double();
        kalman_gain = (prior_innov_cov*(H.transpose()))/denom;
        // posterior innovation covariance matrix update
        posterior_innov_cov = (Matrix::identity(state_dim)-(kalman_gain*H))*prior_innov_cov;
        
        // Previous posterior state estimate
        vector<vector<double>> temp_state;
        temp_state.push_back(estim_state.get_column(k));
        posterior_estim_state = Matrix(temp_state).transpose();
        // New prior state estimate
        prior_estim_state = F*posterior_estim_state;
        // New posterior state estimate
        posterior_estim_state = prior_estim_state + (kalman_gain*(observ[k]-((H*prior_estim_state).to_double())));
        
        // Store the new state estimate
        estim_state.set_column(k, posterior_estim_state.get_column(0));
    }
    
    return estim_state;
}
