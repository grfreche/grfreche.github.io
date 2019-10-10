---
layout: default
title: Machine learning
permalink: machine_learning/cpp_kalman/
---

# C++ files for Kalman filters

## main.cpp

```cpp
//
//  main.cpp
//  Kalman_filter
//
//  Created by Guillaume Frèche on 30/09/2019.
//  Copyright © 2019 Guillaume Frèche. All rights reserved.
//

#include <iostream>
//#include <exception>
#include <random>
#include <chrono>

#include "KalmanFilter.hpp"

int main(int argc, const char * argv[]) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    
    // Time step
    double dt = 1.;
    
    // Kalman filter's model matrices
    // Matrix F
    vector<vector<double>> matF = {{1, dt}, {0, 1}};
    // Matrix Rv
    double sigma_a = 0.2;
    vector<vector<double>> matRv = {{sigma_a*sigma_a*pow(dt, 4.)/4, sigma_a*sigma_a*pow(dt, 3.)/2},
        {sigma_a*sigma_a*pow(dt, 3.)/2, dt*dt*sigma_a*sigma_a}};
    // Matrix H
    vector<vector<double>> matH = {{1, 0}};
    // Matrix Ru
    double sigma_noise = 30;
    vector<vector<double>> matRu = {{sigma_noise*sigma_noise}};
    
    // Construct Kalman filter
    KalmanFilter filter = KalmanFilter(2,1, Matrix(matF), Matrix(matRv), Matrix(matH), Matrix(matRu));
    
    int niter = 500;
    
    // Generate random acceleration
    random_device rd;
    mt19937 generator(rd());
    normal_distribution<double> dist_accel(0, sigma_a);
    vector<double> accel;
    for (int i=0; i<niter; i++)
        accel.push_back(dist_accel(generator));
    
    // Generate true position and speed
    vector<double> true_pos = {0};
    double sigma_speed = 2;
    normal_distribution<double> dist_speed(0, sigma_speed);
    vector<double> true_speed = {dist_speed(generator)};
    for (int i=0; i<niter; i++){
        true_pos.push_back((*true_pos.end())+((*true_speed.end())*dt)+(accel[i]*dt*dt/2));
        true_speed.push_back((*true_speed.end())+(accel[i]*dt));
    }
    
    // Generate noisy observations
    normal_distribution<double> dist_obs(0, sigma_noise);
    vector<double> obs_pos;
    for (int i=0; i<true_pos.size(); i++){
        obs_pos.push_back(true_pos[i]+dist_obs(generator));
    }
    
    // Apply Kalman filter
    Matrix estim_state = filter.estimate_state(obs_pos);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Program duration: " << ((double) duration.count())/double(1000000) << " sec " << endl;
    
    return 0;
}
```
## Matrix.hpp

```cpp
{% include kalman_filters_cpp/Matrix.hpp %}
```

## Matrix.cpp

```cpp
{% include kalman_filters_cpp/Matrix.cpp %}
```

## KalmanFilter.hpp

```cpp
{% include kalman_filters_cpp/KalmanFilter.hpp %}
```

## KalmanFilter.cpp

```cpp
{% include kalman_filters_cpp/KalmanFilter.cpp %}
```
