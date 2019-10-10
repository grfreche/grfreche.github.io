//
//  Matrix.hpp
//  Kalman_filter
//
//  Created by Guillaume Frèche on 30/09/2019.
//  Copyright © 2019 Guillaume Frèche. All rights reserved.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Matrix{
private:
    vector<vector<double>> entries;
    int nb_row;
    int nb_col;
    
    void remove_row(int i);
    void remove_col(int j);
    // Sub matrix obtained by removing row i and column j
    Matrix sub_matrix(int i, int j) const;
    
public:
    // Empty constructor
    Matrix();
    // Copy constructor
    Matrix(const Matrix& mat);
    // Constructor converting a vector of vector into a matrix
    Matrix(const vector<vector<double>>& array);
    
    // Instantiate an all zero matrix
    static Matrix zeros(const int m, const int n);
    // Instantiate identity matrix
    static Matrix identity(const int m);
    
    // Display matrix entries
    void print();
    
    // Operators overload
    Matrix operator+(const Matrix& mat);
    Matrix operator-(const Matrix& mat);
    Matrix operator*(const Matrix& mat);
    Matrix operator*(const double val);
    Matrix operator/(const Matrix& mat);
    Matrix operator/(const double val);
    
    // Transpose matrix
    Matrix transpose() const;
    // Inverse matrix
    Matrix inverse();
    
    // Determinant
    double determinant() const;
    
    // Convert a 1x1 matrix into a double
    double to_double();

    vector<double> get_row(int i);
    vector<double> get_column(int j);
    void set_column(int j, vector<double> col);
};

#endif /* Matrix_hpp */
