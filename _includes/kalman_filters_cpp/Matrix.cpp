//
//  Matrix.cpp
//  Kalman_filter
//
//  Created by Guillaume Frèche on 30/09/2019.
//  Copyright © 2019 Guillaume Frèche. All rights reserved.
//

#include "Matrix.hpp"

// Empty constructor
Matrix::Matrix(){
    entries = {};
    nb_row = 0;
    nb_col = 0;
}

// Copy constructor
Matrix::Matrix(const Matrix& mat){
    entries = mat.entries;
    nb_row = mat.nb_row;
    nb_col = mat.nb_col;
}

// Constructor converting a vector of vector into a matrix
Matrix::Matrix(const vector<vector<double>>& array){
    entries = array;
    nb_row = (int) array.size();
    nb_col = (int) array[0].size();
}

// Instantiate an all zero matrix
Matrix Matrix::zeros(const int m, const int n){
    Matrix res;
    if((m>0)&&(n>0)){
        res.entries = vector<vector<double>>(m, vector<double>(n, 0));
        res.nb_row = m;
        res.nb_col = n;
    }
    else{
        throw invalid_argument ("Arguments must be a positive integer.");
    }
    return res;
}

// Instantiate identity matrix
Matrix Matrix::identity(const int m){
    Matrix res;
    if (m>0){
        res = zeros(m,m);
        for (int i=0; i<m; i++){
            res.entries[i][i] = 1;
        }
    }
    else{
        throw invalid_argument ("Argument must be a positive integer.");
    }
    return res;
}

// Display matrix entries
void Matrix::print(){
    for (int i=0; i < nb_row; i++){
        for (int j=0; j < nb_col; j++){
            cout << entries[i][j] << " ";
        }
        cout << endl;
    }
}

Matrix Matrix::operator+(const Matrix& mat){
    Matrix res;
    if ((nb_row!=mat.nb_row)||(nb_col!=mat.nb_col)){
        throw invalid_argument ( "Matrices should have the same dimensions." );
    }
    else{
        res = Matrix::zeros(nb_row, nb_col);
        for (int i=0; i< nb_row; i++){
            for (int j=0; j< nb_col; j++){
                res.entries[i][j] = entries[i][j] + mat.entries[i][j];
            }
        }
    }
    return res;
}

Matrix Matrix::operator-(const Matrix& mat){
    Matrix res;
    if ((nb_row!=mat.nb_row)||(nb_col!=mat.nb_col)){
        throw invalid_argument ( "Matrices should have the same dimensions " );
    }
    else{
        res = Matrix::zeros(nb_row, nb_col);
        for (int i=0; i< nb_row; i++){
            for (int j=0; j< nb_col; j++){
                res.entries[i][j] = entries[i][j] - mat.entries[i][j];
            }
        }
    }
    return res;
}

Matrix Matrix::operator*(const Matrix& mat){
    Matrix res;
    //pair<int, int> curr_shape = get_shape();
    //pair<int, int> mat_shape = mat.get_shape();
    
    if (nb_col!=mat.nb_row){
        throw invalid_argument ( "Matrices should have the same common dimension " );
    }
    else{
        res = Matrix::zeros(nb_row, mat.nb_col);
        for (int i=0; i< nb_row; i++){
            for (int j=0; j< mat.nb_col; j++){
                double temp = 0;
                for (int k=0; k < nb_col; k++){
                    temp += entries[i][k]*mat.entries[k][j];
                }
                res.entries[i][j] = temp;
            }
        }
    }
    return res;
}

Matrix Matrix::operator*(const double val){
    Matrix res = Matrix(entries);
    for (int i=0; i< nb_row; i++){
        for (int j=0; j< nb_col; j++){
            res.entries[i][j] = entries[i][j]*val;
        }
    }
    return res;
}

Matrix Matrix::operator/(const Matrix& mat){
    Matrix res;
    if ((nb_row!=mat.nb_row)||(nb_col!=mat.nb_col)){
        throw invalid_argument ( "Matrices should have the same dimensions " );
    }
    else{
        res = Matrix::zeros(nb_row, nb_col);
        Matrix mat_transp = mat.transpose();
        double det_mat = mat.determinant();
        for (int i=0; i < nb_row; i++){
            for (int j=0; j< nb_col; j++){
                Matrix temp_mat = Matrix(mat_transp);
                temp_mat.set_column(j, get_row(i));
                res.entries[i][j] = temp_mat.determinant()/det_mat;
            }
        }
    }
    return res;
}

Matrix Matrix::operator/(const double val){
    Matrix res = Matrix(entries);
    //pair<int,int> res_shape = res.get_shape();
    for (int i=0; i<nb_row; i++){
        for (int j=0; j<nb_col; j++){
            res.entries[i][j] = entries[i][j] / val;
            //res.set_entry(i, j, res.get_entry(i, j)/val);
        }
    }
    return res;
}

Matrix Matrix::transpose() const{
    Matrix res = Matrix::zeros(nb_col, nb_row);
    for(int i=0; i< nb_col; i++){
        for(int j=0; j<nb_row; j++){
            res.entries[i][j] = entries[j][i];
        }
    }
    return res;
}

Matrix Matrix::inverse(){
    //pair<int,int> curr_shape = get_shape();
    return Matrix::identity(nb_row)/Matrix(entries);
}

double Matrix::determinant() const{
    double res;
    if (nb_row!=nb_col){
        throw invalid_argument ( " Determinant only works for square matrices " );
    }
    else{
        if (nb_row==1){
            res = entries[0][0];
        }
        else{
            res = 0;
            for (int i=0; i<nb_row; i++){
                Matrix sub_mat = sub_matrix(i, 0);
                res += pow(-1, (double) i) * entries[i][0] * (sub_mat.determinant());
            }
        }
    }
    return res;
}

// Convert a 1x1 matrix into a double
double Matrix::to_double(){
    int res;
    if ((nb_row!=1)||(nb_col!=1)){
        throw invalid_argument ( "Matrix must be of shape 1x1" );
    }
    else{
        res = entries[0][0];
    }
    return res;
}

void Matrix::remove_row(int i){
    entries.erase(entries.begin()+i);
}

void Matrix::remove_col(int j){
    for (int i=0; i < entries.size();i++){
        entries[i].erase(entries[i].begin()+j);
    }
}

Matrix Matrix::sub_matrix(int i, int j) const{
    Matrix res = Matrix(entries);
    res.remove_row(i);
    res.remove_col(j);
    res.nb_row--;
    res.nb_col--;
    return res;
}

vector<double> Matrix::get_row(int i){
    return entries[i];
}

vector<double> Matrix::get_column(int j){
    vector<double> res;
    for (int i=0; i< nb_row; i++)
        res.push_back(entries[i][j]);
    return res;
}

void Matrix::set_column(int j, vector<double> col){
    for (int i=0; i<nb_row; i++)
        entries[i][j] = col[i];
}
