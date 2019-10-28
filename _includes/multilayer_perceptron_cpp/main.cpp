//
//  main.cpp
//  Multiple Layer Perceptron
//
//  Created by Guillaume Frèche on 11/10/2019.
//  Copyright © 2019 Guillaume Frèche. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>

using namespace std;

class Layer{
protected:
    int nb_neur;
    vector<double> nu;
    
    Layer(int n) : nb_neur(n), nb_neur_prev(0) {
        nu = vector<double>(nb_neur, 0);
    }

private:
    int nb_neur_prev;
    vector<double> sigma;
    vector<vector<double>> weights;
    vector<double> biases;
    vector<double> dxi_dnu;
    
    virtual void feed_forward(vector<double> x){}
    virtual void compute_dnu_dxi(vector<double> y) {}
    virtual void compute_dnu_dxi(Layer* nextLayer) {}
    virtual void back_propagation(double mu, Layer* prevLayer) {}
    
    static double sigmoid(double x){
        return 1/(1+exp(-x));
    }
    
    static double deriv_sigmoid(double x){
        double sigm = sigmoid(x);
        return sigm*(1-sigm);
    }
    
    friend class NextLayer;
    friend class HiddenLayer;
    friend class OutputLayer;
    friend class Perceptron;
};

class InputLayer : public Layer{
private:
    InputLayer(int n) : Layer(n){}
    
    void feed_forward(vector<double> x){
        (nb_neur != x.size()) ? throw invalid_argument("Input layer and vector must be the same size.") : nu = x;
    }
    
    friend class Perceptron;
};

class NextLayer : public Layer{
protected:
    NextLayer(int n, int nprev) : Layer(n){
        sigma = vector<double>(nb_neur, 0);
        dxi_dnu = vector<double>(nb_neur, 0);
        nb_neur_prev = nprev;
        
        random_device rd;
        mt19937 generator(rd());
        normal_distribution<double> dist(0,1);
        for (int i=0; i<nb_neur; i++){
            weights.push_back(vector<double>());
            for(int j=0; j<nb_neur_prev; j++){
                weights[i].push_back(dist(generator));
            }
            biases.push_back(dist(generator));
        }
    }
    
    NextLayer(int n, int nprev, vector<vector<double>> w, vector<double> b) : Layer(n){
        sigma = vector<double>(nb_neur, 0);
        dxi_dnu = vector<double>(nb_neur, 0);
        nb_neur_prev = nprev;
        
        weights = w;
        biases = b;
    }

private:
    void feed_forward(vector<double> x){
        for (int i=0; i<nb_neur; i++){
            double temp = 0;
            for (int j = 0; j<nb_neur_prev; j++)
                temp += weights[i][j]*x[j];
            sigma[i] = temp + biases[i];
            nu[i] = sigmoid(sigma[i]);
        }
    }

    void back_propagation(double mu, Layer* prevLayer) {
        for (int i=0; i<nb_neur; i++){
            double dnu_db = deriv_sigmoid(sigma[i]);
            biases[i] -= mu * dxi_dnu[i] * dnu_db;
            for (int j = 0; j<nb_neur_prev; j++)
                weights[i][j] -= mu * dxi_dnu[i] * prevLayer->nu[j] * dnu_db;
        }
    }
};

class HiddenLayer : public NextLayer{
private:
    HiddenLayer(int n, int nprev) : NextLayer(n, nprev){}
    HiddenLayer(int n, int nprev, vector<vector<double>> w, vector<double> b) : NextLayer(n, nprev, w, b){}

    void compute_dnu_dxi(Layer* nextLayer){
        for (int i=0; i<nb_neur; i++){
            double sum = 0;
            for (int h=0; h<nextLayer->nb_neur; h++){
                sum += nextLayer->dxi_dnu[h] * nextLayer->weights[h][i] * deriv_sigmoid(nextLayer->sigma[h]);
            }
            dxi_dnu[i] = sum;
        }
    }
    
    friend class Perceptron;
};

class OutputLayer : public NextLayer{
private:
    OutputLayer(int n, int nprev) : NextLayer(n, nprev){}
    OutputLayer(int n, int nprev, vector<vector<double>> w, vector<double> b) : NextLayer(n, nprev, w, b){}

    void compute_dnu_dxi(vector<double> y) {
        for (int i=0; i< nb_neur; i++)
            dxi_dnu[i] = 2 * (nu[i] - y[i]);
    }
    
    friend class Perceptron;
};

class Perceptron{
public:
    Perceptron(vector<int> nbs){
        nb_layers = (int) nbs.size();
        if (nb_layers<2)
            throw invalid_argument("Multilayer perceptron must have at least 2 layers.");
        else{
            layers.push_back(new InputLayer(nbs[0]));
            for (int i=1; i<nbs.size()-1; i++){
                layers.push_back(new HiddenLayer(nbs[i],nbs[i-1]));
            }
            layers.push_back(new OutputLayer(*(nbs.end()-1),*(nbs.end()-2)));
        }
    }
    
    Perceptron(string filename){
        ifstream filesave;
        filesave.open(filename);
        filesave >> nb_layers;
        if (nb_layers<2)
            throw invalid_argument("Multilayer perceptron must have at least 2 layers.");
        else{
            int nb;
            vector<int> nbs;
            for (int h=0; h<nb_layers; h++){
                filesave >> nb;
                nbs.push_back(nb);
            }
            layers.push_back(new InputLayer(nbs[0]));
            
            double val;
            for (int i=1; i<nbs.size()-1; i++){
                vector<vector<double>> weights;
                vector<double> biases;
                for (int j=0; j<nbs[i]; j++){
                    weights.push_back(vector<double>());
                    for (int k=0; k<nbs[i-1]; k++){
                        filesave >> val;
                        weights[j].push_back(val);
                    }
                }
                for (int j=0; j<nbs[i]; j++){
                    filesave >> val;
                    biases.push_back(val);
                }
                layers.push_back(new HiddenLayer(nbs[i],nbs[i-1], weights, biases));
            }
            vector<vector<double>> weights;
            vector<double> biases;
            for (int j=0; j<nbs[nbs.size()-1]; j++){
                weights.push_back(vector<double>());
                for (int k=0; k<nbs[nbs.size()-2]; k++){
                    filesave >> val;
                    weights[j].push_back(val);
                }
            }
            for (int j=0; j<nbs[nbs.size()-1]; j++){
                filesave >> val;
                biases.push_back(val);
            }
            layers.push_back(new OutputLayer(nbs[nbs.size()-1],nbs[nbs.size()-2], weights, biases));
        }
        filesave.close();
    }
    
    void feed_forward(vector<double> x){
        layers[0]->feed_forward(x);
        for (int i=1; i<layers.size(); i++){
            layers[i]->feed_forward(layers[i-1]->nu);
        }
    }
    
    double compute_error(vector<double> y){
        double res = 0;
        vector<Layer*>::iterator it = layers.end()-1;
        for (int i=0; i<(*it)->nu.size(); i++){
            res += pow((*it)->nu[i]-y[i],2);
        }
        return res;
    }
    
    void saveToFile(string filename){
        int prec = 12;
        ofstream filesave;
        filesave.open(filename);
        // Save the number of layers
        filesave << nb_layers;
        // Save the number of neurons in every layer
        for (int i=0; i<nb_layers; i++){
            filesave << " " << layers[i]->nb_neur ;
        }
        filesave << endl;
        // Save weights and biases
        for (int h = 1; h<nb_layers; h++){
            for(int i=0; i<layers[h]->nb_neur; i++){
                filesave << setprecision(prec) << layers[h]->weights[i][0];
                for(int j=1; j<layers[h]->nb_neur_prev; j++){
                    filesave << " " << setprecision(prec) << layers[h]->weights[i][j];
                }
                filesave << endl;
            }
            filesave << setprecision(prec) << layers[h]->biases[0];
            for(int i=1; i<layers[h]->nb_neur; i++){
                filesave << " " << setprecision(prec) << layers[h]->biases[i];
            }
            filesave << endl;
        }
        filesave.close();
    }
    
    void train(string test_filename, int nb_iter){
        vector<vector<double>> test_input;
        vector<vector<double>> test_output;
        
        int nb_ex, nb_input, nb_output;
        
        ifstream filetest;
        filetest.open(test_filename);
        
        filetest >> nb_ex >> nb_input >> nb_output;
        
        double val;
        for (int i=0; i<nb_ex; i++){
            test_input.push_back(vector<double>());
            for (int j=0; j < nb_input; j++){
                filetest >> val;
                test_input[i].push_back(val);
            }
            test_output.push_back(vector<double>());
            for (int j=0; j < nb_output; j++){
                filetest >> val;
                test_output[i].push_back(val);
            }
        }
        
        filetest.close();
        
        train(test_input, test_output, nb_iter, 1);
    }
    
    void feed_forward(string res_filename, vector<vector<double>> test_input){
        ofstream filesave;
        filesave.open(res_filename);
        
        filesave << test_input.size() << " " << test_input[0].size() << " " << layers[nb_layers-1]->nb_neur << endl;
        
        for (int i=0; i<test_input.size(); i++){
            feed_forward(test_input[i]);
            for (int j=0; j<test_input[i].size(); j++){
                filesave << test_input[i][j] << " ";
            }
            for (int j=0; j<layers[nb_layers-1]->nb_neur; j++){
                filesave << layers[nb_layers-1]->nu[j] << " ";
            }
            filesave << endl;
        }
        
        filesave.close();
    }
    
    void feed_forward(string test_filename, string res_filename){
        int nb_tests, size_input;
        ifstream test_file;
        test_file.open(test_filename);
        test_file >> nb_tests >> size_input;
        double val;
        vector<vector<double>> test_input;
        for (int i=0; i<nb_tests; i++){
            test_input.push_back(vector<double>());
            for (int j=0; j<size_input; j++){
                test_file >> val;
                test_input[i].push_back(val);
            }
        }
        test_file.close();
        
        feed_forward(res_filename, test_input);
    }
    
private:
    int nb_layers;
    vector<Layer*> layers;
    
    void back_propagation(vector<double> y, double mu){
        vector<Layer*>::iterator it = layers.end()-1;
        (*it)->compute_dnu_dxi(y);
        (*it)->back_propagation(mu, *(it-1));
        for (int i=(int) layers.size()-2; i>=1; i--){
            layers[i]->compute_dnu_dxi(layers[i+1]);
            layers[i]->back_propagation(mu, layers[i-1]);
        }
    }
    
    void train(vector<vector<double>> test_input, vector<vector<double>> test_output, int nb_iter, double mu){
        int nb_ex = (int) test_input.size();
        for (int iter = 0; iter<nb_iter; iter++){
            for (int i=0; i<nb_ex; i++){
                feed_forward(test_input[i]);
                back_propagation(test_output[i], mu);
            }
        }
    }
};

int main(int argc, const char * argv[]) {
    
    if(argc<2){
        cout << "This is not how this exec is supposed to be called..." << endl;
    }
    else{
        string command(argv[1]);
        if (command=="train"){
            string train_filename(argv[2]);
            string save_filename(argv[3]);
            
            vector<int> nbs;
            for (int i=4; i<argc-1; i++){
                nbs.push_back(atoi(argv[i]));
            }
            int nb_iter = atoi(argv[argc-1]);
            Perceptron MLP(nbs);
            MLP.train(train_filename, nb_iter);
            MLP.saveToFile(save_filename);
        }
        else if (command=="test"){
            cout << "I understand you want to test a neural network" << endl;
            string save_filename(argv[2]);
            string test_filename(argv[3]);
            string res_filename(argv[4]);
            Perceptron MLP(save_filename);
            MLP.feed_forward(test_filename, res_filename);
        }
        else{
            cout << "I do not recognize this command" << endl;
        }
    }
    
    return 0;
}
