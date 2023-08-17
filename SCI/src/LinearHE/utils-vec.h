
#ifndef UTILS_VEC_H__
#define UTILS_VEC_H__

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <iostream>
#include <math.h>
#include <random>
#include <string>
#include <vector>
#include <sstream>

using namespace std;
using namespace chrono;

// Stopwatch for timing operations in microseconds.
class StopWatch{
    public:
        vector<double> lapTimes;

        // resets start time of the stop watch.
        void reset(){start_ = high_resolution_clock::now(); lap_ = start_;
            lapTimes = vector<double>(10);
        }
        
        // Returns time since started in microseconds.
        double time(){
            duration<double, std::micro> dur = stop_() - start_;
            return dur.count();
        }

        // Returns time since previous call in microseconds.
        double lap(){
            duration<double, std::micro> dur = stop_() - lap_;
            lap_ = stop_();
            return dur.count();
        }

        // increments time at index idx in the times vector.
        void lap(int idx){
            duration<double, std::micro> dur = stop_() - lap_;
            lap_ = stop_();
            lapTimes.at(idx) += dur.count();
        }

        StopWatch(){reset();}

    private:

        high_resolution_clock::time_point start_;

        high_resolution_clock::time_point lap_;

        high_resolution_clock::time_point stop_() {
            return high_resolution_clock::now();
        }
};

struct matrixPT{
    vector<double> data;
    bool isTransperent = true;
    int nrows;
    int ncols;
};

struct FCkerMasks{
    vector<matrixPT> data;
    bool preRotated;
    vector<int> fcParms;
};

void save_str(string my_str, string filename);

vector<vector<double>> gen2D_UID(int dim0, int dim1, int min = 0, int max = 8);
vector<vector<double>> gen2D_URD(int dim0, int dim1, double min = 0, double max = 8);
vector<vector<double>> gen2DinRange(int dim0, int dim1);
vector<vector<int>> zeros2Dint(int dim0, int dim1);
vector<vector<double>> zeros2D(int dim0, int dim1);
int genScalar_UID(int min = 0, int max = 8);
void print1D(vector<int> inp1D, int num_digits = 3, int max_row = 0);
void print1D(vector<uint64_t> inp1D, int num_digits = 3, int max_row = 0);
void print1D(vector<double> inp1D, int num_digits = 6, int num_decimal_digits = 2, int max_row = 0);
void print2D(vector<vector<int>> inp2D, int num_digits = 3, int max_row = 0);
void print2D(vector<vector<uint64_t>> inp2D, int num_digits = 3, int max_row = 0);
void print2D(vector<vector<double>> inp2D, int num_digits = 6, int num_decimal_digits = 2, int max_row = 0);
void print3D(vector<vector<vector<int>>> inp3D, int num_digits = 3, int max_row = 0);
void print3D(vector<vector<vector<uint64_t>>> inp3D, int num_digits = 3, int max_row = 0);
void print3D(vector<vector<vector<double>>> inp3D, int num_digits = 6, int num_decimal_digits = 2, int max_row = 0);
bool warnNotEq(int A, int B, string tag);
bool sizeCheck2D(vector<vector<int>> A, vector<vector<int>> B, string tag = "sizeCheck2D");
bool sizeCheck2D(vector<vector<double>> A, vector<vector<double>> B, string tag = "sizeCheck2D");
vector<vector<int>> matMult(vector<vector<int>> A, vector<vector<int>> B);
vector<vector<double>> matMult(vector<vector<double>> A, vector<vector<double>> B);
vector<int> matMult(vector<vector<int>> A, vector<int> B);
vector<double> matMult(vector<vector<double>> A, vector<double> B);
vector<vector<int>> add2D(vector<vector<int>> A, vector<vector<int>> B);
vector<vector<double>> add2D(vector<vector<double>> A, vector<vector<double>> B);
vector<vector<int>> sub2D(vector<vector<int>> A, vector<vector<int>> B);
vector<vector<double>> sub2D(vector<vector<double>> A, vector<vector<double>> B);
vector<vector<int>> mul2D(vector<vector<int>> A, vector<vector<int>> B);
vector<vector<double>> mul2D(vector<vector<double>> A, vector<vector<double>> B);
vector<vector<double>> scale2D(vector<vector<double>> A, double scaling_factor, bool to_int = false);
int sum2D(vector<vector<int>> inp2D);
uint64_t sum2D(vector<vector<uint64_t>> inp2D);
double sum2D(vector<vector<double>> inp2D);
bool isScalar(vector<int> inp1D, int scalar, double threshold = 1e-8);
bool isScalar(vector<double> inp1D, double scalar, double threshold = 1e-8);
double MAE2D(vector<vector<int>> A, vector<vector<int>> B, bool norm = false, double threshold = 0);
double MAE2D(vector<vector<double>> A, vector<vector<double>> B, bool norm = false, double threshold = 0);
double maxAE1D(vector<double> A, vector<double> B, double scale = 1, bool round_to_int = true);
double max1D(vector<double> A, bool abs_max = false);
double min1D(vector<double> A, bool abs_min = false);
vector<vector<int>> pad2D(vector<vector<int>> inp2D, int padT, int padB, int padL, int padR);
vector<vector<double>> pad2D(vector<vector<double>> inp2D, int padT, int padB, int padL, int padR);
vector<vector<int>> unpad2D(vector<vector<int>> inp2D, int unpadT, int unpadB, int unpadL, int unpadR);
vector<vector<double>> unpad2D(vector<vector<double>> inp2D, int unpadT, int unpadB, int unpadL, int unpadR);
int ceil2Pow(int val);
vector<vector<int>> pad2pow2D(vector<vector<int>> inp2D);
vector<vector<double>> pad2pow2D(vector<vector<double>> inp2D);
vector<vector<double>> repMat2D(vector<vector<double>> inp2D, int nReps);
vector<vector<int>> repMat2D(vector<vector<int>> inp2D, int nReps);
vector<vector<uint64_t>> rotate2D(vector<vector<uint64_t>> inp2D, int nRotRows, int nRotCols);
vector<vector<double>> rotate2D(vector<vector<double>> inp2D, int nRotRows, int nRotCols);
vector<int> slice1D(vector<int>& arr, int X, int Y);
vector<uint64_t> slice1D(vector<uint64_t>& arr, int X, int Y);
vector<double> slice1D(vector<double>& arr, int X, int Y);
vector<int> rowEncode2D(vector<vector<int>> inp2D);
vector<uint64_t> rowEncode2D(vector<vector<uint64_t>> inp2D);
vector<double> rowEncode2D(vector<vector<double>> inp2D);
uint64_t float_to_fixed(double flt_val, double scale = 1);
double fixed_to_float(uint64_t flt_val, double scale = 1);

matrixPT matPTEncode(vector<vector<double>> inp2D);
vector<vector<double>> matPTDecode(matrixPT mat);
matrixPT addPT(matrixPT Apt, matrixPT Bpt);
matrixPT subPT(matrixPT Apt, matrixPT Bpt);
matrixPT mulPT(matrixPT Apt, matrixPT Bpt);
double sumPT(matrixPT Apt);
matrixPT rotatePT(matrixPT mat, int rotVal);
matrixPT rotateRowsPT(matrixPT mat, int rotVal);
FCkerMasks encodeFCkernel(vector<vector<double>> A, vector<int> fcParms, bool preRotate = true);
matrixPT encodeFCinput(vector<vector<double>> B, vector<int> fcParms);
matrixPT matMultPTvanilla(FCkerMasks AmaskPTs, matrixPT BexpPT);
matrixPT matMultPToptim(FCkerMasks AmaskPTs, matrixPT BexpPT);
matrixPT matMultPTpart1(FCkerMasks AmaskPTs, matrixPT BexpPT);
matrixPT matMultPTpart2(matrixPT AB2pt1, vector<int> fcParms);
std::vector<int> extractNumbers(const std::string& input);

#endif // UTILS_VEC_H__