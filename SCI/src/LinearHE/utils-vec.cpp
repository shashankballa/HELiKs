#include "utils-vec.h"

using namespace std;

//Saves a string to a new file
void save_str(string my_str, string filename){
    ofstream myfile;
    myfile.open(filename);
    myfile << my_str;
    myfile.close();
}

//Samples (dim0, dim1) 2D vector from a Uniform Integer Distribution between (min, max).
vector<vector<double>> gen2D_UID(int dim0, int dim1, int min, int max){
    random_device rnd_device;
    mt19937 mersenne_engine {rnd_device()};
    uniform_int_distribution<int> dist {min, max};
    
    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine);};

    vector<vector<double>> out;
    for(int i = 0; i < dim0; i++){
        vector<double> row (dim1);
        generate(begin(row), end(row), gen);
        out.push_back(row);
    }
    return out;
}

//Samples (dim0, dim1) 2D vector from a Uniform Real Distribution between (min, max).
vector<vector<double>> gen2D_URD(int dim0, int dim1, double min, double max){
    random_device rnd_device;
    mt19937 mersenne_engine {rnd_device()};
    uniform_real_distribution<double> dist {min, max};
    
    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine);};

    vector<vector<double>> out;
    for(int i = 0; i < dim0; i++){
        vector<double> row (dim1);
        generate(begin(row), end(row), gen);
        out.push_back(row);
    }
    return out;
}

//Samples (dim0, dim1) 2D vector with elements in row-major ascending order.
vector<vector<double>> gen2DinRange(int dim0, int dim1){
    vector<vector<double>> out;
    for(int i = 0; i < dim0; i++){ 
        vector<double> row;
        for(int j = 0; j<dim1; j++) row.push_back(i*dim1+j+1);
        out.push_back(row);
    }
    return out;
}

//Generates a 2D vector of zeros.
vector<vector<int>> zeros2Dint(int dim0, int dim1){
    vector<vector<int>> out(dim0, vector<int>(dim1));
    return out;
}

//Generates a 2D vector of zeros.
vector<vector<double>> zeros2D(int dim0, int dim1){
    vector<vector<double>> out(dim0, vector<double>(dim1));
    return out;
}

//Samples random integer from a Uniform Integer Distribution between (min, max).
int genScalar_UID(int min, int max){
    random_device rnd_device;
    mt19937 mersenne_engine {rnd_device()};
    uniform_int_distribution<int> dist {min, max};
    
    return dist(mersenne_engine);
}

//Prints 1D vector.
void print1D(vector<int> inp1D, int num_digits, int max_row){;
    if (max_row <= 0) max_row += inp1D.size();
    for(int i{}; i < max_row; i++)
    printf("%*d ", num_digits, inp1D.at(i));
    cout << endl;
}

//Prints 1D vector.
void print1D(vector<uint64_t> inp1D, int num_digits, int max_row){
    if (max_row <= 0) max_row += inp1D.size();
    for(int i{}; i < max_row; i++)
    printf("%.*e ", num_digits, (double) inp1D.at(i));
    cout << endl;
}

//Prints 1D vector.
void print1D(vector<double> inp1D, int num_digits, int num_decimal_digits, int max_row){
    if (max_row <= 0) max_row += inp1D.size();
    for(int i{}; i < max_row; i++)
    printf("%*.*f ", num_digits, num_decimal_digits , inp1D.at(i));
    cout << endl;
}

//Prints 2D vector.
void print2D(vector<vector<int>> inp2D, int num_digits, int max_row){
    for(auto row: inp2D){
        print1D(row, num_digits, max_row);
    }
}

//Prints 2D vector.
void print2D(vector<vector<uint64_t>> inp2D, int num_digits, int max_row){
    for(auto row: inp2D){
        print1D(row, num_digits, max_row);
    }
}

//Prints 2D vector.
void print2D(vector<vector<double>> inp2D, int num_digits, int num_decimal_digits, int max_row){
    for(auto row: inp2D){
        print1D(row, num_digits, num_decimal_digits, max_row);
    }
}

//Prints 3D vector.
void print3D(vector<vector<vector<int>>> inp3D, int num_digits, int max_row){
    for(auto mat: inp3D){
        print2D(mat, num_digits, max_row);
        cout << endl;
    }
}

//Prints 3D vector.
void print3D(vector<vector<vector<uint64_t>>> inp3D, int num_digits, int max_row){
    for(auto mat: inp3D){
        print2D(mat, num_digits, max_row);
        cout << endl;
    }
}

//Prints 3D vector.
void print3D(vector<vector<vector<double>>> inp3D, int num_digits, int num_decimal_digits, int max_row){
    for(auto mat: inp3D){
        print2D(mat, num_digits, num_decimal_digits, max_row);
        cout << endl;
    }
}

//Checks if 2D vectors A, B are same size.
bool sizeCheck2D(vector<vector<int>> A, vector<vector<int>> B, string tag){
    int d0 = A.size();
    int d1 = A.at(0).size();
    if(d0 != B.size()){
        cout << "ERROR in " << tag << ": dim0 mismatch! " << d0  << " != " << B.size() << endl;
        return false;
    }
    if(d1 != B.at(0).size()){
        cout << "ERROR in " << tag << ": dim0 mismatch! " << d1  << " != "<< B.at(0).size() << endl;
        return false;
    }
    return true;
}

//Checks if two numbers are equal and print warning if not
bool warnNotEq(int A, int B, string tag){
    if(A != B){
        cout << "WARNING " << tag << ": " << A  << " != " << B << endl;
        return false;
    }
    return true;
}

//Compares sizes of 2D vectors A, B and returns true if same.
bool sizeCheck2D(vector<vector<double>> A, vector<vector<double>> B, string tag){
    int d0 = A.size();
    int d1 = A.at(0).size();
    if(d0 != B.size()){
        cout << "ERROR in " << tag << ": dim0 mismatch! " << d0  << " != " << B.size() << endl;
        return false;
    }
    if(d1 != B.at(0).size()){
        cout << "ERROR in " << tag << ": dim1 mismatch! " << d1  << " != "<< B.at(0).size() << endl;
        return false;
    }
    return true;
}

// Vanilla Matrix-Matrix Multiplication: A*B.
vector<vector<int>> matMult(vector<vector<int>> A, vector<vector<int>> B){
    vector<vector<int>> out (A.size(), vector<int>(B.at(0).size(), 0));
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<B.at(0).size(); j++){
            for(int k = 0; k< A.at(0).size(); k++){
                out.at(i).at(j) = out.at(i).at(j) + A.at(i).at(k)*B.at(k).at(j);
            }
        }
    }
    return out;
}

// Vanilla Matrix-Vector Multiplication: A*b.
vector<int> matMult(vector<vector<int>> A, vector<int> B){
    vector<int> out (A.size());
    for(int i = 0; i < A.size(); i++){
        for(int k = 0; k< A.at(0).size(); k++){
            out.at(i) = out.at(i) + A.at(i).at(k)*B.at(k);
        }
        
    }
    return out;
}

//Vanilla Matrix Multiplication: A*B.
vector<vector<double>> matMult(vector<vector<double>> A, vector<vector<double>> B){
    vector<vector<double>> out (A.size(), vector<double>(B.at(0).size(), 0));
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<B.at(0).size(); j++){
            for(int k = 0; k< A.at(0).size(); k++){
                out.at(i).at(j) = out.at(i).at(j) + A.at(i).at(k)*B.at(k).at(j);
            }
        }
    }
    return out;
}

// Vanilla Matrix-Vector Multiplication: A*b.
vector<double> matMult(vector<vector<double>> A, vector<double> B){
    vector<double> out (A.size());
    for(int i = 0; i < A.size(); i++){
        for(int k = 0; k< A.at(0).size(); k++){
            out.at(i) = out.at(i) + A.at(i).at(k)*B.at(k);
        }
        
    }
    return out;
}

//Adds 2D vectors A, B elementwise.
vector<vector<int>> add2D(vector<vector<int>> A, vector<vector<int>> B){
    vector<vector<int>> out (A.size(), vector<int>(A.at(0).size(), 0));
    if(not sizeCheck2D(A, B, "add2D")){
        return out;
    }
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<A.at(0).size(); j++){
            out.at(i).at(j) = A.at(i).at(j)+B.at(i).at(j);
        }
    }
    return out;
}

//Adds 2D vectors A, B elementwise.
vector<vector<double>> add2D(vector<vector<double>> A, vector<vector<double>> B){
    vector<vector<double>> out (A.size(), vector<double>(A.at(0).size(), 0));
    if(not sizeCheck2D(A, B, "add2D")){
        return out;
    }
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<A.at(0).size(); j++){
            out.at(i).at(j) = A.at(i).at(j)+B.at(i).at(j);
        }
    }
    return out;
}

//Subtracts 2D vectors A, B elementwise (A-B).
vector<vector<int>> sub2D(vector<vector<int>> A, vector<vector<int>> B){
    vector<vector<int>> out (A.size(), vector<int>(A.at(0).size(), 0));
    if(not sizeCheck2D(A, B, "sub2D")){
        return out;
    }
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<A.at(0).size(); j++){
            out.at(i).at(j) = A.at(i).at(j)-B.at(i).at(j);
        }
    }
    return out;
}

//Subtracts 2D vectors A, B elementwise (A-B).
vector<vector<double>> sub2D(vector<vector<double>> A, vector<vector<double>> B){
    vector<vector<double>> out (A.size(), vector<double>(A.at(0).size(), 0));
    if(not sizeCheck2D(A, B, "sub2D")){
        return out;
    }
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<A.at(0).size(); j++){
            out.at(i).at(j) = A.at(i).at(j)-B.at(i).at(j);
        }
    }
    return out;
}

//Multiplies 2D vectors A, B elementwise.
vector<vector<int>> mul2D(vector<vector<int>> A, vector<vector<int>> B){
    vector<vector<int>> out (A.size(), vector<int>(A.at(0).size(), 0));
    if(not sizeCheck2D(A, B, "sub2D")){
        return out;
    }
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<A.at(0).size(); j++){
            out.at(i).at(j) = A.at(i).at(j)*B.at(i).at(j);
        }
    }
    return out;
}

//Multiplies 2D vectors A, B elementwise.
vector<vector<double>> mul2D(vector<vector<double>> A, vector<vector<double>> B){
    vector<vector<double>> out (A.size(), vector<double>(A.at(0).size(), 0));
    if(not sizeCheck2D(A, B, "sub2D")){
        return out;
    }
    for(int i = 0; i<A.size(); i++){
        for(int j = 0; j<A.at(0).size(); j++){
            out.at(i).at(j) = A.at(i).at(j)*B.at(i).at(j);
        }
    }
    return out;
}

//Scales all elements of a 2D vector with the given scaling_factor.
vector<vector<double>> scale2D(vector<vector<double>> inp2D, double scaling_factor, bool to_int){
    vector<vector<double>> out;
    for(auto row: inp2D){
        vector<double> out_row;
        for(auto val: row){
            val = val*scaling_factor;
            if (to_int) val = round(val);
            out_row.push_back(val);
        }
        out.push_back(out_row);
    }
    return out;
}

//Sums all elements of a 2D vector.
int sum2D(vector<vector<int>> inp2D){
    int out{};
    for(auto row: inp2D)
    for(auto val: row)
    out = out + val;
    return out;
}

//Sums all elements of a 2D vector.
uint64_t sum2D(vector<vector<uint64_t>> inp2D){
    double out{};
    for(auto row: inp2D)
    for(auto val: row)
    out = out + val;
    return out;
}

//Sums all elements of a 2D vector.
double sum2D(vector<vector<double>> inp2D){
    double out{};
    for(auto row: inp2D)
    for(auto val: row)
    out = out + val;
    return out;
}

//Checks if all values in a 1D vector are same scalar
bool isScalar(vector<int> inp1D, int scalar, double threshold){
    for(auto val: inp1D){
        if(abs(val - scalar)>threshold) return false;
    }
    return true;
}

//Checks if all values in a 1D vector are same scalar
bool isScalar(vector<double> inp1D, double scalar, double threshold){
    for(auto val: inp1D){
        if(abs(val - scalar)>threshold) return false;
    }
    return true;
}

//Computes Mean Absolute Error between two 2D vectors. Prints the MAE is threshold (>0) is passed.
double MAE2D(vector<vector<int>> A, vector<vector<int>> B, bool norm, double threshold){
    double out{};
    if(not sizeCheck2D(A, B, "MAE2D")){
        return out;
    }
    for(int i = 0; i < A.size(); i++){
        for(int j = 0; j< A.at(0).size(); j++){
            auto err = abs(A.at(i).at(j)-B.at(i).at(j));
            out += norm ? err/abs(A.at(i).at(j)) : err;
        }
    }
    out /= (A.size()*A.at(0).size());
    if(threshold>0){
        cout << "MAE: " << out;
        if(out < threshold) cout << " is less than threshold: "<< threshold <<": PASS!" << endl;
        else cout << " is greater than threshold: "<< threshold <<": FAIL!" << endl;
    }
    return out;
}

//Computes Mean Absolute Error between two 2D vectors. Prints the MAE is threshold (>0) is passed.
double MAE2D(vector<vector<double>> A, vector<vector<double>> B, bool norm, double threshold){
    double out{};
    if(not sizeCheck2D(A, B, "MAE2D")){
        return out;
    }
    for(int i = 0; i < A.size(); i++){
        for(int j = 0; j< A.at(0).size(); j++){
            auto err = abs(A.at(i).at(j)-B.at(i).at(j));
            out += norm ? err/abs(A.at(i).at(j)) : err;
        }
    }
    out /= (A.size()*A.at(0).size());
    if(threshold>0){
        cout << "MAE: " << out;
        if(out < threshold) cout << " is less than threshold: "<< threshold <<": PASS!" << endl;
        else cout << " is greater than threshold: "<< threshold <<": FAIL!" << endl;
    }
    return out;
}


//Computes the max of the absolute errors between elements of two 1D vectors.  
double maxAE1D(vector<double> A, vector<double> B, double scale, bool round_to_int){
    double out{};
    if(not sizeCheck2D({A}, {B}, "MAE2D")){
        return out;
    }
    for(int j = 0; j< A.size(); j++){
        auto valA = scale*A.at(j);
        auto valB = scale*B.at(j);
        if (round_to_int){
            valA = round(valA);
            valB = round(valB);
        }
        auto err = abs(valA - valB);
        out = err > out ? err : out;
    }
    return out;
}

//Returns the max element 1D vectors.  
double max1D(vector<double> A, bool abs_max){
    double out = A.at(0);
    if(abs_max) out = abs(out);
    for(auto val : A){
        if (abs_max) val = abs(val);
        out = val > out ? val : out;
    }
    return out;
}

//Returns the min element 1D vectors.  
double min1D(vector<double> A, bool abs_min){
    double out = A.at(0);
    if(abs_min) out = abs(out);
    for(auto val : A){
        if (abs_min) val = abs(val);
        out = val < out ? val : out;
    }
    return out;
}

//Pads 2D vector to [(padT + dim0 + padB), (padL + dim1 + padR)], i.e adds padT rows on top, padB rows
//in bottom, padL columns on left and padR columns on right.
vector<vector<int>> pad2D(vector<vector<int>> inp2D, int padT, int padB, int padL, int padR){
    auto out = zeros2Dint(padT+padB, inp2D.at(0).size());
    out.insert(out.begin()+padT, inp2D.begin(), inp2D.end());
    for(int i = 0; i < out.size(); i++){
        out.at(i).insert(out.at(i).begin(), padL, 0);
        out.at(i).insert(out.at(i).end(), padR, 0);
    }
    return out;
}

//Pads 2D vector to [(padT + dim0 + padB), (padL + dim1 + padR)], i.e adds padT rows on top, padB rows
//in bottom, padL columns on left and padR columns on right.
vector<vector<double>> pad2D(vector<vector<double>> inp2D, int padT, int padB, int padL, int padR){
    auto out = zeros2D(padT+padB, inp2D.at(0).size());
    out.insert(out.begin()+padT, inp2D.begin(), inp2D.end());
    for(int i = 0; i < out.size(); i++){
        out.at(i).insert(out.at(i).begin(), padL, 0);
        out.at(i).insert(out.at(i).end(), padR, 0);
    }
    return out;
}

//Unpads [dim0, dim1] 2D vector to [(-unpadT + dim0 - unpadB), (-unpadL + dim1 - unpadR)], i.e removes 
//top unpadT rows, bottom unpadB rows, left unpadL columns and right unpadR columns.
vector<vector<int>> unpad2D(vector<vector<int>> inp2D, int unpadT, int unpadB, int unpadL, int unpadR){
    int hi  = inp2D.size();
    int wi  = inp2D.at(0).size();
    int hout = hi -unpadT-unpadB;
    int wout = wi -unpadL-unpadR;
    vector<vector<int>> out (hout, vector<int>(wout, 0));
    for(int i = unpadT; i < hi-unpadB; i++){
        int ii = i-unpadT;
        out.at(ii) = vector<int>(inp2D.at(i).begin()+unpadL, inp2D.at(i).end()-unpadR);  
    }
    return out;
}

//Unpads [dim0, dim1] 2D vector to [(-unpadT + dim0 - unpadB), (-unpadL + dim1 - unpadR)], i.e removes 
//top unpadT rows, bottom unpadB rows, left unpadL columns and right unpadR columns.
vector<vector<double>> unpad2D(vector<vector<double>> inp2D, int unpadT, int unpadB, int unpadL, int unpadR){
    int hi  = inp2D.size();
    int wi  = inp2D.at(0).size();
    int hout = hi -unpadT-unpadB;
    int wout = wi -unpadL-unpadR;
    vector<vector<double>> out (hout, vector<double>(wout, 0));
    for(int i = unpadT; i < hi-unpadB; i++){
        int ii = i-unpadT;
        out.at(ii) = vector<double>(inp2D.at(i).begin()+unpadL, inp2D.at(i).end()-unpadR);  
    }
    return out;
}

//Computes smallest 2 power greater than the input.
int ceil2Pow(int val){
    return pow(2, ceil(log(val)/log(2)));
}

//Pads 2D vector to dimensions of 2 powers with zeros on bottom and right.
vector<vector<int>> pad2pow2D(vector<vector<int>> inp2D){
    return pad2D(inp2D, 0, ceil2Pow(inp2D.size())-inp2D.size(), 0, ceil2Pow(inp2D.at(0).size())-inp2D.at(0).size());
}

//Pads 2D vector to dimensions of 2 powers with zeros on bottom and right.
vector<vector<double>> pad2pow2D(vector<vector<double>> inp2D){
    return pad2D(inp2D, 0, ceil2Pow(inp2D.size())-inp2D.size(), 0, ceil2Pow(inp2D.at(0).size())-inp2D.at(0).size());
}

//Repeats a 2D vector nReps times in the first dimension, i.e. vertically.
vector<vector<int>> repMat2D(vector<vector<int>> inp2D, int nReps){
    auto out = inp2D;
    while(nReps>1){
        out.insert(out.end(), inp2D.begin(), inp2D.end());
        nReps -= 1;
    }
    return out;
}

//Repeats a 2D vector nReps times in the first dimension, i.e. vertically.
vector<vector<double>> repMat2D(vector<vector<double>> inp2D, int nReps){
    auto out = inp2D;
    while(nReps>1){
        out.insert(out.end(), inp2D.begin(), inp2D.end());
        nReps -= 1;
    }
    return out;
}

// Rotates a 2D vector: nRotRows rows up and nRotCols cols left.
vector<vector<uint64_t>> rotate2D(vector<vector<uint64_t>> inp2D, int nRotRows, int nRotCols){
 
    auto out = inp2D;

    nRotRows %= out.size();
    if(nRotRows<0) nRotRows += out.size();
    nRotCols %= out.at(0).size();
    if(nRotCols<0) nRotCols += out.at(0).size();

    if (nRotRows > 0) rotate(out.begin(), out.begin()+nRotRows, out.end());
    if (nRotCols > 0){
        for(int i{}; i < out.size(); i++) rotate(out.at(i).begin(), out.at(i).begin()+nRotCols, out.at(i).end());
    }
    return out;
}

// Rotates a 2D vector: nRotRows rows up and nRotCols cols left.
vector<vector<double>> rotate2D(vector<vector<double>> inp2D, int nRotRows, int nRotCols){
 
    auto out = inp2D;

    nRotRows %= out.size();
    if(nRotRows<0) nRotRows += out.size();
    nRotCols %= out.at(0).size();
    if(nRotCols<0) nRotCols += out.at(0).size();

    if (nRotRows > 0) rotate(out.begin(), out.begin()+nRotRows, out.end());
    if (nRotCols > 0){
        for(int i{}; i < out.size(); i++) rotate(out.at(i).begin(), out.at(i).begin()+nRotCols, out.at(i).end());
    }
    return out;
}

// Function to slice a given vector
// from range X to Y
// source: https://www.geeksforgeeks.org/slicing-a-vector-in-c/
vector<int> slice1D(vector<int>& arr,
                    int X, int Y)
{
    if(Y<=0) Y = arr.size() + Y - 1;
 
    // Starting and Ending iterators
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y + 1;
 
    // To store the sliced vector
    vector<int> result(Y - X + 1);
 
    // Copy vector using copy function()
    copy(start, end, result.begin());
 
    // Return the final sliced vector
    return result;
}

// Function to slice a given vector
// from range X to Y
// source: https://www.geeksforgeeks.org/slicing-a-vector-in-c/
vector<uint64_t> slice1D(vector<uint64_t>& arr,
                    int X, int Y)
{
    if(Y<=0) Y = arr.size() + Y - 1;
 
    // Starting and Ending iterators
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y + 1;
 
    // To store the sliced vector
    vector<uint64_t> result(Y - X + 1);
 
    // Copy vector using copy function()
    copy(start, end, result.begin());
 
    // Return the final sliced vector
    return result;
}

// Function to slice a given vector
// from range X to Y
// source: https://www.geeksforgeeks.org/slicing-a-vector-in-c/
vector<double> slice1D(vector<double>& arr,
                    int X, int Y)
{
    if(Y<=0) Y = arr.size() + Y - 1;

    // Starting and Ending iterators
    auto start = arr.begin() + X;
    auto end = arr.begin() + Y + 1;
 
    // To store the sliced vector
    vector<double> result(Y - X + 1);
 
    // Copy vector using copy function()
    copy(start, end, result.begin());
 
    // Return the final sliced vector
    return result;
}

vector<int> rowEncode2D(vector<vector<int>> inp2D){
    vector<int> data;
    for(auto row: inp2D){
        data.insert(data.end(), row.begin(), row.end());
    }
    return data;
}

vector<uint64_t> rowEncode2D(vector<vector<uint64_t>> inp2D){
    vector<uint64_t> data;
    for(auto row: inp2D){
        data.insert(data.end(), row.begin(), row.end());
    }
    return data;
}

vector<double> rowEncode2D(vector<vector<double>> inp2D){
    vector<double> data;
    for(auto row: inp2D){
        data.insert(data.end(), row.begin(), row.end());
    }
    return data;
}

uint64_t float_to_fixed(double flt_val, double scale){
    return (uint64_t) round(flt_val*scale);
}

double fixed_to_float(uint64_t fxd_val, double scale){
    return (double) fxd_val*scale;
}

matrixPT matPTEncode(vector<vector<double>> inp2D){
    matrixPT out;
    out.nrows = inp2D.size();
    out.ncols = inp2D.at(0).size();
    vector<double> data;
    int sum{};
    for(auto row: inp2D){
        data.insert(data.end(), row.begin(), row.end());
        if(sum>0) continue;
        for(auto val: row){
            sum = sum + val;
            if (sum > 0){
                out.isTransperent = false;
                break;
            }
        }
    }
    out.data = data;
    return out;
}

vector<vector<double>> matPTDecode(matrixPT mat){
    vector<vector<double>> out(mat.nrows);
    for(int i = 0; i < mat.nrows; i++){
        auto start = mat.data.begin() + i*mat.ncols;
        auto end   = start + mat.ncols;
        out.at(i).insert(out.at(i).begin(), start, end);
    }
    return out;
}

matrixPT addPT(matrixPT Apt, matrixPT Bpt){
    auto A     = matPTDecode(Apt);
    auto B     = matPTDecode(Bpt);
    auto out   = add2D(A, B);
    auto outPT = matPTEncode(out);
    return outPT;
}

matrixPT subPT(matrixPT Apt, matrixPT Bpt){
    auto A     = matPTDecode(Apt);
    auto B     = matPTDecode(Bpt);
    auto out   = sub2D(A, B);
    auto outPT = matPTEncode(out);
    return outPT;
}

matrixPT mulPT(matrixPT Apt, matrixPT Bpt){
    auto A     = matPTDecode(Apt);
    auto B     = matPTDecode(Bpt);
    auto out   = mul2D(A, B);
    auto outPT = matPTEncode(out);
    return outPT;
}

double sumPT(matrixPT Apt){
    auto A     = matPTDecode(Apt);
    auto out   = sum2D(A);
    return out;
}

matrixPT rotatePT(matrixPT mat, int rotVal){
    matrixPT out = mat;

    int len = out.nrows * out.ncols;
    rotVal = rotVal % len;
    if(rotVal<0) rotVal += len;
    if(rotVal>0){
        auto matData = out.data;
        rotate(matData.begin(), matData.begin()+rotVal, matData.end());
        out.data = matData;
    }
    return out;
}

matrixPT rotateRowsPT(matrixPT mat, int rotVal){
    auto out = rotatePT(mat, rotVal*mat.ncols);
    return out;
}

FCkerMasks encodeFCkernel(vector<vector<double>> A, vector<int> fcParms, bool preRotate){
    
    // A pre-processing (A will be known in plaintext at the server)

    int nout  = fcParms.at(0); // number of out neurons
    int nin   = fcParms.at(1); // number of in neurons
    int bsize = fcParms.at(2); // batchsize of input

    int d0 = A.size();
    int d1 = A.at(0).size();

    if(d0 != nout){
        cout << "encodeFCkernel Error: dim0 mismatch! " << d0  << " != " << nout << endl;
    }
    if(d1 != nin){
        cout << "encodeFCkernel Error: dim1 mismatch! " << d1  << " != "<< nin << endl;
    }

    vector<int> dSorted = (nin>nout)? vector<int>{nin, nout} : vector<int>{nout, nin};
    int dMax = dSorted.at(0);
    int dMin = dSorted.at(1);
    
    vector<vector<vector<double>>> kerMasks(dMin);
    vector<matrixPT> kerMaskPTs(dMin);

    for (int i = 0; i<dMin; i++){
        for(int j = 0; j<dMax; j++){
            kerMasks.at(i).push_back(vector<double>(bsize, A.at(j%nout).at((j+i)%nin)));
        }
        kerMaskPTs.at(i) = matPTEncode(kerMasks.at(i));
        if(preRotate) kerMaskPTs.at(i) = rotateRowsPT(kerMaskPTs.at(i), -1*i);
    }

    FCkerMasks AmaskPTs;

    AmaskPTs.data = kerMaskPTs;
    AmaskPTs.preRotated = preRotate;
    AmaskPTs.fcParms = fcParms;

    return AmaskPTs;
}

matrixPT encodeFCinput(vector<vector<double>> B, vector<int> fcParms){
    
    // B pre-processing (B will be encrypted at the client)
    
    int nout  = fcParms.at(0); // number of out neurons
    int nin   = fcParms.at(1); // number of in neurons
    int bsize = fcParms.at(2); // batchsize of input

    int dOut = (nin>nout)? nin : nout;

    auto BexpPT = matPTEncode(repMat2D(B, dOut/nin));
    
    return BexpPT;
}

matrixPT matMultPTvanilla(FCkerMasks AmaskPTs, matrixPT BexpPT){
    
    // Online matmult
    
    int nRots{}, nMults{}, nAdds{};

    int nout  = AmaskPTs.fcParms.at(0); // number of out neurons
    int nin   = AmaskPTs.fcParms.at(1); // number of in neurons
    int bsize = AmaskPTs.fcParms.at(2); // batchsize of input

    vector<int> dSorted = (nin>nout)? vector<int>{nin, nout} : vector<int>{nout, nin};
    int dMax = dSorted.at(0);
    int dMin = dSorted.at(1);

    auto AB2 = zeros2D(dMax, bsize);
    auto AB2pt = matPTEncode(AB2);

    vector<matrixPT> psumPTs;
    matrixPT psumPT;
    vector<int> rotVals;
    for(int i = 0; i<nin; i++){
        if(not AmaskPTs.preRotated){
            psumPT = rotateRowsPT(BexpPT, i);
            ++nRots;
            psumPT = mulPT(AmaskPTs.data.at(i%dMin), psumPT);
            ++nMults;
        }
        else{
            psumPT = mulPT(AmaskPTs.data.at(i%dMin), BexpPT);
            ++nMults;
            psumPT = rotateRowsPT(psumPT, i);
            ++nRots;
        }
        AB2pt = addPT(AB2pt, psumPT);
        ++nAdds;
    }
    cout  << "nMults:" << nMults << ", nRots:" << nRots << ", nAdds:" << nAdds << endl;
    return AB2pt;
}

matrixPT matMultPToptim(FCkerMasks AmaskPTs, matrixPT BexpPT){
    // Online matmult
    
    int nRots{}, nMults{}, nAdds{};

    int nout  = AmaskPTs.fcParms.at(0); // number of out neurons
    int nin   = AmaskPTs.fcParms.at(1); // number of in neurons
    int bsize = BexpPT.ncols; // batchsize of input

    vector<int> dSorted = (nin>nout)? vector<int>{nin, nout} : vector<int>{nout, nin};
    int dMax = dSorted.at(0);
    int dMin = dSorted.at(1);

    auto AB2 = zeros2D(dMax, bsize);
    auto AB2pt1 = matPTEncode(AB2);

    vector<matrixPT> psumPTs;
    matrixPT psumPT;
    vector<int> rotVals;
    for(int i = 0; i<dMin; i++){
        // cout<< "~~~~~~~~ rotVal:"<< i <<endl;
        if(AmaskPTs.preRotated){
            psumPT = mulPT(AmaskPTs.data.at(i%dMin), BexpPT);
            ++nMults;
            // cout << "psumPT after mult ~~~~~~~" <<endl;
            // print2D(matPTDecode(psumPT));
            psumPT = rotateRowsPT(psumPT, i); 
            ++nRots;
            // cout << "psumPT after rot ~~~~~~~~" <<endl;
            // print2D(matPTDecode(psumPT));
        }
        // cout << endl;
        AB2pt1 = addPT(AB2pt1, psumPT); 
        ++nAdds;
    }
    // cout << "AB2pt1 ~~~~~~~~~~~~" <<endl;
    // print2D(matPTDecode(AB2pt1));

    auto AB2pt  = AB2pt1;
    
    int nOutRots = log2(nin/dMin);
    cout << "nOutRots:" << nOutRots << endl;

    int outRotVal = nin;
    while (outRotVal > dMin){
        outRotVal >>= 1;
        auto AB2pt1Rot = rotateRowsPT(AB2pt, outRotVal); 
        ++nRots;
        // cout << "AB2pt1 after rot ~~~~~~~~" <<endl;
        // print2D(matPTDecode(AB2pt1Rot));
        AB2pt = addPT(AB2pt, AB2pt1Rot); 
        ++nAdds;
    }
    cout  << "nMults:" << nMults << ", nRots:" << nRots << ", nAdds:" << nAdds << endl;

    AB2pt.nrows = nout;
    return AB2pt;
}

matrixPT matMultPTpart1(FCkerMasks AmaskPTs, matrixPT BexpPT){
    
    // Online matmult
    
    int nRots{}, nMults{}, nAdds{};

    int nout  = AmaskPTs.fcParms.at(0); // number of out neurons
    int nin   = AmaskPTs.fcParms.at(1); // number of in neurons
    int bsize = AmaskPTs.fcParms.at(2); // batchsize of input

    vector<int> dSorted = (nin>nout)? vector<int>{nin, nout} : vector<int>{nout, nin};
    int dMax = dSorted.at(0);
    int dMin = dSorted.at(1);

    auto AB2 = zeros2D(dMax, bsize);
    auto AB2pt1 = matPTEncode(AB2);

    vector<matrixPT> psumPTs;
    matrixPT psumPT;
    vector<int> rotVals;
    for(int i = 0; i<dMin; i++){
        psumPT = mulPT(AmaskPTs.data.at(i%dMin), BexpPT);
        psumPT = rotateRowsPT(psumPT, i);
        AB2pt1 = addPT(AB2pt1, psumPT);
    }
    return AB2pt1;
}

matrixPT matMultPTpart2(matrixPT AB2pt1, vector<int> fcParms){

    int nout  = fcParms.at(0); // number of out neurons
    int nin   = fcParms.at(1); // number of in neurons
    int bsize = fcParms.at(2); // batchsize of input

    vector<int> dSorted = (nin>nout)? vector<int>{nin, nout} : vector<int>{nout, nin};
    int dMax = dSorted.at(0);
    int dMin = dSorted.at(1);

    auto AB2pt = AB2pt1;

    int outRotVal = nin;
    while (outRotVal > dMin){
        outRotVal >>= 1;
        auto AB2pt1Rot = rotateRowsPT(AB2pt, outRotVal);
        AB2pt = addPT(AB2pt, AB2pt1Rot);
    }
    return AB2pt;
}

vector<int> getTilingParms(vector<int> fcParms, int nSlots){
    
    int nout  = fcParms.at(0); // number of out neurons
    int nin   = fcParms.at(1); // number of in neurons
    int bsize = fcParms.at(2); // batchsize of input

    vector<int> dSorted = (nin>nout)? vector<int>{nin, nout} : vector<int>{nout, nin};
    int dMax = dSorted.at(0);
    int dMin = dSorted.at(1);

    // The input to the FC layer is (nin, bsize)
    // output of the FC layer is (nout, bsize)

    // number of batchs per ciphertext
    int nBpCT = (bsize > nSlots)? nSlots : bsize;
    cout << "nBpCT:" << nBpCT;

    // number of rows per ciphertext
    int nRowpCT = nSlots/nBpCT;
    cout << ", nRowpCT:" << nRowpCT;

    // number of input ciphertexts with (nRowpCT, nBpCT) blocks
    int nInCTs = (dMax*bsize)/nSlots;
    cout << ", nInCTs:" << nInCTs;

    // number of output ciphertexts with (nRowpCT, nBpCT) blocks
    int nOutCTs = nout/nRowpCT * (bsize/nBpCT);
    cout << ", nOutCTs:" << nOutCTs << endl;

    return {nRowpCT, nBpCT, nInCTs, nOutCTs};
}

// Extracts numbers from a string
vector<int> extractNumbers(const string& input) {
    vector<int> numbers;
    istringstream iss(input);
    string token;

    while (getline(iss, token, ',')) {
        // Remove leading and trailing spaces
        size_t start = token.find_first_not_of(" ");
        size_t end = token.find_last_not_of(" ");
        if (start != string::npos && end != string::npos) {
            token = token.substr(start, end - start + 1);
        }

        // Convert the cleaned token to an integer
        if (!token.empty()) {
            numbers.push_back(stoi(token));
        }
    }

    return numbers;
}