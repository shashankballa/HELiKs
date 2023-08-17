/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "LinearHE/fc-field.h"

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
vector<string> roles = {"Public", "Server", "Client"};
size_t he_slot_count = POLY_MOD_DEGREE;
int bitlength = 32;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int num_rows = 10;
int common_dim = 63;
int filter_precision = 15;
vector<int> coeff_modulus = GET_COEFF_MOD_CF2();
bool verbose = false;
bool use_tiling = false;

string options_str = "1100001010";
int numops = 10;
vector<bool> options(numops);

/*
  options[0] -> verify_output
  options[1] -> verbose
  options[2] -> mul_then_rot
  options[3] -> rot_one_step
  options[4] -> pre_comp_ntt
  options[5] -> use_symm_key
  options[6] -> skip_he_ras
  options[7] -> enable_ss
  options[8] -> print_times
  options[9] -> use_tiling
*/

void MatMul(FCField &he_fc_bfv, int32_t num_rows, int32_t common_dim, 
            vector<bool> options) {
  int num_cols = 1;
  vector<vector<uint64_t>> A(num_rows);   // Weights
  vector<vector<uint64_t>> B(common_dim); // Image
  vector<vector<uint64_t>> C(num_rows);
  PRG128 prg;
  for (int i = 0; i < num_rows; i++) {
    A[i].resize(common_dim);
    C[i].resize(num_cols);
    if (party == ALICE) {
      prg.random_data(A[i].data(), common_dim * sizeof(uint64_t));
      for (int j = 0; j < common_dim; j++) {
        A[i][j] = ((int64_t)A[i][j]) >> (64 - filter_precision);
      }
    }
  }
  for (int i = 0; i < common_dim; i++) {
    B[i].resize(1);
    prg.random_mod_p<uint64_t>(B[i].data(), num_cols, prime_mod);
  }
  uint64_t comm_start = he_fc_bfv.io->counter;
  he_fc_bfv.matrix_multiplication(num_rows, common_dim, num_cols, A, B, C, 
                                  options);
  uint64_t comm_end = he_fc_bfv.io->counter;
  cout << roles[party] << " Sent MB: " 
       << (comm_end - comm_start) / (1.0 * (1ULL << 20)) << endl;
}

int main(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("t", num_threads, "Number of threads");
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("n", num_rows, "Rows in Weight Matrix");
  amap.arg("c", common_dim, "Image Length / Columns in Weight Matrix");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("l", bitlength, "Bitlength of inputs");
  amap.arg("sc", he_slot_count, "HE Slot Count");
  amap.arg("o", options_str, "Options for computation");
  amap.parse(argc, argv);
  prime_mod = sci::default_prime_mod.at(bitlength);

  for (int i = 0; i < options_str.size(); i++) {
    options[i] = options_str[i] == '1' ? true : false;
  }
  verbose    = options[1];
  use_tiling = options[9];

  if(!use_tiling){
    he_slot_count = 
      min(
        (int) SEAL_POLY_MOD_DEGREE_MAX,
        max(
          (int) POLY_MOD_DEGREE, 
          max(
            2 * next_pow2(common_dim), 
            next_pow2(num_rows))));
  }

  if(options[2]) // use smaller parameters for mul_then_rot
    coeff_modulus = GET_COEFF_MOD_HLK();
  
  cout << "[test_field_fc]";
  if (verbose){
    cout << " Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
    cout << "+ Rows, Cols: (" << num_rows << ", " << common_dim << ")\n";
    cout << "+ Role (p#) : " + roles[party] + " (p" << party << ")\n";
    cout << "+ Bitlength : " << bitlength << endl;
    cout << "+ # Threads : " << num_threads << endl;
    cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
  }
  else{
    cout << " (" << num_rows << " x " << common_dim << ") Options: " 
         << options_str << endl;
  }
  
  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port, 
                        false, !verbose);

  FCField he_fc_bfv(party, io, coeff_modulus, he_slot_count, verbose);

  MatMul(he_fc_bfv, num_rows, common_dim, options);

  io->flush();
  return 0;
}
