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

#include "LinearHE/conv-field.h"

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int image_h = 56;
int inp_chans = 64;
int filter_h = 3;
int out_chans = 64;
int pad_l = 0;
int pad_r = 0;
int stride = 2;
int filter_precision = 12;
int slots_count_ = 8192;

auto coeff_mod = GET_COEFF_MOD_HLK();

vector<string> roles = {"Public", "Server", "Client"};

bool verbose = false;
bool verify_output = false;

string options_str = "1111";
int numops = 4;
vector<bool> options(numops);

/*
  options[0] -> verify_output
  options[1] -> verbose
  options[2] -> use_heliks
  options[3] -> print_cnts
*/

void Conv(ConvField &he_conv, int32_t H, int32_t CI, int32_t FH, int32_t CO,
          int32_t zPadHLeft, int32_t zPadHRight, int32_t strideH, vector<bool> options) {
          
  bool verify_output = options[0];
  bool verbose = options[1];
  int newH = 1 + (H + zPadHLeft + zPadHRight - FH) / strideH;
  int N = 1;
  int W = H;
  int FW = FH;
  int zPadWLeft = zPadHLeft;
  int zPadWRight = zPadHRight;
  int strideW = strideH;
  int newW = newH;
  vector<vector<vector<vector<uint64_t>>>> inputArr(N);
  vector<vector<vector<vector<uint64_t>>>> filterArr(FH);
  vector<vector<vector<vector<uint64_t>>>> outArr(N);

  PRG128 prg;
  for (int i = 0; i < N; i++) {
    outArr[i].resize(newH);
    for (int j = 0; j < newH; j++) {
      outArr[i][j].resize(newW);
      for (int k = 0; k < newW; k++) {
        outArr[i][j][k].resize(CO);
      }
    }
  }
  if (party == ALICE) {
    for (int i = 0; i < FH; i++) {
      filterArr[i].resize(FW);
      for (int j = 0; j < FW; j++) {
        filterArr[i][j].resize(CI);
        for (int k = 0; k < CI; k++) {
          filterArr[i][j][k].resize(CO);
          prg.random_data(filterArr[i][j][k].data(), CO * sizeof(uint64_t));
          for (int h = 0; h < CO; h++) {
            // filterArr[i][j][k][h] = (i+1)*1000 + (j+1)*100 + (k+1)*10 + (h+1);
            filterArr[i][j][k][h] = ((int64_t)filterArr[i][j][k][h]) >> (64 - filter_precision);
          }
        }
      }
    }
  }
  for (int i = 0; i < N; i++) {
    inputArr[i].resize(H);
    for (int j = 0; j < H; j++) {
      inputArr[i][j].resize(W);
      for (int k = 0; k < W; k++) {
        inputArr[i][j][k].resize(CI);
        prg.random_mod_p<uint64_t>(inputArr[i][j][k].data(), CI, prime_mod);
      }
    }
  }
  uint64_t comm_start_NTT = he_conv.io->counter;
  he_conv.convolution(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                      zPadWRight, strideH, strideW, inputArr, filterArr, outArr,
                      options);
  uint64_t comm_end_NTT = he_conv.io->counter;
  cout << "[Conv] " << roles[party] << " Sent: " << (comm_end_NTT - comm_start_NTT) / (1.0 * (1ULL << 20))
       << endl;
}

int main(int argc, char **argv) {

  string coeff_mod_str = to_string(coeff_mod[0]);
  for (int i = 1; i < coeff_mod.size(); i++) {
    coeff_mod_str += "," + to_string(coeff_mod[i]);
  }
  
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("nt", num_threads, "Number of Threads");
  amap.arg("l", bitlength, "Bitlength");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("p", port, "Port Number");
  amap.arg("h", image_h, "Image Height/Width");
  amap.arg("f", filter_h, "Filter Height/Width");
  amap.arg("i", inp_chans, "Input Channels");
  amap.arg("o", out_chans, "Ouput Channels");
  amap.arg("s", stride, "stride");
  amap.arg("pl", pad_l, "Left Padding");
  amap.arg("pr", pad_r, "Right Padding");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.arg("op", options_str, "Options for computation");
  amap.arg("cmod", coeff_mod_str, "Coeff Modulus");
  amap.parse(argc, argv);
  prime_mod = sci::default_prime_mod.at(bitlength);

  for (int i = 0; i < options_str.size(); i++) {
    options[i] = options_str[i] == '1' ? true : false;
  }

  bool use_heliks = options[2];
  verbose = options[1];



  int paddedH = image_h + pad_l + pad_r;
  int paddedW = paddedH;
  int newH    = 1 + (paddedH - filter_h) / stride;
  int newW    = newH;
  int limitH  = filter_h + ((paddedH - filter_h) / stride) * stride;
  int limitW  = limitH;

  for (int s_row = 0; s_row < stride; s_row++) {
    for (int s_col = 0; s_col < stride; s_col++) {
      int lH  = ((limitH - s_row + stride - 1) / stride);
      int lW  = lH;
      int lFH = ((filter_h - s_row + stride - 1) / stride);
      int lFW = lFH;

      auto _slots_count_ =
            min(SEAL_POLY_MOD_DEGREE_MAX, 
            max(8192, 2 * next_pow2(lH * lW)));

      if(_slots_count_ > slots_count_) slots_count_ = _slots_count_;
    }
  }

  // slots_count_ =
  //       min(SEAL_POLY_MOD_DEGREE_MAX, 
  //       max(8192, 2 * next_pow2((image_h/stride) * (image_h/stride))));

  auto coeff_mod = extractNumbers(coeff_mod_str);

  if(party==1){
    cout << "=================================================================="
        << endl;
    cout << "Role: " << party << " - Bitlength: " << bitlength
        << " - Mod: " << prime_mod << " - Image: " << image_h << "x" << image_h
        << "x" << inp_chans << " - Filter: " << filter_h << "x" << filter_h
        << "x" << out_chans << "\n- Stride: " << stride << "x" << stride
        << " - Padding: " << pad_l << "x" << pad_r
        << " - # Threads: " << num_threads << endl;
    cout << "=================================================================="
        << endl;
  }

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port, false, true);
  
  ConvField *he_conv;

  if(use_heliks){
    if(slots_count_ >= 8192){
      he_conv = new ConvField(party, io, coeff_mod, slots_count_, verbose);
    }
    else{
      coeff_mod = {40, 28, 40};
      he_conv = new ConvField(party, io, coeff_mod, 4096, verbose);
    }
  } else {
    he_conv = new ConvField(party, io);
  }

  Conv(*he_conv, image_h, inp_chans, filter_h, out_chans, pad_l, pad_r, stride,
        options);

  io->flush();
  return 0;
}


/*

using namespace std;
using namespace seal;
using namespace sci;

int party = 0;
int bitlength = 32;
int num_threads = 4;
int port = 8000;
string address = "127.0.0.1";
int image_h = 56;
int inp_chans = 64;
int filter_h = 3;
int out_chans = 64;
int pad_l = 0;
int pad_r = 0;
int stride = 2;
int filter_precision = 12;
int slots_count_ = 8192;
bool verify_output = false;
bool verbose = false;

auto coeff_mod = GET_COEFF_MOD_HLK();
// string coeff_mod_str = to_string(coeff_mod[0]);
// for (int i = 1; i < coeff_mod.size(); i++) {
//   coeff_mod_str += "," + to_string(coeff_mod[i]);
// }

vector<string> roles = {"ALICE", "BOB"};


void Conv(ConvField &he_conv, int32_t H, int32_t CI, int32_t FH, int32_t CO,
          int32_t zPadHLeft, int32_t zPadHRight, int32_t strideH, bool verify_output, bool verbose) {
  int newH = 1 + (H + zPadHLeft + zPadHRight - FH) / strideH;
  int N = 1;
  int W = H;
  int FW = FH;
  int zPadWLeft = zPadHLeft;
  int zPadWRight = zPadHRight;
  int strideW = strideH;
  int newW = newH;
  vector<vector<vector<vector<uint64_t>>>> inputArr(N);
  vector<vector<vector<vector<uint64_t>>>> filterArr(FH);
  vector<vector<vector<vector<uint64_t>>>> outArr(N);

  PRG128 prg;
  for (int i = 0; i < N; i++) {
    outArr[i].resize(newH);
    for (int j = 0; j < newH; j++) {
      outArr[i][j].resize(newW);
      for (int k = 0; k < newW; k++) {
        outArr[i][j][k].resize(CO);
      }
    }
  }
  if (party == ALICE) {
    for (int i = 0; i < FH; i++) {
      filterArr[i].resize(FW);
      for (int j = 0; j < FW; j++) {
        filterArr[i][j].resize(CI);
        for (int k = 0; k < CI; k++) {
          filterArr[i][j][k].resize(CO);
          prg.random_data(filterArr[i][j][k].data(), CO * sizeof(uint64_t));
          for (int h = 0; h < CO; h++) {
            // filterArr[i][j][k][h] = (i+1)*1000 + (j+1)*100 + (k+1)*10 + (h+1);
            filterArr[i][j][k][h] = ((int64_t)filterArr[i][j][k][h]) >> (64 - filter_precision);
          }
        }
      }
    }
  }
  for (int i = 0; i < N; i++) {
    inputArr[i].resize(H);
    for (int j = 0; j < H; j++) {
      inputArr[i][j].resize(W);
      for (int k = 0; k < W; k++) {
        inputArr[i][j][k].resize(CI);
        prg.random_mod_p<uint64_t>(inputArr[i][j][k].data(), CI, prime_mod);
      }
    }
  }
  INIT_TIMER;
  uint64_t comm_start_NTT = he_conv.io->counter;
  START_TIMER;
  he_conv.convolution_NTT_SB(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                      zPadWRight, strideH, strideW, inputArr, filterArr, outArr,
                      verify_output, verbose);
  STOP_TIMER("Conv-NTT " + roles[party-1]);
  uint64_t comm_end_NTT = he_conv.io->counter;
  cout << "Conv-NTT " << roles[party-1] << " Sent: " << (comm_end_NTT - comm_start_NTT) / (1.0 * (1ULL << 20))
       << endl;
}

int main(int argc, char **argv) {

  string coeff_mod_str = to_string(coeff_mod[0]);
  for (int i = 1; i < coeff_mod.size(); i++) {
    coeff_mod_str += "," + to_string(coeff_mod[i]);
  }
  
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("nt", num_threads, "Number of Threads");
  amap.arg("l", bitlength, "Bitlength");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.arg("p", port, "Port Number");
  amap.arg("h", image_h, "Image Height/Width");
  amap.arg("f", filter_h, "Filter Height/Width");
  amap.arg("i", inp_chans, "Input Channels");
  amap.arg("o", out_chans, "Ouput Channels");
  amap.arg("s", stride, "stride");
  amap.arg("sc", slots_count_, "stride");
  amap.arg("pl", pad_l, "Left Padding");
  amap.arg("pr", pad_r, "Right Padding");
  amap.arg("fp", filter_precision, "Filter Precision");
  amap.arg("v", verbose, "Verbose");
  amap.arg("vo", verify_output, "Verify Output");
  amap.arg("cmod", coeff_mod_str, "Coeff Modulus");
  amap.parse(argc, argv);
  prime_mod = sci::default_prime_mod.at(bitlength);


  auto coeff_mod = extractNumbers(coeff_mod_str);

  if(party==1){
    cout << "=================================================================="
        << endl;
    cout << "Role: " << party << " - Bitlength: " << bitlength
        << " - Mod: " << prime_mod << " - Image: " << image_h << "x" << image_h
        << "x" << inp_chans << " - Filter: " << filter_h << "x" << filter_h
        << "x" << out_chans << "\n- Stride: " << stride << "x" << stride
        << " - Padding: " << pad_l << "x" << pad_r
        << " - # Threads: " << num_threads << endl;
    cout << "=================================================================="
        << endl;
  }


  slots_count_ =
      min(SEAL_POLY_MOD_DEGREE_MAX, 
      max(8192, 2 * next_pow2((image_h/stride) * (image_h/stride))));

  NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port, false, true);
  
  ConvField *he_conv;
  ConvField *he_conv;

  if(slots_count_ >= 8192){
    he_conv = new ConvField(party, io, coeff_mod, slots_count_, verbose);
  }
  else{
    coeff_mod = {40, 28, 40};
    he_conv = new ConvField(party, io, coeff_mod, 4096, verbose);
  }

  Conv(*he_conv, image_h, inp_chans, filter_h, out_chans, pad_l, pad_r, stride,
        verify_output, verbose);

  io->flush();
  return 0;
}
*/