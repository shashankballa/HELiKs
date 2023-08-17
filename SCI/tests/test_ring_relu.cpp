/*
Authors: Deevashwer Rathee
Copyright:
Copyright (c) 2021 Microsoft Research
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

#include "NonLinear/relu-ring.h"
#include "Math/math-functions.h"
#include <fstream>
#include <iostream>
#include <thread>

#define MAX_THREADS 4

using namespace sci;
using namespace std;

int party = 0;
int num_relu = 35, port = 32000;
int b = 4;
int batch_size = 0;
string address = "127.0.0.1";
int num_threads = 4;
bool six_comparison = true;
int s_x = 28;
int bitlength = 32;

uint64_t mask_x = (bitlength == 64 ? -1 : ((1ULL << bitlength) - 1));

sci::IOPack *iopackArr[MAX_THREADS];
sci::OTPack *otpackArr[MAX_THREADS];

void relu_thread(int tid, uint64_t *x, uint64_t *y, int num_ops, uint64_t six) {
  MathFunctions *math;
  if (tid & 1) {
    math = new MathFunctions(3 - party, iopackArr[tid], otpackArr[tid]);
  } else {
    math = new MathFunctions(party, iopackArr[tid], otpackArr[tid]);
  }
  math->ReLU(num_ops, x, y, bitlength, six);

  delete math;
}

void ring_relu_thread(int tid, uint64_t *z, uint64_t *x, int lnum_relu) {
  ReLURingProtocol<uint64_t> *relu_oracle;
  if (tid & 1) {
    relu_oracle = new ReLURingProtocol<uint64_t>(3 - party, RING,
                                                  iopackArr[tid], bitlength, b,
                                                  otpackArr[tid]);
  } else {
    relu_oracle = new ReLURingProtocol<uint64_t>(
        party, RING, iopackArr[tid], bitlength, b, otpackArr[tid]);
  }
  if (batch_size) {
    for (int j = 0; j < lnum_relu; j += batch_size) {
      if (batch_size <= lnum_relu - j) {
        relu_oracle->relu(z + j, x + j, batch_size);
      } else {
        relu_oracle->relu(z + j, x + j, lnum_relu - j);
      }
    }
  } else {
    relu_oracle->relu(z, x, lnum_relu);
  }

  delete relu_oracle;
  return;
}

int main(int argc, char **argv) {
  /************* Argument Parsing  ************/
  /********************************************/
  ArgMapping amap;
  amap.arg("r", party, "Role of party: ALICE = 1; BOB = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("N", num_relu, "Number of ReLU operations");
  amap.arg("nt", num_threads, "Number of threads");
  amap.arg("six", six_comparison, "ReLU6?");
  amap.arg("ip", address, "IP Address of server (ALICE)");

  amap.parse(argc, argv);

  assert(num_threads <= MAX_THREADS);

  
  cout << "========================================================" << endl;
  cout << "======================= RELU-OT ========================" << endl;
  cout << "Role: " << party << " - Bitlength: " << bitlength
       << " - Radix Base: " << b << "\n# ReLUs: " << num_relu
       << " - Batch Size: " << batch_size << " - # Threads: " << num_threads
       << endl;
  cout << "========================================================" << endl;


  /************ Generate Test Data ************/
  /********************************************/
  PRG128 prg;

  uint64_t *x = new uint64_t[num_relu];
  uint64_t *y = new uint64_t[num_relu];

  prg.random_data(x, num_relu * sizeof(uint64_t));

  for (int i = 0; i < num_relu; i++) {
    x[i] &= mask_x;
  }
  uint64_t six;
  if (six_comparison)
    six = (6ULL << s_x);
  else
    six = 0;

  /********** Setup IO and Base OTs ***********/
  /********************************************/

  for (int i = 0; i < num_threads; i++) {
    iopackArr[i] = new IOPack(party, port + i, address);
    if (i & 1) {
      otpackArr[i] = new OTPack(iopackArr[i], 3 - party);
    } else {
      otpackArr[i] = new OTPack(iopackArr[i], party);
    }
  }
  std::cout << "All Base OTs Done" << std::endl;

  /************** Fork Threads ****************/
  /********************************************/

  uint64_t total_comm = 0;
  uint64_t thread_comm[num_threads];
  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm();
  }
  auto start = clock_start();
  std::thread relu_threads[num_threads];
  int chunk_size = num_relu / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int offset = i * chunk_size;
    int lnum_ops;
    if (i == (num_threads - 1)) {
      lnum_ops = num_relu - offset;
    } else {
      lnum_ops = chunk_size;
    }
    relu_threads[i] =
        // std::thread(relu_thread, i, x + offset, y + offset, lnum_ops, six);
        std::thread(ring_relu_thread, i, x + offset, y + offset, lnum_ops);
  }
  for (int i = 0; i < num_threads; ++i) {
    relu_threads[i].join();
  }
  long long t = time_from(start);

  for (int i = 0; i < num_threads; i++) {
    thread_comm[i] = iopackArr[i]->get_comm() - thread_comm[i];
    total_comm += thread_comm[i];
  }

  /************** Verification ****************/
  /********************************************/
  if (party == ALICE) {
    iopackArr[0]->io->send_data(x, num_relu * sizeof(uint64_t));
    iopackArr[0]->io->send_data(y, num_relu * sizeof(uint64_t));
  } else { // party == BOB
    uint64_t *x0 = new uint64_t[num_relu];
    uint64_t *y0 = new uint64_t[num_relu];
    iopackArr[0]->io->recv_data(x0, num_relu * sizeof(uint64_t));
    iopackArr[0]->io->recv_data(y0, num_relu * sizeof(uint64_t));

    for (int i = 0; i < num_relu; i++) {
      int64_t X = signed_val(x[i] + x0[i], bitlength);
      int64_t Y = signed_val(y[i] + y0[i], bitlength);
      int64_t expectedY = X;
      if (X < 0)
        expectedY = 0;
      if (six != 0) {
        if (X > int64_t(six))
          expectedY = six;
      }
      // cout << X << "\t" << Y << "\t" << expectedY << endl;
      assert(Y == expectedY);
    }

    cout << "ReLU" << (six == 0 ? "" : "6") << " Tests Passed" << endl;

    delete[] x0;
    delete[] y0;
  }

  /**** Process & Write Benchmarking Data *****/
  /********************************************/
  cout << "Comm. Sent/ell  : "
            << double(total_comm * 8) / (bitlength * num_relu) << std::endl;
            
  cout << "Number of ReLU/s: " << (double(num_relu) / t) * 1e6 << std::endl;
  cout << "Total Comm. (MB): " << double(total_comm)/ (1<<20)  << std::endl;
  cout << "Total ReLU Time : " << t/1000
       << " ms" << endl;
  // cout << "Number of ReLU/s:\t" << (double(num_relu) / t) * 1e6 << std::endl;
  // cout << "ReLU Time\t" << t / (1000.0) << " ms" << endl;
  // cout << "ReLU Bytes Sent\t" << total_comm << " bytes" << endl;

  /******************* Cleanup ****************/
  /********************************************/
  delete[] x;
  delete[] y;
  for (int i = 0; i < num_threads; i++) {
    delete iopackArr[i];
    delete otpackArr[i];
  }
}
