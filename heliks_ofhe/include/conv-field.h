/*
Original Author: ryanleh
Modified Work Copyright (c) 2020 Microsoft Research

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Modified by Deevashwer Rathee
*/

#ifndef CONV_FIELD_H__
#define CONV_FIELD_H__

#include <random>
#include <cassert>

#include "utils.h"
#include "openfhe.h"

struct ConvMetadata {
  int slot_count;
  // Number of plaintext slots in a half ciphertext
  // (since ciphertexts are a two column matrix)
  int32_t pack_num;
  // Number of std::vector<std::vector<int64_t>>s that can fit in a half ciphertext
  int32_t chans_per_half;
  // Number of input ciphertexts for convolution
  int32_t inp_ct;
  // Number of output ciphertexts
  int32_t out_ct;
  // std::vector<std::vector<std::vector<int64_t>>> and std::vector<std::vector<std::vector<std::vector<int64_t>>>> metadata
  int32_t image_h;
  int32_t image_w;
  size_t image_size;
  int32_t inp_chans;
  int32_t filter_h;
  int32_t filter_w;
  int32_t filter_size;
  int32_t out_chans;
  // How many total ciphertext halves the input and output take up
  int32_t inp_halves;
  int32_t out_halves;
  // The modulo used when deciding which output channels to pack into a mask
  int32_t out_mod;
  // How many permutations of ciphertexts are needed to generate all
  // intermediate rotation sets
  int32_t half_perms;
  /* The number of rotations for each ciphertext half */
  int32_t half_rots;
  // Total number of convolutions needed to generate all
  // intermediate rotations sets
  int32_t convs;
  int32_t stride_h;
  int32_t stride_w;
  int32_t output_h;
  int32_t output_w;
  int32_t pad_t;
  int32_t pad_b;
  int32_t pad_r;
  int32_t pad_l;

  vector<vector<vector<int>>> rot_amts;
  map<int, vector<vector<int>>> rot_map;

  int64_t prime_mod;
  int64_t mult_depth;
  int64_t min;
  int64_t max;

  bool print_cnts;
  bool use_heliks;
};

/* Use casting to do two conditionals instead of one - check if a > 0 and a < b
 */
inline bool condition_check(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

std::vector<std::vector<std::vector<int64_t>>> 
    pad_image(ConvMetadata data, std::vector<std::vector<std::vector<int64_t>>> &image);

void i2c(const std::vector<std::vector<std::vector<int64_t>>> &image, 
            std::vector<std::vector<int64_t>> &column, const int filter_h,
            const int filter_w, const int stride_h, const int stride_w,
            const int output_h, const int output_w);

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> 
    HE_preprocess_noise(const int64_t *const *secret_share, const ConvMetadata &data,
            lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, 
            lbcrypto::PublicKey<lbcrypto::DCRTPoly> &publicKey);

std::vector<std::vector<int64_t>> 
    preprocess_image_OP(std::vector<std::vector<std::vector<int64_t>>> &image, ConvMetadata data);

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> 
    filter_rotations(std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &input, 
            const ConvMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
            std::vector<int> &counts);

std::vector<lbcrypto::Plaintext> 
    HE_encode_input(std::vector<std::vector<int64_t>> &pt,
            const ConvMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc);

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> 
    HE_encrypt(std::vector<std::vector<int64_t>> &pt,
            const ConvMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
            lbcrypto::PublicKey<lbcrypto::DCRTPoly> &publicKey);

std::vector<std::vector<std::vector<lbcrypto::Plaintext>>> 
    HE_preprocess_filters_OP(std::vector<std::vector<std::vector<std::vector<int64_t>>>> &filters, 
            const ConvMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc);

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> 
    HE_conv_OP(std::vector<std::vector<std::vector<lbcrypto::Plaintext>>> &masks,
            std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> &rotations,
            const ConvMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, 
            lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &zero,
            std::vector<int> &counts);

std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> 
    HE_output_rotations(std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &convs, 
            const ConvMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
            lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &zero, 
            std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &enc_noise,
            std::vector<int> &counts);

int64_t **HE_decrypt(std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> &enc_result,
            const ConvMetadata &data, lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc, 
            lbcrypto::PrivateKey<lbcrypto::DCRTPoly> &secretKey);

class ConvField {
 public:
  int party;
  ConvMetadata data;
  
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> zero;

  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
  lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys;

  ConvField(){};
  ConvField(int ring_dim, bool use_heliks = true);

  int64_t get_prime_mod() { return data.prime_mod; };

  void gen_context(int ring_dim);

  void configure(bool verbose = false);

  std::vector<std::vector<std::vector<int64_t>>> 
    ideal_functionality(std::vector<std::vector<std::vector<int64_t>>> &image, 
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &filters);

  void non_strided_conv(int32_t H, int32_t W, int32_t CI, int32_t FH,
            int32_t FW, int32_t CO, std::vector<std::vector<std::vector<int64_t>>> *image, 
            std::vector<std::vector<std::vector<std::vector<int64_t>>>> *filters,
            std::vector<std::vector<std::vector<int64_t>>> &outArr,
            vector<int> &counts,
            bool verbose = false);

  void non_strided_conv_MR(int32_t H, int32_t W, int32_t CI, int32_t FH,
            int32_t FW, int32_t CO, std::vector<std::vector<std::vector<int64_t>>> *image, 
            std::vector<std::vector<std::vector<std::vector<int64_t>>>> *filters,
            std::vector<std::vector<std::vector<int64_t>>> &outArr,
            vector<int> &counts,
            bool verbose = false);

  void convolution(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW,
            int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft,
            int32_t zPadWRight, int32_t strideH, int32_t strideW,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &inputArr,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &filterArr,
            std::vector<std::vector<std::vector<std::vector<int64_t>>>> &outArr,
            std::vector<bool> options = {false, false, false, false});

  void convolution_MR(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW,
            int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft,
            int32_t zPadWRight, int32_t strideH, int32_t strideW,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &inputArr,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &filterArr,
            std::vector<std::vector<std::vector<std::vector<int64_t>>>> &outArr,
            std::vector<bool> options = {false, false, true, false});

  void convolution_heliks(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW,
            int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft,
            int32_t zPadWRight, int32_t strideH, int32_t strideW,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &inputArr,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &filterArr,
            std::vector<std::vector<std::vector<std::vector<int64_t>>>> &outArr,
            std::vector<bool> options = {false, false, true, false});

  void verify(int H, int W, int CI, int CO, std::vector<std::vector<std::vector<int64_t>>> &image,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> *filters,
            const std::vector<std::vector<std::vector<std::vector<int64_t>>>> &outArr);

};

#endif
