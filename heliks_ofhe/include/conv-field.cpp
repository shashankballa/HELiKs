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

#include "conv-field.h"
#include "utils.h"

using namespace std;
using namespace lbcrypto;

/* pad_image
*/
vector<vector<vector<int64_t>>> pad_image(ConvMetadata data, vector<vector<vector<int64_t>>> &image) {
  int image_h = data.image_h;
  int image_w = data.image_w;
  vector<vector<vector<int64_t>>> p_image;

  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;
  int pad_top = data.pad_t;
  int pad_left = data.pad_l;

  for (vector<vector<int64_t>> &channel : image) {
    p_image.push_back(pad2D(channel, data.pad_t, data.pad_b, data.pad_l, data.pad_r));
  }
  return p_image;
}

/* i2c
*/
// Adapted im2col algorithm from Caffe framework 
void i2c(const vector<vector<vector<int64_t>>> &image, vector<vector<int64_t>> &column, const int filter_h,
         const int filter_w, const int stride_h, const int stride_w,
         const int output_h, const int output_w) {
  int height = image[0].size();
  int width = image[0].at(0).size();
  int channels = image.size();

  int col_width = column.at(0).size();

  // Index counters for images
  int column_i = 0;
  const int channel_size = height * width;
  for (auto &channel : image) {
    for (int filter_row = 0; filter_row < filter_h; filter_row++) {
      for (int filter_col = 0; filter_col < filter_w; filter_col++) {
        int input_row = filter_row;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!condition_check(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              int row_i = column_i / col_width;
              int col_i = column_i % col_width;
              column.at(row_i).at(col_i) = 0;
              column_i++;
            }
          } else {
            int input_col = filter_col;
            for (int output_col = output_w; output_col; output_col--) {
              if (condition_check(input_col, width)) {
                int row_i = column_i / col_width;
                int col_i = column_i % col_width;
                column.at(row_i).at(col_i) = channel.at(input_row).at(input_col);
                column_i++;
              } else {
                int row_i = column_i / col_width;
                int col_i = column_i % col_width;
                column.at(row_i).at(col_i) = 0;
                column_i++;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

/* printConvMetaData
*/
void printConvMetaData(ConvMetadata data){
  printf("ConvMetadata: \n");
  printf("+ slot_count: %i, pack_num: %i \n", data.slot_count, data.pack_num);
  printf("+ chans_per_half: %i \n", data.chans_per_half);
  printf("+ inp_ct : %i, out_ct : %i \n", data.inp_ct, data.out_ct);
  printf("+ image_h: %i, image_w: %i \n", data.image_h, data.image_w);
  printf("+ image_size: %li \n", data.image_size);
  printf("+ inp_chans: %i \n", data.inp_chans);
  printf("+ filter_h: %i, filter_w: %i \n", data.filter_h, data.filter_w);
  printf("+ filter_size: %i \n", data.filter_size);
  printf("+ out_chans: %i \n", data.out_chans);
  printf("+ inp_halves: %i, out_halves: %i \n", data.inp_halves, data.out_halves);
  printf("+ out_mod: %i \n", data.out_mod);
  printf("+ half_perms: %i, half_rots: %i \n", data.half_perms, data.half_rots);
  printf("+ convs: %i \n", data.convs);
  printf("+ stride_h: %i, stride_w: %i \n", data.stride_h, data.stride_w);
  printf("+ output_h: %i, output_w: %i \n", data.output_h, data.output_w);
  printf("+ pad_t   : %i, pad_b   : %i \n", data.pad_t, data.pad_b);
  printf("+ pad_r   : %i, pad_l   : %i \n", data.pad_r, data.pad_l);
}

/*
  Performs a rotation of the ciphertext by the given amount. The rotation is 
  performed using the Near Adjacent Form (NAF) of the rotation amount.
*/
Ciphertext<DCRTPoly> rotate_arb2(Ciphertext<DCRTPoly> &ct, int rot, 
                                CryptoContext<DCRTPoly> &cc){
  
  Ciphertext<DCRTPoly> result = ct;

  int ring_dim = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension();
  rot = rot % ring_dim;

  if(rot == 0) return result;

 // Convert the steps to Near Adjacent Form (NAF)
  vector<int> naf_steps = naf(rot);

  // cout << "+ [rotate_arb2] ring_dim:" << ring_dim << "; rot: " << rot << "; naf_steps: ";
  for(auto step : naf_steps){
    // cout << step << " ";
    while (step > ring_dim/2){
      step -= ring_dim;
    }
    result = cc->EvalRotate(result, step);
  }
  // cout << endl;
  return result;
}

/* HE_preprocess_noise
*/
// Generates a masking vector of random noise that will be applied to parts of
// the ciphertext that contain leakage from the convolution
vector<Ciphertext<DCRTPoly>> HE_preprocess_noise(const int64_t *const *secret_share,
                                       const ConvMetadata &data,
                                       CryptoContext<DCRTPoly> &cc,
                                       PublicKey<DCRTPoly> &pk) {
  // vector<vector<int64_t>> noise(data.out_ct,
  //                                vector<int64_t>(data.slot_count, 0ULL));

   vector<vector<int64_t>> noise = gen2D_UID_int64( data.out_ct, data.slot_count, data.min, data.max);

  // Sample randomness into vector
  // for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
  //   prg.random_mod_p<int64_t>(noise[ct_idx].data(), data.slot_count,
  //                              prime_mod);
  // }
  vector<Ciphertext<DCRTPoly>> enc_noise(data.out_ct);

  // Puncture the vector with 0s where an actual convolution result value lives
// #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    int out_base = 2 * ct_idx * data.chans_per_half;
    for (int out_c = 0;
         out_c < 2 * data.chans_per_half && out_c + out_base < data.out_chans;
         out_c++) {
      int half_idx = out_c / data.chans_per_half;
      int half_off = out_c % data.chans_per_half;
      for (int col = 0; col < data.output_h; col++) {
        for (int row = 0; row < data.output_w; row++) {
          int noise_idx =
              half_idx * data.pack_num + half_off * data.image_size +
              col * data.stride_w * data.image_w + row * data.stride_h;
          int share_idx = col * data.output_w + row;
          noise[ct_idx][noise_idx] = secret_share[out_base + out_c][share_idx];
        }
      }
    }
    Plaintext tmp = cc->MakePackedPlaintext(noise[ct_idx]);
    enc_noise[ct_idx] = cc->Encrypt(pk, tmp);
    // batch_encoder.encode(noise[ct_idx], tmp);
    // encryptor.encrypt(tmp, enc_noise[ct_idx]);
    // evaluator.mod_switch_to_next_inplace(enc_noise[ct_idx]); // UNDO AFTER TESTING NOISE
  }
  return enc_noise;
}

/* preprocess_image_OP
*/
// Preprocesses the input image for output packing. Ciphertext is packed in
// RowMajor order. In this mode simply pack all the input channels as tightly as
// possible where each channel is padded to the nearest of two
vector<vector<int64_t>> preprocess_image_OP(vector<vector<vector<int64_t>>> &image, ConvMetadata data) {
  vector<vector<int64_t>> ct(data.inp_ct,
                              vector<int64_t>(data.slot_count, 0));
  int inp_c = 0;
  for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
    int inp_c_limit = (ct_idx + 1) * 2 * data.chans_per_half;
    for (; inp_c < data.inp_chans && inp_c < inp_c_limit; inp_c++) {
      // Calculate which half of ciphertext the output channel
      // falls in and the offest from that half,
      int half_idx = (inp_c % (2 * data.chans_per_half)) / data.chans_per_half;
      int half_off = inp_c % data.chans_per_half;
      for (int row = 0; row < data.image_h; row++) {
        for (int col = 0; col < data.image_w; col++) {
          int idx = half_idx * data.pack_num + half_off * data.image_size +
                    row * data.image_w + col;
          ct[ct_idx][idx] = image[inp_c].at(row).at(col);
        }
      }
    }
  }
  return ct;
}

/* filter_rotations
*/
// Evaluates the filter rotations necessary to convole an input. Essentially,
// think about placing the filter in the top left corner of the padded image and
// sliding the image over the filter in such a way that we capture which
// elements of filter multiply with which elements of the image. We account for
// the zero padding by zero-puncturing the masks. This function can evaluate
// plaintexts and ciphertexts.
vector<vector<Ciphertext<DCRTPoly>>> filter_rotations(vector<Ciphertext<DCRTPoly>> &input,
                                            const ConvMetadata &data,
                                            CryptoContext<DCRTPoly> &cc
                                            , vector<int> &counts
                                            ) {
  int num_mul{}, num_add{}, num_mod{}, num_rot{}, num_skip{};
  vector<vector<Ciphertext<DCRTPoly>>> rotations(input.size(),
                                       vector<Ciphertext<DCRTPoly>>(data.filter_size));
  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;

  // This tells us how many filters fit on a single row of the padded image
  int f_per_row = data.image_w + pad_w - data.filter_w + 1;

  // This offset calculates rotations needed to bring filter from top left
  // corner of image to the top left corner of padded image
  int offset = f_per_row * data.pad_t + data.pad_l;

  // For each element of the filter, rotate the padded image s.t. the top
  // left position always contains the first element of the image it touches
// #pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for (int f = 0; f < data.filter_size; f++) {
    for (size_t ct_idx = 0; ct_idx < input.size(); ct_idx++) {
      int f_row = f / data.filter_w;
      int f_col = f % data.filter_w;
      int row_offset = f_row * data.image_w - offset;
      int rot_amt = row_offset + f_col;
      int idx = f_row * data.filter_w + f_col;
      rotations[ct_idx][idx] = rotate_arb2(input[ct_idx], rot_amt, cc); num_rot++;
      // evaluator->rotate_rows(input[ct_idx], rot_amt, *gal_keys,
      //                        rotations[ct_idx][idx]); num_rot++;
    }
  }
  counts[0] += num_mul;
  counts[1] += num_add;
  counts[2] += num_mod;
  counts[3] += num_rot;
  // printf("[filter_rotations] num_mul: %3i, num_add: %3i, num_mod: %3i, num_rot: %3i \n", num_mul, num_add, num_mod, num_rot);
  
  if(data.print_cnts){
    cout << "[filter_rotations] HE operation counts: " << endl;
    cout << "+ mul: " << num_mul << endl;
    cout << "+ add: " << num_add << endl;
    cout << "+ mod: " << num_mod << endl;
    cout << "+ rot: " << num_rot << endl;
  }

  return rotations;
}

/* HE_encode_input
*/
// Encodes the given input image into a plaintexts
vector<Plaintext> HE_encode_input(vector<vector<int64_t>> &pt,
                              const ConvMetadata &data,
                              CryptoContext<DCRTPoly> &cc) {
  vector<Plaintext> pts(data.inp_ct);
// #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (size_t pt_idx = 0; pt_idx < data.inp_ct; pt_idx++) {
    pts[pt_idx] = cc->MakePackedPlaintext(pt[pt_idx]);
  }
  return pts;
}

/* HE_encrypt
*/
// Encrypts the given input image
vector<Ciphertext<DCRTPoly>> HE_encrypt(vector<vector<int64_t>> &pt,
                              const ConvMetadata &data, 
                              CryptoContext<DCRTPoly> &cc,
                              PublicKey<DCRTPoly> &pk) {
  
  auto pts = HE_encode_input(pt, data, cc);
  
  vector<Ciphertext<DCRTPoly>> ct(pts.size());
// #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (size_t ct_idx = 0; ct_idx < pt.size(); ct_idx++) {
    ct[ct_idx] = cc->Encrypt(pk, pts[ct_idx]);
  }
  return ct;
}

/* HE_preprocess_filters_OP
*/
// Creates filter masks for an image input that has been output packed.
vector<vector<vector<vector<int64_t>>>> HE_preprocess_filters_OP(
    vector<vector<vector<vector<int64_t>>>> &filters, const ConvMetadata &data) {
  // Mask is convolutions x cts per convolution x mask size
  vector<vector<vector<vector<int64_t>>>> clear_masks(
      data.convs, vector<vector<vector<int64_t>>>(
                      data.inp_ct, vector<vector<int64_t>>(data.filter_size)));
  // Since a half in a permutation may have a variable number of rotations we
  // use this index to track where we are at in the masks tensor
  // Build each half permutation as well as it's inward rotations
// #pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for (int perm = 0; perm < data.half_perms; perm += 2) {
    for (int rot = 0; rot < data.half_rots; rot++) {
      int conv_idx = perm * data.half_rots;
      for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        // The output channel the current ct starts from
        // int out_base = (((perm/2) + ct_idx)*2*data.chans_per_half) %
        // data.out_mod;
        int out_base = (perm * data.chans_per_half) % data.out_mod;
        // Generate all inward rotations of each half -- half_rots loop
        for (int f = 0; f < data.filter_size; f++) {
          vector<vector<int64_t>> masks(2,
                                         vector<int64_t>(data.slot_count, 0));
          for (int half_idx = 0; half_idx < 2; half_idx++) {
            int inp_base = (2 * ct_idx + half_idx) * data.chans_per_half;
            for (int chan = 0; chan < data.chans_per_half &&
                               (chan + inp_base) < data.inp_chans;
                 chan++) {
              // Pull the value of this mask
              int f_w = f % data.filter_w;
              int f_h = f / data.filter_w;
              // Set the coefficients of this channel for both
              // permutations
              int64_t val, val2;
              int out_idx, out_idx2;

              int offset = neg_mod(chan - rot, (int64_t)data.chans_per_half);
              if (half_idx) {
                // If out_halves < 1 we may repeat within a
                // ciphertext
                // TODO: Add the log optimization for this case
                if (data.out_halves > 1)
                  out_idx = offset + out_base + data.chans_per_half;
                else
                  out_idx = offset + out_base;
                out_idx2 = offset + out_base;
              } else {
                out_idx = offset + out_base;
                out_idx2 = offset + out_base + data.chans_per_half;
              }
              val = (out_idx < data.out_chans)
                        ? filters[out_idx][inp_base + chan].at(f_h).at(f_w)
                        : 0;
              val2 = (out_idx2 < data.out_chans)
                         ? filters[out_idx2][inp_base + chan].at(f_h).at(f_w)
                         : 0;
              // Iterate through the whole image and figure out which
              // values the filter value touches - this is the same
              // as for input packing
              for (int curr_h = 0; curr_h < data.image_h;
                   curr_h += data.stride_h) {
                for (int curr_w = 0; curr_w < data.image_w;
                     curr_w += data.stride_w) {
                  // curr_h and curr_w simulate the current top-left position of
                  // the filter. This detects whether the filter would fit over
                  // this section. If it's out-of-bounds we set the mask index
                  // to 0
                  bool zero = ((curr_w + f_w) < data.pad_l) ||
                              ((curr_w + f_w) >= (data.image_w + data.pad_l)) ||
                              ((curr_h + f_h) < data.pad_t) ||
                              ((curr_h + f_h) >= (data.image_h + data.pad_l));
                  // Calculate which half of ciphertext the output channel
                  // falls in and the offest from that half,
                  int idx = half_idx * data.pack_num + chan * data.image_size +
                            curr_h * data.image_w + curr_w;
                  // Add both values to appropiate permutations
                  masks[0][idx] = zero ? 0 : val;
                  if (data.half_perms > 1) {
                    masks[1][idx] = zero ? 0 : val2;
                  }
                }
              }
            }
          }
          clear_masks[conv_idx + rot][ct_idx][f] = masks[0];
          // batch_encoder.encode(masks[0],
          //                      encoded_masks[conv_idx + rot][ct_idx][f]);
          if (data.half_perms > 1) {
            clear_masks[conv_idx + data.half_rots + rot][ct_idx][f] = masks[1];
            // batch_encoder.encode(
            //     masks[1],
            //     encoded_masks[conv_idx + data.half_rots + rot][ct_idx][f]);
          }
        }
      }
    }
  }
  return clear_masks;
}

/* HE_preprocess_filters_OP
*/
// Creates filter masks for an image input that has been output packed.
vector<vector<vector<Plaintext>>> HE_preprocess_filters_OP(
    vector<vector<vector<vector<int64_t>>>> &filters, const ConvMetadata &data, CryptoContext<DCRTPoly> &cc) {
  // Mask is convolutions x cts per convolution x mask size

  auto clear_masks = HE_preprocess_filters_OP(filters, data);

  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;

  // This tells us how many filters fit on a single row of the padded image
  int f_per_row = data.image_w + pad_w - data.filter_w + 1;

  // This offset calculates rotations needed to bring filter from top left
  // corner of image to the top left corner of padded image
  int offset = f_per_row * data.pad_t + data.pad_l;

  vector<vector<vector<Plaintext>>> encoded_masks(
      data.convs, vector<vector<Plaintext>>(
                      data.inp_ct, vector<Plaintext>(data.filter_size)));
  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    for(int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++){
      for(int f = 0; f < data.filter_size; f++){
        vector<int64_t> _mask;
        if(data.use_heliks){

        int f_row = f / data.filter_w;
        int f_col = f % data.filter_w;
        int row_offset = f_row * data.image_w - offset;
        int rot_amt = row_offset + f_col;
        int idx = f_row * data.filter_w + f_col;

        _mask = clear_masks[conv_idx][ct_idx][idx];
        
        auto _mask_rot = {slice1D(_mask, 0, data.slot_count/2 - 1),
                          slice1D(_mask, data.slot_count/2,   0  )};
        _mask = rowEncode2D(rotate2D(_mask_rot, 0, -rot_amt));
        } else {
          _mask = clear_masks[conv_idx][ct_idx][f];
        }
        encoded_masks[conv_idx][ct_idx][f] = cc->MakePackedPlaintext(_mask);
      }
    }
  }
  return encoded_masks;
}

/* HE_output_rotations
*/
// Takes the result of an output-packed convolution, and rotates + adds all the
// ciphertexts to get a tightly packed output
vector<Ciphertext<DCRTPoly>> HE_output_rotations(vector<Ciphertext<DCRTPoly>> &convs,
                                       const ConvMetadata &data,
                                       CryptoContext<DCRTPoly> &cc,
                                       Ciphertext<DCRTPoly> &zero,
                                       vector<Ciphertext<DCRTPoly>> &enc_noise
                                      , vector<int> &counts
                                      ) {
  int num_mul{}, num_add{}, num_mod{}, num_rot{}, num_skip{};
  vector<Ciphertext<DCRTPoly>> partials(data.half_perms);
  Ciphertext<DCRTPoly> zero_next_level = zero;
  // evaluator.mod_switch_to_next_inplace(zero_next_level); // UNDO AFTER TESTING!!!
  // Init the result vector to all 0
  vector<Ciphertext<DCRTPoly>> result(data.out_ct);
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    result[ct_idx] = zero_next_level;
  }

  // For each half perm, add up all the inside channels of each half
// #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int perm = 0; perm < data.half_perms; perm += 2) {
    partials[perm] = zero_next_level;
    if (data.half_perms > 1) partials[perm + 1] = zero_next_level;
    // The output channel the current ct starts from
    int total_rots = data.half_rots;
    for (int in_rot = 0; in_rot < total_rots; in_rot++) {
      int conv_idx = perm * data.half_rots + in_rot;
      int rot_amt;
      rot_amt =
          -neg_mod(-in_rot, (int64_t)data.chans_per_half) * data.image_size;

      // evaluator.rotate_rows_inplace(convs[conv_idx], rot_amt, gal_keys); num_rot++;
      convs[conv_idx] = rotate_arb2(convs[conv_idx], rot_amt, cc); num_rot++;

      // evaluator.add_inplace(partials[perm], convs[conv_idx]); num_add++;
      partials[perm] = cc->EvalAdd(partials[perm], convs[conv_idx]); num_add++;

      // Do the same for the column swap if it exists
      if (data.half_perms > 1) {
        // evaluator.rotate_rows_inplace(convs[conv_idx + data.half_rots], rot_amt,
        //                               gal_keys); num_rot++;
        convs[conv_idx + data.half_rots] = 
          rotate_arb2(convs[conv_idx + data.half_rots], rot_amt, cc); num_rot++;
        // evaluator.add_inplace(partials[perm + 1],
        //                       convs[conv_idx + data.half_rots]); num_add++;
        partials[perm + 1] = cc->EvalAdd(partials[perm + 1],
                              convs[conv_idx + data.half_rots]); num_add++;
      }
    }
    // The correct index for the correct ciphertext in the final output
    int out_idx = (perm / 2) % data.out_ct;
    if (perm == 0) {
      // The first set of convolutions is aligned correctly
      // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
      result[out_idx] = cc->EvalAdd(result[out_idx], partials[perm]); num_add++;
      //
      if (data.out_halves == 1 && data.inp_halves > 1) {
        // If the output fits in a single half but the input
        // doesn't, add the two columns
        // evaluator.rotate_columns_inplace(partials[perm], gal_keys); num_rot++;
        partials[perm] = rotate_arb2(partials[perm], data.slot_count/2, cc); num_rot++;

        // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
        result[out_idx] = cc->EvalAdd(result[out_idx], partials[perm]); num_add++;
      }
      //
      // Do the same for column swap if exists and we aren't on a repeat
      if (data.half_perms > 1) {
        // evaluator.rotate_columns_inplace(partials[perm + 1], gal_keys); num_rot++;
        partials[perm + 1] = rotate_arb2(partials[perm + 1], data.slot_count/2, cc); num_rot++;
        // evaluator.add_inplace(result[out_idx], partials[perm + 1]); num_add++;
        result[out_idx] = cc->EvalAdd(result[out_idx], partials[perm + 1]); num_add++;
      }
    } else {
      // Rotate the output ciphertexts by one and add
      // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
      result[out_idx] = cc->EvalAdd(result[out_idx], partials[perm]); num_add++;
      // If we're on a tight half we add both halves together and
      // don't look at the column flip
      if (data.half_perms > 1) {
        // evaluator.rotate_columns_inplace(partials[perm + 1], gal_keys); num_rot++;
        partials[perm + 1] = rotate_arb2(partials[perm + 1], data.slot_count/2, cc); num_rot++;
        // evaluator.add_inplace(result[out_idx], partials[perm + 1]); num_add++;
        result[out_idx] = cc->EvalAdd(result[out_idx], partials[perm + 1]); num_add++;
      }
    }
  }
  //// Add the noise vector to remove any leakage
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    // evaluator.add_inplace(result[ct_idx], enc_noise[ct_idx]); num_add++;
    result[ct_idx] = cc->EvalAdd(result[ct_idx], enc_noise[ct_idx]); num_add++;
    // evaluator.mod_switch_to_next_inplace(result[ct_idx]);
  }
  counts[0] += num_mul;
  counts[1] += num_add;
  counts[2] += num_mod;
  counts[3] += num_rot;
  // printf("[HE_output_rotations] num_mul: %3i, num_add: %3i, num_mod: %3i, num_rot: %3i \n", num_mul, num_add, num_mod, num_rot);
  if(data.print_cnts){
    cout << "[HE_output_rotations] HE operation counts: " << endl;
    cout << "+ mul: " << num_mul << endl;
    cout << "+ add: " << num_add << endl;
    cout << "+ mod: " << num_mod << endl;
    cout << "+ rot: " << num_rot << endl;
  }
  return result;
}

/* HE_decrypt
*/
// Decrypts and reshapes convolution result
int64_t **HE_decrypt(vector<Ciphertext<DCRTPoly>> &enc_result, const ConvMetadata &data,
                      CryptoContext<DCRTPoly> &cc, PrivateKey<DCRTPoly> &sk) {
  // Decrypt ciphertext
  vector<vector<int64_t>> result(data.out_ct);

  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    Plaintext tmp;
    // decryptor.decrypt(enc_result[ct_idx], tmp);
    cc->Decrypt(sk, enc_result[ct_idx], &tmp);
    // batch_encoder.decode(tmp, result[ct_idx]);
    result[ct_idx] = tmp->GetPackedValue();
  }

  int64_t **final_result = new int64_t *[data.out_chans];
  // Extract correct values to reshape
  for (int out_c = 0; out_c < data.out_chans; out_c++) {
    int ct_idx = out_c / (2 * data.chans_per_half);
    int half_idx = (out_c % (2 * data.chans_per_half)) / data.chans_per_half;
    int half_off = out_c % data.chans_per_half;
    // Depending on the padding type and stride the output values won't be
    // lined up so extract them into a temporary channel before placing
    // them in resultant vector<vector<vector<int64_t>>>
    final_result[out_c] = new int64_t[data.output_h * data.output_w];
    for (int col = 0; col < data.output_h; col++) {
      for (int row = 0; row < data.output_w; row++) {
        int idx = half_idx * data.pack_num + half_off * data.image_size +
                  col * data.stride_w * data.image_w + row * data.stride_h;
        final_result[out_c][col * data.output_w + row] = result[ct_idx][idx];
      }
    }
  }
  return final_result;
}

void ConvField::gen_context(int ring_dim) {
  
  CCParams<CryptoContextBGVRNS> parameters;
  parameters.SetPlaintextModulus(data.prime_mod);
  parameters.SetMultiplicativeDepth(data.mult_depth);
  parameters.SetRingDim(ring_dim);
  data.slot_count = parameters.GetRingDim();

  cc = GenCryptoContext(parameters);

  // Enable the features that you wish to use.
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  cc->Enable(ADVANCEDSHE);

  keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc->EvalSumKeyGen(keys.secretKey);
  
  vector<int32_t> indexList;
  int32_t index = 1;

  while (index <= data.slot_count/2) {
    indexList.push_back(index);
    indexList.push_back(-1 * index);
    index *= 2;
  }
  
  cc->EvalRotateKeyGen(keys.secretKey, indexList);

  vector<int64_t> zero_vec(data.slot_count, 0);
  Plaintext zero_pt = cc->MakePackedPlaintext(zero_vec);
  zero = cc->Encrypt(keys.publicKey, zero_pt);
}

/* ConvField
*/
ConvField::ConvField(int ring_dim, bool use_heliks) {

  data.prime_mod = 65537;
  data.min = -1; // -(prime_mod / 2);
  data.max = +1; // (prime_mod / 2) - 1;
  data.use_heliks = use_heliks;
  data.mult_depth = use_heliks ? 1: 2;
  
  gen_context(ring_dim);
}

/* ConvField
ConvField::ConvField(int party, sci::NetIO *io, vector<int> CoeffModBits, int slot_count, bool verbose){
  this->party = party;
  this->io = io;

  this->slot_count = slot_count;
  generate_new_keys(party, io, slot_count, CoeffModBits, context[0], encryptor[0],
                    decryptor[0], evaluator[0], encoder[0], gal_keys[0],
                    zero[0], verbose);
}
*/

/* ConvField
ConvField::~ConvField() {
  for (int i = 0; i < 2; i++) {
    if (context[i]) {
      free_keys(party, encryptor[i], decryptor[i], evaluator[i], encoder[i],
                gal_keys[i], zero[i]);
    }
  }
}
*/

/* configure
*/
void ConvField::configure(bool verbose) {
  // If using Output packing we pad image_size to the nearest power of 2
  data.image_size = ceil2Pow(data.image_h * data.image_w);
  data.filter_size = data.filter_h * data.filter_w;

  assert(data.out_chans > 0 && data.inp_chans > 0);
  // Doesn't currently support a channel being larger than a half ciphertext
  assert(data.image_size <= (data.slot_count / 2));

  data.pack_num = data.slot_count / 2;
  data.chans_per_half = data.pack_num / data.image_size;
  data.inp_ct = ceil((float)data.inp_chans / (2 * data.chans_per_half));
  data.out_ct = ceil((float)data.out_chans / (2 * data.chans_per_half));

  data.inp_halves = ceil((float)data.inp_chans / data.chans_per_half);
  data.out_halves = ceil((float)data.out_chans / data.chans_per_half);

  // The modulo is calculated per ciphertext instead of per half since we
  // should never have the last out_half wrap around to the first in the
  // same ciphertext
  data.out_mod = data.out_ct * 2 * data.chans_per_half;

  data.half_perms = (data.out_halves % 2 != 0 && data.out_halves > 1)
                        ? data.out_halves + 1
                        : data.out_halves;
                        
  data.half_rots =
      (data.inp_halves > 1 || data.out_halves > 1)
          ? data.chans_per_half
          : max(data.chans_per_half, max(data.out_chans, data.inp_chans));

  data.convs = data.half_perms * data.half_rots;

  // data.convs = data.out_halves * data.chans_per_half;

  // data.convs = data.out_chans

  data.output_h = 1 + (data.image_h + data.pad_t + data.pad_b - data.filter_h) /
                          data.stride_h;
  data.output_w = 1 + (data.image_w + data.pad_l + data.pad_r - data.filter_w) /
                          data.stride_w;

  if (verbose) {
    cout << "[configure] ";
    printConvMetaData(data);
  }
}

/* ideal_functionality
*/
vector<vector<vector<int64_t>>> 
  ConvField::ideal_functionality(vector<vector<vector<int64_t>>> &image, 
      const vector<vector<vector<vector<int64_t>>>> &filters) {
  int channels = data.inp_chans;
  int filter_h = data.filter_h;
  int filter_w = data.filter_w;
  int output_h = data.output_h;
  int output_w = data.output_w;
  int stride_h = data.stride_h;
  int stride_w = data.stride_w;
  int pad_t = data.pad_t;
  int pad_b = data.pad_b;
  int pad_l = data.pad_l;
  int pad_r = data.pad_r;

  // Pad image
  vector<vector<vector<int64_t>>> p_image(channels);
  for(int c = 0; c < channels; c++){
    p_image.at(c) = pad2D(image.at(c), pad_t, pad_b, pad_l, pad_r);
  }

  // Perform convolution
  vector<vector<vector<int64_t>>> result;

  for(int i = 0; i < filters.size(); i++){
    vector<vector<vector<int64_t>>> filter = filters.at(i);
    vector<vector<int64_t>> out(output_h, vector<int64_t>(output_w, 0));
    for(int c = 0; c < channels; c++){
      for(int h = 0; h < output_h; h++){
        for(int w = 0; w < output_w; w++){
          for(int fh = 0; fh < filter_h; fh++){
            for(int fw = 0; fw < filter_w; fw++){
              out.at(h).at(w) += p_image.at(c).at(h*stride_h+fh).at(w*stride_w+fw) * filter.at(c).at(fh).at(fw);
            }
          }
        }
      }
    }
    result.push_back(out);
  }

  // auto p_image = pad_image(data, image);
  // const int col_height = filter_h * filter_w * channels;
  // const int col_width = output_h * output_w;
  // vector<vector<int64_t>> image_col(col_height, vector<int64_t>(col_width));
  // i2c(p_image, image_col, data.filter_h, data.filter_w, data.stride_h,
  //     data.stride_w, data.output_h, data.output_w);

  // // For each filter, flatten it into and multiply with image_col
  // vector<vector<vector<int64_t>>> result;
  // for (auto &filter : filters) {
  //   vector<vector<int64_t>> filter_col(1, vector<int64_t>(col_height));
  //   // Use im2col with a filter size 1x1 to translate
  //   i2c(filter, filter_col, 1, 1, 1, 1, filter_h, filter_w);

  //   // vector<vector<int64_t>> tmp = filter_col * image_col;

  //   vector<vector<int64_t>> tmp(1, vector<int64_t>(col_width));

  //   vector<vector<int64_t>> out(output_h);
  //   for(int i = 0; i < output_h; i++){
  //       auto start = tmp.at(0).begin() + i*output_w;
  //       auto end   = start + output_w;
  //       out.at(i).insert(out.at(i).begin(), start, end);
  //   }

  //   // Reshape result of multiplication to the right size
  //   // SEAL stores matrices in RowMajor form
  //   result.push_back(out);
  // }
  return result;
}

/* verify
*/
void ConvField::verify(int H, int W, int CI, int CO, vector<vector<vector<int64_t>>> &image,
                       const vector<vector<vector<vector<int64_t>>>> *filters,
                       const vector<vector<vector<vector<int64_t>>>> &outArr) {
/*
  int newH = outArr[0].size();
  int newW = outArr[0][0].size();

  if (party == BOB) {
    for (int i = 0; i < CI; i++) {
      io->send_data(image[i].data(), H * W * sizeof(int64_t));
    }
    for (int i = 0; i < newH; i++) {
      for (int j = 0; j < newW; j++) {
        io->send_data(outArr[0][i][j].data(),
                      sizeof(int64_t) * data.out_chans);
      }
    }
  } else  // party == ALICE
  {
    vector<vector<vector<int64_t>>> image_0(CI);  // = new vector<vector<int64_t>>[CI];
    for (int i = 0; i < CI; i++) {
      image_0[i].resize(H, W);
      io->recv_data(image_0[i].data(), H * W * sizeof(int64_t));
    }

    for (int i = 0; i < CI; i++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          image[i].at(h).at(w) = (image[i].at(h).at(w) + image_0[i].at(h).at(w)) % prime_mod;
        }
      }
    }

    vector<vector<vector<int64_t>>> result = ideal_functionality(image, *filters);
    vector<vector<vector<vector<int64_t>>>> outArr_0;
    outArr_0.resize(1);
    outArr_0[0].resize(newH);
    for (int i = 0; i < newH; i++) {
      outArr_0[0][i].resize(newW);
      for (int j = 0; j < newW; j++) {
        outArr_0[0][i][j].resize(CO);
        io->recv_data(outArr_0[0][i][j].data(), sizeof(int64_t) * CO);
      }
    }
    for (int i = 0; i < newH; i++) {
      for (int j = 0; j < newW; j++) {
        for (int k = 0; k < CO; k++) {
          outArr_0[0][i][j][k] =
              (outArr_0[0][i][j][k] + outArr[0][i][j][k]) % prime_mod;
        }
      }
    }

    bool pass = true;
    for (int i = 0; i < CO; i++) {
      for (int j = 0; j < newH; j++) {
        for (int k = 0; k < newW; k++) {
          if ((int64_t)outArr_0[0][j][k][i] !=
              neg_mod(result[i](j, k), (int64_t)prime_mod)) {
            pass = false;
          }
        }
      }
    }

    if (pass) {
      cout << GREEN << "[Server] Successful Operation" << RESET << endl;
    } else {
      cout << RED << "[Server] Failed Operation" << RESET << endl;
      cout << RED << "WARNING: The implementation assumes that the computation"
           << endl;
      cout << "performed locally by the server (on the model and its input "
              "share)"
           << endl;
      cout << "fits in a 64-bit integer. The failed operation could be a result"
           << endl;
      cout << "of overflowing the bound." << RESET << endl;
    }
  }

*/
}

/* printChannelDims
*/
void printChannelDims(vector<vector<int64_t>> channel){
  printf("vector<vector<int64_t>> dimensions: (%i, %i), size: %i\n",
         (int) channel.size(), (int) channel.at(0).size(),
         (int) channel.size());
}

/* printImageDims
*/
void printImageDims(vector<vector<vector<int64_t>>> image){
  printf("vector<vector<vector<int64_t>>> dimensions: (%i, %i, %i), vector<vector<int64_t>> size: %i\n",
         (int) image[0].size(), (int) image[0].at(0).size(),
         (int) image.size(), (int) image[0].size());
}

/* printImage
void printImage(vector<vector<vector<int64_t>>> image, int num_digits = 2, bool no_exp = false){
  for (int _ch{}; _ch < image.size(); _ch++){
    for (int _row{}; _row < image[_ch].size(); _row++){
      for (int _col{}; _col < image[_ch].at(0).size(); _col++){
        if(no_exp) printf("%.*d ", num_digits, (int64_t) image[_ch].at(_row).at(_col));
        else printf("%.*e ", num_digits, (double) image[_ch].at(_row).at(_col));
      }
      cout << endl;
    }
    cout << endl;
  }
}
*/

/* HE_preprocess_filters_OP_MR
*/
// Pre-processes the filter into Plaintexts such that multiply can be performed
// before the rotation in fc_online.
vector<vector<vector<Plaintext>>> HE_preprocess_filters_OP_MR(
    vector<vector<vector<vector<int64_t>>>> &filters, const ConvMetadata &data, 
    CryptoContext<DCRTPoly> &cc
    , vector<vector<vector<int>>> &rot_amts
    , vector<map<int, vector<vector<int>>>> &rot_maps
    ) {

  // Mask is convolutions x cts per convolution x mask size
  vector<vector<vector<Plaintext>>> encoded_masks(
      data.convs, vector<vector<Plaintext>>(
                      data.inp_ct, vector<Plaintext>(data.filter_size)));

  auto clear_masks = HE_preprocess_filters_OP(filters, data); 

  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;

  // This tells us how many filters fit on a single row of the padded image
  int f_per_row = data.image_w + pad_w - data.filter_w + 1;

  // This offset calculates rotations needed to bring filter from top left
  // corner of image to the top left corner of padded image
  int offset = f_per_row * data.pad_t + data.pad_l;

  vector<vector<vector<vector<int64_t>>>> clear_masks_rot(
      data.convs, vector<vector<vector<int64_t>>>(
                      data.inp_ct, vector<vector<int64_t>>(data.filter_size)));

  rot_amts = vector<vector<vector<int>>> (data.convs, vector<vector<int>>(
                                data.inp_ct, vector<int>(data.filter_size)));
  
// #pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    for(int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++){
      for(int f = 0; f< data.filter_size; f++){

        int f_row = f / data.filter_w;
        int f_col = f % data.filter_w;
        int row_offset = f_row * data.image_w - offset;
        int rot_amt = row_offset + f_col;
        int idx = f_row * data.filter_w + f_col;

        rot_amts.at(conv_idx).at(ct_idx).at(idx) = rot_amt;

        vector<int64_t> _mask = clear_masks[conv_idx][ct_idx][idx];
        
        vector<vector<int64_t>> _mask_rot = {slice1D(_mask, 0, data.slot_count/2 - 1),
                                              slice1D(_mask, data.slot_count/2,   0  )};

        clear_masks_rot[conv_idx][ct_idx][idx] = rowEncode2D(
                                                  rotate2D(_mask_rot, 0, -rot_amt));
                                                  
        // batch_encoder.encode(
        //           clear_masks_rot[conv_idx][ct_idx][f],
        //           encoded_masks[conv_idx][ct_idx][f]);
        encoded_masks[conv_idx][ct_idx][f] = cc->MakePackedPlaintext(clear_masks_rot[conv_idx][ct_idx][idx]);

        // if(!encoded_masks[conv_idx][ct_idx][f].is_zero()){
          rot_maps[conv_idx][rot_amts.at(conv_idx).at(ct_idx).at(f)]
                                            .push_back(vector<int>{conv_idx, ct_idx, f});
        // }
      }
    }
  }
  return encoded_masks;
}

/* HE_preprocess_filters_NTT_MR
// Same as HE_preprocess_filters_OP_MR but transforms the plaintexts to NTT
// before returning the results. 
vector<vector<vector<Plaintext>>> HE_preprocess_filters_NTT_MR(
    vector<vector<vector<vector<int64_t>>>> &filters, const ConvMetadata &data, BatchEncoder &batch_encoder
    , Evaluator &evaluator
    , parms_id_type parms_id
    , vector<vector<vector<int>>> &rot_amts
    , vector<map<int, vector<vector<int>>>> &rot_maps
    ) {

  // Mask is convolutions x cts per convolution x mask size
  vector<vector<vector<Plaintext>>> encoded_masks(
      data.convs, vector<vector<Plaintext>>(
                      data.inp_ct, vector<Plaintext>(data.filter_size)));

  vector<vector<vector<vector<int64_t>>>> clear_masks(
      data.convs, vector<vector<vector<int64_t>>>(
                      data.inp_ct, vector<vector<int64_t>>(data.filter_size)));

  // Since a half in a permutation may have a variable number of rotations we
  // use this index to track where we are at in the masks tensor
  // Build each half permutation as well as it's inward rotations
#pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for (int perm = 0; perm < data.half_perms; perm += 2) {
    for (int rot = 0; rot < data.half_rots; rot++) {
      int conv_idx = perm * data.half_rots;
      for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
        // The output channel the current ct starts from
        // int out_base = (((perm/2) + ct_idx)*2*data.chans_per_half) %
        // data.out_mod;
        int out_base = (perm * data.chans_per_half) % data.out_mod;
        // Generate all inward rotations of each half -- half_rots loop
        for (int f = 0; f < data.filter_size; f++) {
          vector<vector<int64_t>> masks(2,
                                         vector<int64_t>(data.slot_count, 0));
          for (int half_idx = 0; half_idx < 2; half_idx++) {
            int inp_base = (2 * ct_idx + half_idx) * data.chans_per_half;
            for (int chan = 0; chan < data.chans_per_half &&
                               (chan + inp_base) < data.inp_chans;
                 chan++) {
              // Pull the value of this mask
              int f_w = f % data.filter_w;
              int f_h = f / data.filter_w;
              // Set the coefficients of this channel for both
              // permutations
              int64_t val, val2;
              int out_idx, out_idx2;

              int offset = neg_mod(chan - rot, (int64_t)data.chans_per_half);
              if (half_idx) {
                // If out_halves < 1 we may repeat within a
                // ciphertext
                // TODO: Add the log optimization for this case
                if (data.out_halves > 1)
                  out_idx = offset + out_base + data.chans_per_half;
                else
                  out_idx = offset + out_base;
                out_idx2 = offset + out_base;
              } else {
                out_idx = offset + out_base;
                out_idx2 = offset + out_base + data.chans_per_half;
              }
              val = (out_idx < data.out_chans)
                        ? filters[out_idx][inp_base + chan].at(f_h).at(f_w)
                        : 0;
              val2 = (out_idx2 < data.out_chans)
                         ? filters[out_idx2][inp_base + chan].at(f_h).at(f_w)
                         : 0;
              // Iterate through the whole image and figure out which
              // values the filter value touches - this is the same
              // as for input packing
              for (int curr_h = 0; curr_h < data.image_h;
                   curr_h += data.stride_h) {
                for (int curr_w = 0; curr_w < data.image_w;
                     curr_w += data.stride_w) {
                  // curr_h and curr_w simulate the current top-left position of
                  // the filter. This detects whether the filter would fit over
                  // this section. If it's out-of-bounds we set the mask index
                  // to 0
                  bool zero = ((curr_w + f_w) < data.pad_l) ||
                              ((curr_w + f_w) >= (data.image_w + data.pad_l)) ||
                              ((curr_h + f_h) < data.pad_t) ||
                              ((curr_h + f_h) >= (data.image_h + data.pad_l));
                  // Calculate which half of ciphertext the output channel
                  // falls in and the offest from that half,
                  int idx = half_idx * data.pack_num + chan * data.image_size +
                            curr_h * data.image_w + curr_w;
                  // Add both values to appropiate permutations
                  masks[0][idx] = zero ? 0 : val;
                  if (data.half_perms > 1) {
                    masks[1][idx] = zero ? 0 : val2;
                  }
                }
              }
            }
          }

          clear_masks[conv_idx + rot][ct_idx][f] = masks[0];

          if (data.half_perms > 1) {
            clear_masks[conv_idx + data.half_rots + rot][ct_idx][f] = masks[1];
          }
        }
      }
    }
  }
  
  int pad_h = data.pad_t + data.pad_b;
  int pad_w = data.pad_l + data.pad_r;

  // This tells us how many filters fit on a single row of the padded image
  int f_per_row = data.image_w + pad_w - data.filter_w + 1;

  // This offset calculates rotations needed to bring filter from top left
  // corner of image to the top left corner of padded image
  int offset = f_per_row * data.pad_t + data.pad_l;

  vector<vector<vector<vector<int64_t>>>> clear_masks_rot(
      data.convs, vector<vector<vector<int64_t>>>(
                      data.inp_ct, vector<vector<int64_t>>(data.filter_size)));

  rot_amts = vector<vector<vector<int>>> (data.convs, vector<vector<int>>(
                                data.inp_ct, vector<int>(data.filter_size)));
  
#pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    for(int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++){
      for(int f = 0; f< data.filter_size; f++){

        int f_row = f / data.filter_w;
        int f_col = f % data.filter_w;
        int row_offset = f_row * data.image_w - offset;
        int rot_amt = row_offset + f_col;
        int idx = f_row * data.filter_w + f_col;

        rot_amts.at(conv_idx).at(ct_idx).at(idx) = rot_amt;

        vector<int64_t> _mask = clear_masks[conv_idx][ct_idx][idx];
        
        vector<vector<int64_t>> _mask_rot = {slice1D(_mask, 0, data.slot_count/2 - 1),
                                              slice1D(_mask, data.slot_count/2,   0  )};

       clear_masks_rot[conv_idx][ct_idx][idx] = rowEncode2D(rotate2D(_mask_rot, 0, -rot_amt));
      }
    }
  }
  
#pragma omp parallel for num_threads(num_threads) schedule(static) collapse(2)
  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    for(int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++){
      for(int f = 0; f< data.filter_size; f++){
        batch_encoder.encode(
                clear_masks_rot[conv_idx][ct_idx][f],
                encoded_masks[conv_idx][ct_idx][f]);
        if(!encoded_masks[conv_idx][ct_idx][f].is_zero()){
          evaluator.transform_to_ntt_inplace(
            encoded_masks[conv_idx][ct_idx][f], parms_id);
        }
      }
    }
  }

  rot_maps = vector<map<int, vector<vector<int>>>>(data.convs);

  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    for(int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++){
      for(int f = 0; f< data.filter_size; f++){
        if(!encoded_masks[conv_idx][ct_idx][f].is_zero()){
          int rot_amt = rot_amts.at(conv_idx).at(ct_idx).at(f);
          rot_maps[conv_idx][rot_amt].push_back(vector<int>{conv_idx, ct_idx, f});
        }
      }
    }
  }
  return encoded_masks;
}
*/

/* input_to_ntt
vector<Ciphertext> input_to_ntt(vector<Ciphertext> &input,
                              Evaluator &evaluator){
  vector<Ciphertext> input_ntt(input.size());
  for(int ct_idx = 0; ct_idx < input.size(); ct_idx++){
    evaluator.transform_to_ntt(input.at(ct_idx), input_ntt.at(ct_idx));
  }
  return input_ntt;
}
*/

/* HE_conv_OP
*/
// Performs convolution for an output packed image. Returns the intermediate
// rotation sets
vector<Ciphertext<DCRTPoly>> HE_conv_OP(vector<vector<vector<Plaintext>>> &masks,
                              vector<vector<Ciphertext<DCRTPoly>>> &rotations,
                              const ConvMetadata &data, CryptoContext<DCRTPoly> &cc,
                              Ciphertext<DCRTPoly> &zero
                              , vector<int> &counts
                              ) {
  int num_mul{}, num_add{}, num_mod{}, num_rot{}, num_skip{};
  vector<Ciphertext<DCRTPoly>> result(data.convs);

  // Multiply masks and add for each convolution
// #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int conv_idx = 0; conv_idx < data.convs; conv_idx++) {
    result[conv_idx] = zero;
    for (int ct_idx = 0; ct_idx < data.inp_ct; ct_idx++) {
      for (int f = 0; f < data.filter_size; f++) {
        Ciphertext<DCRTPoly> tmp = cc->EvalMult(rotations[ct_idx][f], masks[conv_idx][ct_idx][f]); 
        num_mul++;
        // if (!masks[conv_idx][ct_idx][f].is_zero()) {

          // evaluator.multiply_plain(rotations[ct_idx][f],
          //                          masks[conv_idx][ct_idx][f], tmp); num_mul++;

        result[conv_idx] = cc->EvalAdd(result[conv_idx], tmp); num_add++;

          // evaluator.add_inplace(result[conv_idx], tmp); num_add++;
        // }
        // else{
        //   num_skip++;
        // }
      }
    }
    // evaluator.mod_switch_to_next_inplace(result[conv_idx]); num_mod++;
  }

  counts[0] += num_mul;
  counts[1] += num_add;
  counts[2] += num_mod;
  counts[3] += num_rot;
  // printf("[HE_conv_OP] num_mul: %3i, num_add: %3i, num_mod: %3i, num_rot: %3i \n", num_mul, num_add, num_mod, num_rot);
  if(data.print_cnts){
    cout << "[HE_conv_OP] HE operation counts: " << endl;
    cout << "+ mul: " << num_mul << endl;
    cout << "+ add: " << num_add << endl;
    cout << "+ mod: " << num_mod << endl;
    cout << "+ rot: " << num_rot << endl;
  }
  return result;
}

/* HE_conv_OP_MR
*/
vector<Ciphertext<DCRTPoly>> HE_conv_OP_MR(vector<vector<vector<Plaintext>>> &masks,
                              const ConvMetadata &data, 
                              CryptoContext<DCRTPoly> &cc,
                              Ciphertext<DCRTPoly> &zero
                              , vector<Ciphertext<DCRTPoly>> &input
                              , vector<vector<vector<int>>> rot_amts
                              , vector<map<int, vector<vector<int>>>> &rot_maps
                              , vector<int> &counts
                              ) {
  int num_mul{}, num_add{}, num_mod{}, num_rot{}, num_skip{};
  vector<Ciphertext<DCRTPoly>> result(data.convs);
  Ciphertext<DCRTPoly> zero_next_level = zero;

  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    result[conv_idx] = zero_next_level;
    int nrot{};
    for(auto item : rot_maps[conv_idx]){
      int rot_amt = item.first;
      auto idcs   = item.second;
      Ciphertext<DCRTPoly> psum = cc->EvalMult(input[idcs[0][1]],
          masks[conv_idx][idcs[0][1]][idcs[0][2]]); num_mul++;
      for(int i = 1; i< idcs.size(); i++){
        Ciphertext<DCRTPoly> tmp = cc->EvalMult(input[idcs[i][1]],
            masks[conv_idx][idcs[i][1]][idcs[i][2]]); num_mul++;
        cc->EvalAddInPlace(psum, tmp); num_add++;
      }
      int rot_1 = nrot % data.filter_w + 
                  (nrot / data.filter_w) * data.image_w; nrot++;
      if(rot_1 != rot_amt) {
        printf("Mismatch! expected rot = %3d,  actual rot = %3d \n", rot_1, rot_amt);
        psum = rotate_arb2(psum, rot_amt, cc); num_rot++;
        cc->EvalAddInPlace(result[conv_idx], psum); num_add++;
      } else {
        psum = rotate_arb2(psum, rot_amt, cc); num_rot++;
        cc->EvalAddInPlace(result[conv_idx], psum); num_add++;
      }
    }
  }

  counts[0] += num_mul;
  counts[1] += num_add;
  counts[2] += num_mod;
  counts[3] += num_rot;
  if(data.print_cnts){
    cout << "[HE_conv_OP_MR] HE operation counts: " << endl;
    cout << "+ mul: " << num_mul << endl;
    cout << "+ add: " << num_add << endl;
    cout << "+ mod: " << num_mod << endl;
    cout << "+ rot: " << num_rot << endl;
  }
  return result;
}

/* HE_conv_NTT_MR
vector<Ciphertext> HE_conv_NTT_MR(vector<vector<vector<Plaintext>>> &masks,
                              const ConvMetadata &data, Evaluator &evaluator,
                              Ciphertext &zero
                              , vector<Ciphertext> &input
                              , GaloisKeys *gal_keys
                              , vector<vector<vector<int>>> rot_amts
                              , vector<map<int, vector<vector<int>>>> &rot_maps
                              , vector<int> &counts
                              , Decryptor &decryptor
                              ) {
  int num_mul{}, num_add{}, num_mod{}, num_rot{};
  vector<Ciphertext> result(data.convs);
  Ciphertext zero_next_level = zero;
  evaluator.mod_switch_to_next_inplace(zero_next_level); num_mod++;
  // Init the result vector to all 0
  vector<Ciphertext> ct(3);

  // convs = 2co/cn
#pragma omp parallel for num_threads(num_threads) schedule(static)
  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    result[conv_idx] = zero_next_level;
    int nrot{};

    // f = fh * fw
    for(auto item : rot_maps[conv_idx]){
      int rot_amt = item.first;
      auto idcs   = item.second;
      // printf("idcs[0]: %i, %i \n", idcs[0][1], idcs[0][2]);
      Ciphertext psum = input[idcs[0][1]];
      evaluator.multiply_plain_inplace(psum,
          masks[conv_idx][idcs[0][1]][idcs[0][2]]); num_mul++;
      
      // ci/cn
      for(int i = 1; i< idcs.size(); i++){
        Ciphertext tmp = input[idcs[i][1]];
        // printf("idcs[%i]: %i, %i \n", i, idcs[i][1], idcs[i][2]);
        evaluator.multiply_plain_inplace(tmp,
            masks[conv_idx][idcs[i][1]][idcs[i][2]]); num_mul++;
        evaluator.add_inplace(psum, tmp); num_add++;
      }

      evaluator.transform_from_ntt_inplace(psum);
      ct[0] = psum;
      evaluator.mod_switch_to_next_inplace(psum);
      ct[1] = psum;

      int rot_1 = nrot % data.filter_w + 
                  (nrot / data.filter_w) * data.image_w; nrot++;
      // printf("conv_idx = %3d, nrot = %d, rot = %3d,  actual rot = %3d \n", conv_idx, nrot, rot_1, rot_amt);

      if(rot_1 != rot_amt) { // TODO: add small rotations
        printf("Mismatch! expected rot = %3d,  actual rot = %3d \n", rot_1, rot_amt);
        evaluator.rotate_rows_inplace(psum, rot_amt, *gal_keys); num_rot++;
        evaluator.add_inplace(result[conv_idx], psum); num_add++;
      } else {
        evaluator.rotate_rows_inplace(psum, rot_amt, *gal_keys); num_rot++;
        evaluator.add_inplace(result[conv_idx], psum); num_add++;
      }
      ct[2] = psum;
    }
  }

  // cout << "Noise Budget before mod switch : ";
  // cout << decryptor.invariant_noise_budget(ct[0]) << " bits" << endl;
  // cout << "Noise Budget after mod switch  : ";
  // cout << decryptor.invariant_noise_budget(ct[1]) << " bits" << endl;
  // cout << "Noise Budget after rotation    : ";
  // cout << decryptor.invariant_noise_budget(ct[2]) << " bits" << endl;

  counts[0] += num_mul;
  counts[1] += num_add;
  counts[2] += num_mod;
  counts[3] += num_rot;
  printf("[HE_conv_OP_MR] num_mul: %3i, num_add: %3i, num_mod: %3i, num_rot: %3i \n", num_mul, num_add, num_mod, num_rot);
  return result;
}
*/

/* HE_conv_heliks
*/
// HELiKs algorithm
vector<Ciphertext<DCRTPoly>> HE_conv_heliks(  vector<Ciphertext<DCRTPoly>> &input
                                  , vector<vector<vector<Plaintext>>> &masks
                                  , CryptoContext<DCRTPoly> &cc
                                  , const ConvMetadata &data
                                  , vector<int> &counts
                                  ) {
  int num_mul{}, num_add{}, num_mod{}, num_rot{}, num_ntt{}, num_int{};

  Plaintext mask;
  vector<Ciphertext<DCRTPoly>> result(data.convs);

  int f_per_row = data.image_w + (data.pad_l + data.pad_r) - data.filter_w + 1;

  int offset = f_per_row * data.pad_t + data.pad_l;

// #pragma omp parallel for num_threads(num_threads) schedule(static)
  for(int conv_idx = 0; conv_idx < data.convs; conv_idx++){
    
    int in_idx = 0, fil_idx = data.filter_size - 1;
    
    Ciphertext<DCRTPoly> conv = cc->EvalMult(input[in_idx], 
                            masks[conv_idx][in_idx][fil_idx]); num_mul++;

    for(int in_idx = 1; in_idx < data.inp_ct; in_idx++){
      
      Ciphertext<DCRTPoly> partial_i = cc->EvalMult(input[in_idx], 
                            masks[conv_idx][in_idx][fil_idx]); num_mul++;

      cc->EvalAddInPlace(conv, partial_i); num_add++;
    }

    cc->ModReduceInPlace(conv); num_mod++;
    num_mod++;

    for(int fil_idx = (data.filter_size - 2); fil_idx >= 0; fil_idx--){

      int rot_steps = ((fil_idx % data.filter_w) == (data.filter_w - 1)) ? data.output_w : 1;
      
      conv = rotate_arb2(conv, rot_steps, cc); num_rot++;

      Ciphertext<DCRTPoly> partial = cc->EvalMult(input[in_idx], 
                            masks[conv_idx][in_idx][fil_idx]); num_mul++;

      for(int in_idx = 1; in_idx < data.inp_ct; in_idx++){
        
        Ciphertext<DCRTPoly> partial_i = cc->EvalMult(input[in_idx], 
                            masks[conv_idx][in_idx][fil_idx]); num_mul++;
        
        cc->EvalAddInPlace(partial, partial_i); num_add++;
      }
      
      cc->ModReduceInPlace(partial); num_mod++;

      cc->EvalAddInPlace(conv, partial); num_add++;
    }

    result[conv_idx] = conv;
  }

  counts[0] += num_mul;
  counts[1] += num_add;
  counts[2] += num_mod;
  counts[3] += num_rot;
  // if(data.print_times)
  // printf("[HE_conv_OP_MR] num_mul: %3i, num_add: %3i, num_mod: %3i, num_rot: %3i \n", num_mul, num_add, num_mod, num_rot);
  
  if(data.print_cnts){
    cout << "[HE_conv_heliks] HE operation counts: " << endl;
    cout << "+ mul: " << num_mul << endl;
    cout << "+ add: " << num_add << endl;
    cout << "+ mod: " << num_mod << endl;
    cout << "+ rot: " << num_rot << endl;
  }
  
  return result;
}

/* HE_output_rotations_MR
*/
vector<Ciphertext<DCRTPoly>> HE_output_rotations_MR(vector<Ciphertext<DCRTPoly>> &convs,
                                       const ConvMetadata &data,
                                       CryptoContext<DCRTPoly> &cc, 
                                       Ciphertext<DCRTPoly> &zero,
                                       vector<Ciphertext<DCRTPoly>> &enc_noise
                                       , vector<int> &counts
                                       ) {
  int num_mul{}, num_add{}, num_mod{}, num_rot{};

  vector<bool> out_idx_flags (data.out_ct, false);

  vector<Ciphertext<DCRTPoly>> partials(data.half_perms);
  Ciphertext<DCRTPoly> zero_next_level = zero;
  // evaluator.mod_switch_to_next_inplace(zero_next_level); num_mod++;
  cc->ModReduceInPlace(zero_next_level); num_mod++;
  // Init the result vector to all 0
  vector<Ciphertext<DCRTPoly>> result(data.out_ct);
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    result[ct_idx] = zero_next_level;
  }

  // For each half perm, add up all the inside channels of each half
  // half_perms = 2co/cn
// #pragma omp parallel for num_threads(num_threads) schedule(static)
  for (int perm = 0; perm < data.half_perms; perm += 2) {
    partials[perm] = zero_next_level;
    if (data.half_perms > 1) partials[perm + 1] = zero_next_level;
    // The output channel the current ct starts from
    // half_rots = cn/2
    int total_rots = data.half_rots;
    for (int in_rot = 0; in_rot < total_rots; in_rot++) {
      int conv_idx = perm * data.half_rots + in_rot;
      int rot_amt;
      rot_amt =
          -neg_mod(-in_rot, (int64_t)data.chans_per_half) * data.image_size;

      if (rot_amt != 0){
        // evaluator.rotate_rows_inplace(convs[conv_idx], rot_amt, gal_keys); num_rot++; 
        convs[conv_idx] = rotate_arb2(convs[conv_idx], rot_amt, cc); num_rot++;
        // cout << "1. perm:" << perm << ", in_rot: " << in_rot  << ", convs idx: " << conv_idx  << ", rot_amt: " << rot_amt << endl;
      }
      if(in_rot == 0) {
        partials[perm] = convs[conv_idx];
      } else {
        // evaluator.add_inplace(partials[perm], convs[conv_idx]); num_add++;
        cc->EvalAddInPlace(partials[perm], convs[conv_idx]); num_add++;
      }
      // evaluator.add_inplace(partials[perm], convs[conv_idx]); num_add++;
      // Do the same for the column swap if it exists
      if (data.half_perms > 1) {
        if (rot_amt != 0){
          // evaluator.rotate_rows_inplace(convs[conv_idx + data.half_rots], rot_amt,
          //                             gal_keys); num_rot++; 
          convs[conv_idx + data.half_rots] = rotate_arb2(convs[conv_idx + data.half_rots], rot_amt, cc); num_rot++;
          // cout << "2. perm:" << perm << ", in_rot: " << in_rot  << ", convs idx: " << conv_idx + data.half_rots << ", rot_amt: " << rot_amt << endl;
        }
        if (in_rot == 0) {
          partials[perm + 1] = convs[conv_idx + data.half_rots];
        } else {
          // evaluator.add_inplace(partials[perm + 1],
          //                       convs[conv_idx + data.half_rots]); num_add++;
          cc->EvalAddInPlace(partials[perm + 1],
                                convs[conv_idx + data.half_rots]); num_add++;
        }
        // evaluator.add_inplace(partials[perm + 1],
        //                       convs[conv_idx + data.half_rots]); num_add++;
      }
    }
    // The correct index for the correct ciphertext in the final output
    int out_idx = (perm / 2) % data.out_ct;
    if (perm == 0) {
      // The first set of convolutions is aligned correctly
      if(!out_idx_flags[out_idx]){
        result[out_idx] = partials[perm];
        out_idx_flags[out_idx] = true;
      } else {
        // cout << "ERROR: out_idx already set" << endl;
        // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
        cc->EvalAddInPlace(result[out_idx], partials[perm]); num_add++;
      }
      // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
      //
      if (data.out_halves == 1 && data.inp_halves > 1) {
        // If the output fits in a single half but the input
        // doesn't, add the two columns
        // evaluator.rotate_columns_inplace(partials[perm], gal_keys); num_rot++;
        partials[perm] = rotate_arb2(partials[perm], data.slot_count/2, cc); num_rot++;
        // cout << "3. perm:" << perm << ", out_idx: " << out_idx << ", rot_cols" << endl;
        // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
        cc->EvalAddInPlace(result[out_idx], partials[perm]); num_add++;
      }
      //
      // Do the same for column swap if exists and we aren't on a repeat
      if (data.half_perms > 1) {
        // evaluator.rotate_columns_inplace(partials[perm + 1], gal_keys); num_rot++; 
        partials[perm + 1] = rotate_arb2(partials[perm + 1], data.slot_count/2, cc); num_rot++;
        // cout << "4. perm:" << perm << ", out_idx: " << out_idx  << ", rot_cols" << endl;
        // evaluator.add_inplace(result[out_idx], partials[perm + 1]); num_add++;
        cc->EvalAddInPlace(result[out_idx], partials[perm + 1]); num_add++;
      }
    } else {
      // Rotate the output ciphertexts by one and add
      if(!out_idx_flags[out_idx]){
        result[out_idx] = partials[perm];
        out_idx_flags[out_idx] = true;
      } else {
        // cout << "ERROR: out_idx already set" << endl;
        // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
        cc->EvalAddInPlace(result[out_idx], partials[perm]); num_add++;
      }
      // evaluator.add_inplace(result[out_idx], partials[perm]); num_add++;
      // If we're on a tight half we add both halves together and
      // don't look at the column flip
      if (data.half_perms > 1) {
        // evaluator.rotate_columns_inplace(partials[perm + 1], gal_keys); num_rot++; 
        partials[perm + 1] = rotate_arb2(partials[perm + 1], data.slot_count/2, cc); num_rot++;
        // cout << "5. perm:" << perm << ", out_idx: " << out_idx  << ", rot_cols" << endl;
        // evaluator.add_inplace(result[out_idx], partials[perm + 1]); num_add++;
        cc->EvalAddInPlace(result[out_idx], partials[perm + 1]); num_add++;
      }
    }
  }
  //// Add the noise vector to remove any leakage
  for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
    // evaluator.add_inplace(result[ct_idx], enc_noise[ct_idx]); num_add++;
    cc->EvalAddInPlace(result[ct_idx], enc_noise[ct_idx]); num_add++;
    // evaluator.mod_switch_to_next_inplace(result[ct_idx]);
  }
  counts[1] += num_add;
  counts[2] += num_mod;
  counts[3] += num_rot;
  // printf("[HE_output_rotations_MR] num_mul: %3i, num_add: %3i, num_mod: %3i, num_rot: %3i \n", num_mul, num_add, num_mod, num_rot);
  
  if(data.print_cnts){
    cout << "[HE_output_rotations_MR] HE operation counts: " << endl;
    cout << "+ mul: " << num_mul << endl;
    cout << "+ add: " << num_add << endl;
    cout << "+ mod: " << num_mod << endl;
    cout << "+ rot: " << num_rot << endl;
  }
  return result;
}

/* HE_preprocess_noise_MR
*/
vector<Ciphertext<DCRTPoly>> HE_preprocess_noise_MR(const int64_t *const *secret_share,
                                       const ConvMetadata &data,
                                       CryptoContext<DCRTPoly> &cc,
                                       PublicKey<DCRTPoly> &public_key) {
  
  vector<Ciphertext<DCRTPoly>> enc_noise = HE_preprocess_noise(secret_share, data, cc, public_key);

  if(data.use_heliks){
    for (int ct_idx = 0; ct_idx < data.out_ct; ct_idx++) {
      
      // evaluator.mod_switch_to_next_inplace(enc_noise[ct_idx]);
      cc->ModReduceInPlace(enc_noise[ct_idx]);
    }
  }
  return enc_noise;
}

/* non_strided_conv
*/
void ConvField::non_strided_conv(int32_t H, int32_t W, int32_t CI, int32_t FH,
                                 int32_t FW, int32_t CO, vector<vector<vector<int64_t>>> *image,
                                 vector<vector<vector<vector<int64_t>>>> *filters,
                                 vector<vector<vector<int64_t>>> &outArr,
                                 vector<int> &counts,
                                 bool verbose) {

  data.image_h = H;
  data.image_w = W;
  data.inp_chans = CI;
  data.out_chans = CO;
  data.filter_h = FH;
  data.filter_w = FW;
  data.pad_t = 0;
  data.pad_b = 0;
  data.pad_l = 0;
  data.pad_r = 0;
  data.stride_h = 1;
  data.stride_w = 1;
  auto _slot_count = max(data.slot_count, 2 * ceil2Pow(H * W));

  if (_slot_count != data.slot_count) {
    gen_context(_slot_count);
  }

  configure(verbose);

  auto sw = StopWatch();
  double prep_mat_time{}, prep_noise_time{}, processing_time{};

    auto pt = preprocess_image_OP(*image, data);
    if (verbose) cout << "[Client] Image preprocessed" << endl;

    auto ct = HE_encrypt(pt, data, cc, keys.publicKey);
    if (verbose) cout << "[Client] Image encrypted and sent" << endl;

    int data_min = data.min;
    int data_max = data.max;

    vector<vector<int64_t>> sec_share_vec = gen2D_UID_int64(data.out_chans, 
                                                  data.output_h * data.output_w, data_min, data_max);

    
    int64_t **secret_share = new int64_t *[CO];
    for (int chan = 0; chan < CO; chan++) {
      secret_share[chan] = sec_share_vec.at(chan).data();
    }

    vector<Ciphertext<DCRTPoly>> noise_ct = HE_preprocess_noise(
        secret_share, data, cc, keys.publicKey);

    if (data.print_cnts) {
      prep_noise_time = sw.lap();
      cout << "[Server] HE_preprocess_noise runtime:" << prep_noise_time << endl;
      cout << "[Server] Noise processed. Shape: (";
      cout << noise_ct.size() << ")" << endl;
    }

    vector<vector<vector<Plaintext>>> masks_OP;
    masks_OP = HE_preprocess_filters_OP(*filters, data, cc);

    if (data.print_cnts) {
      prep_mat_time = sw.lap();
      cout << "[Server] HE_preprocess_filters_OP runtime:" << prep_mat_time << endl;
      cout << "[Server] Filters processed. Shape: (";
      cout << masks_OP.size() << ", " << masks_OP[0].size() << ", ";
      cout << masks_OP[0][0].size() << ")" << endl;
      cout << "[Server] Total offline pre-processing time: ";
      cout << prep_mat_time + prep_noise_time << endl;
    }

    vector<Ciphertext<DCRTPoly>> conv_result;
    vector<Ciphertext<DCRTPoly>> result;
    vector<vector<Ciphertext<DCRTPoly>>> rotations(data.inp_ct);
    for (int i = 0; i < data.inp_ct; i++) {
      rotations[i].resize(data.filter_size);
    }
        
    if(data.use_heliks){
      conv_result = HE_conv_heliks( ct, masks_OP, cc 
                                      , data
                                      , counts
                                      );
      if (verbose) {cout << "[Server] Convolution done. Shape: (";
      cout << conv_result.size() << ")" << endl;}

      auto result = HE_output_rotations_MR(conv_result, data,cc,
                                      zero, noise_ct
                                      , counts
                                      );
      if (verbose) cout << "[Server] Output Rotations done" << endl;
    } else {
      rotations = filter_rotations(ct, data, cc
                                  , counts
                                  );
      if (verbose) cout << "[Server] Filter Rotations done" << endl;

      conv_result =
          HE_conv_OP(masks_OP, rotations, data, cc, zero
                      , counts
                      );
      if (verbose) {cout << "[Server] Convolution done. Shape: (";
      cout << conv_result.size() << ")" << endl;}

      result = HE_output_rotations(conv_result, data, cc,
                                  zero, noise_ct
                                  , counts
                                  );
      if (verbose) cout << "[Server] Output Rotations done" << endl;

      for (size_t ct_idx = 0; ct_idx < result.size(); ct_idx++) {
        // evaluator_->mod_switch_to_next_inplace(result[ct_idx]);
        cc->ModReduceInPlace(result[ct_idx]);
      }
      if(verbose) cout << "[Server] Modulus Reduction done" << endl;
    }
    
    if (data.print_cnts) {
      processing_time = sw.lap();
      cout << "[Server] Total online processing time: ";
      cout << processing_time << endl;
      cout << "[Server] Result computed and sent" << endl;
    }

    // auto HE_result = HE_decrypt(result, data, cc, keys.secretKey);

    // if (verbose) cout << "[Client] Result received and decrypted" << endl;
}

/* non_strided_conv_MR
*/
void ConvField::non_strided_conv_MR(int32_t H, int32_t W, int32_t CI, int32_t FH,
                                 int32_t FW, int32_t CO, vector<vector<vector<int64_t>>> *image,
                                 vector<vector<vector<vector<int64_t>>>> *filters,
                                 vector<vector<vector<int64_t>>> &outArr,
                                 vector<int> &counts,
                                 bool verbose
                                 ) {
  data.image_h = H;
  data.image_w = W;
  data.inp_chans = CI;
  data.out_chans = CO;
  data.filter_h = FH;
  data.filter_w = FW;
  data.pad_t = 0;
  data.pad_b = 0;
  data.pad_l = 0;
  data.pad_r = 0;
  data.stride_h = 1;
  data.stride_w = 1;
  auto _slot_count = max(data.slot_count, 2 * ceil2Pow(H * W));

  if (_slot_count != data.slot_count) {
    gen_context(_slot_count);
  }

  configure(verbose);

  auto sw = StopWatch();
  double prep_mat_time{}, prep_noise_time{}, processing_time{};

    auto pt = preprocess_image_OP(*image, data);
    if (verbose) cout << "[Client] Image preprocessed" << endl;

    auto ct = HE_encrypt(pt, data, cc, keys.publicKey);
    if (verbose) cout << "[Client] Image encrypted and sent" << endl;

    vector<vector<int64_t>> sec_share_vec = gen2D_UID_int64(data.out_chans, 
                                data.output_h * data.output_w, data.min, data.max);

    int64_t **secret_share = new int64_t *[CO];
    for (int chan = 0; chan < CO; chan++) {
      secret_share[chan] = sec_share_vec.at(chan).data();
    }

    vector<Ciphertext<DCRTPoly>> noise_ct = HE_preprocess_noise(
        secret_share, data, cc, keys.publicKey);

    if (data.print_cnts) {
      prep_noise_time = sw.lap();
      cout << "[Server] HE_preprocess_noise runtime:" << prep_noise_time << endl;
      cout << "[Server] Noise processed. Shape: (";
      cout << noise_ct.size() << ")" << endl;
    }

    vector<vector<vector<Plaintext>>> masks_OP;
    vector<vector<vector<int>>> rot_amts;
    vector<map<int, vector<vector<int>>>> rot_maps;
    masks_OP = HE_preprocess_filters_OP_MR(*filters, data, cc
                                          , rot_amts
                                          , rot_maps
                                          );

    if (data.print_cnts) {
      cout << "[Server] HE_preprocess_filters_OP_MR runtime:" << prep_mat_time << endl;
      cout << "[Server] Filters processed. Shape: (";
      cout << masks_OP.size() << ", " << masks_OP[0].size() << ", ";
      cout << masks_OP[0][0].size() << ")" << endl;
      cout << "[Server] Total offline pre-processing time: ";
      cout << prep_mat_time + prep_noise_time << endl;
    }

    auto conv_result =
        HE_conv_OP_MR(masks_OP, data, cc, zero
                      , ct
                      , rot_amts
                      , rot_maps
                      , counts
                      );

    if (verbose) {cout << "[Server] Convolution done. Shape: (";
    cout << conv_result.size() << ")" << endl;}

    auto result = HE_output_rotations(conv_result, data, cc,
                                 zero, noise_ct
                                 , counts
                                 );

    if (verbose) {cout << "[Server] Output Rotations done. Shape: (";
    cout << result.size() << ")" << endl;}

    for (size_t ct_idx = 0; ct_idx < result.size(); ct_idx++) {
      // evaluator_->mod_switch_to_next_inplace(result[ct_idx]);
      cc->ModReduceInPlace(result[ct_idx]);
    }

    if (data.print_cnts) {
      cout << "[Server] Result computed and sent" << endl;
      processing_time = sw.lap();
      cout << "[Server] Total online processing time: ";
      cout << processing_time << endl;
    }

    // for (int idx = 0; idx < data.output_h * data.output_w; idx++) {
    //   for (int chan = 0; chan < CO; chan++) {
    //     outArr[idx / data.output_w][idx % data.output_w][chan] +=
    //         (prime_mod - secret_share[chan][idx]);
    //   }
    // }
    // for (int i = 0; i < data.out_chans; i++) delete[] secret_share[i];
    // delete[] secret_share;
}

/* convolution
*/
// Alice privately holds `filterArr`.
// The `inputArr' is secretly shared between Alice and Bob.
// The underlying Arithmetic is `prime_mod`.
void ConvField::convolution(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
    int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW,
    const vector<vector<vector<vector<int64_t>>>> &inputArr,
    const vector<vector<vector<vector<int64_t>>>> &filterArr,
    vector<vector<vector<vector<int64_t>>>> &outArr, 
    vector<bool> options) {
  bool verify_output;
  bool verbose;

  if (options.size() > 0) verify_output   = options[0]; else verify_output   = false;
  if (options.size() > 1) verbose         = options[1]; else verbose         = false;
  if (options.size() > 2) data.use_heliks = options[2]; else data.use_heliks = true ;
  if (options.size() > 3) data.print_cnts = options[3]; else data.print_cnts = false;

  if(verbose){
    cout << "[convolution] Options: " << endl;
    cout << "+ verify_output: " << verify_output << endl;
    cout << "+ verbose      : " << verbose << endl;
    cout << "+ use_heliks   : " << data.use_heliks << endl;
    cout << "+ print_cnts   : " << data.print_cnts << endl;
  }

  int64_t prime_mod = data.prime_mod;

  vector<int> counts(10);
  int paddedH = H + zPadHLeft + zPadHRight;
  int paddedW = W + zPadWLeft + zPadWRight;
  int newH = 1 + (paddedH - FH) / strideH;
  int newW = 1 + (paddedW - FW) / strideW;
  int limitH = FH + ((paddedH - FH) / strideH) * strideH;
  int limitW = FW + ((paddedW - FW) / strideW) * strideW;

  for (int i = 0; i < newH; i++) {
    for (int j = 0; j < newW; j++) {
      for (int k = 0; k < CO; k++) {
        outArr[0][i][j][k] = 0;
      }
    }
  }

  vector<vector<vector<int64_t>>> image;
  vector<vector<vector<vector<int64_t>>>> filters;

  image.resize(CI);
  for (int chan = 0; chan < CI; chan++) {
    vector<vector<int64_t>> tmp_chan(H, vector<int64_t>(W));
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        tmp_chan.at(h).at(w) =
            neg_mod((int64_t)inputArr[0][h][w][chan], (int64_t)prime_mod);
      }
    }
    image[chan] = tmp_chan;
  }

  filters.resize(CO);
  for (int out_c = 0; out_c < CO; out_c++) {
    vector<vector<vector<int64_t>>> tmp_img(CI);
    for (int inp_c = 0; inp_c < CI; inp_c++) {
      vector<vector<int64_t>> tmp_chan(FH, vector<int64_t>(FW));
      for (int idx = 0; idx < FH * FW; idx++) {
        int64_t val = (int64_t)filterArr[idx / FW][idx % FW][inp_c][out_c];
        if (val > int64_t(prime_mod / 2)) {
          val = val - prime_mod;
        }
        tmp_chan.at(idx / FW).at(idx % FW) = val;
      }
      tmp_img[inp_c] = tmp_chan;
    }
    filters[out_c] = tmp_img;
  }

  // if (party == BOB) {
    for (int s_row = 0; s_row < strideH; s_row++) {
      for (int s_col = 0; s_col < strideW; s_col++) {
        int lH = ((limitH - s_row + strideH - 1) / strideH);
        int lW = ((limitW - s_col + strideW - 1) / strideW);
        vector<vector<vector<int64_t>>> lImage(CI);
        for (int chan = 0; chan < CI; chan++) {
          vector<vector<int64_t>> tmp_chan(lH, vector<int64_t>(lW));
          // lImage[chan] = new int64_t[lH*lW];
          for (int row = 0; row < lH; row++) {
            for (int col = 0; col < lW; col++) {
              int idxH = row * strideH + s_row - zPadHLeft;
              int idxW = col * strideW + s_col - zPadWLeft;
              if ((idxH < 0 || idxH >= H) || (idxW < 0 || idxW >= W)) {
                tmp_chan.at(row).at(col) = 0;
              } else {
                tmp_chan.at(row).at(col) =
                    neg_mod(inputArr[0][idxH][idxW][chan], (int64_t)prime_mod);
              }
            }
          }
          lImage[chan] = tmp_chan;
        }
        int lFH = ((FH - s_row + strideH - 1) / strideH);
        int lFW = ((FW - s_col + strideW - 1) / strideW);
        vector<vector<vector<vector<int64_t>>>> lFilters(CO);
        for (int out_c = 0; out_c < CO; out_c++) {
          vector<vector<vector<int64_t>>> tmp_img(CI);
          for (int inp_c = 0; inp_c < CI; inp_c++) {
            vector<vector<int64_t>> tmp_chan(lFH, vector<int64_t>(lFW));
            for (int row = 0; row < lFH; row++) {
              for (int col = 0; col < lFW; col++) {
                int idxFH = row * strideH + s_row;
                int idxFW = col * strideW + s_col;
                tmp_chan.at(row).at(col) = neg_mod(
                    filterArr[idxFH][idxFW][inp_c][out_c], (int64_t)prime_mod);
              }
            }
            tmp_img[inp_c] = tmp_chan;
          }
          lFilters[out_c] = tmp_img;
        }

        if (lFH > 0 && lFW > 0) {
          non_strided_conv(lH, lW, CI, lFH, lFW, CO, &lImage, &lFilters, outArr[0], 
                          counts,
                          verbose);
        }
      }
    }
    for (int idx = 0; idx < newH * newW; idx++) {
      for (int chan = 0; chan < CO; chan++) {
        outArr[0][idx / newW][idx % newW][chan] = neg_mod(
            (int64_t)outArr[0][idx / newW][idx % newW][chan], prime_mod);
      }
    }

  //   if (verify_output) verify(H, W, CI, CO, image, nullptr, outArr);
  // } else  // party == ALICE
  // {
  //   filters.resize(CO);
  //   for (int out_c = 0; out_c < CO; out_c++) {
  //     vector<vector<vector<int64_t>>> tmp_img(CI);
  //     for (int inp_c = 0; inp_c < CI; inp_c++) {
  //       vector<vector<int64_t>> tmp_chan(FH, vector<int64_t>(FW));
  //       for (int idx = 0; idx < FH * FW; idx++) {
  //         int64_t val = (int64_t)filterArr[idx / FW][idx % FW][inp_c][out_c];
  //         if (val > int64_t(prime_mod / 2)) {
  //           val = val - prime_mod;
  //         }
  //         tmp_chan.at(idx / FW).at(idx % FW) = val;
  //       }
  //       tmp_img[inp_c] = tmp_chan;
  //     }
  //     filters[out_c] = tmp_img;
  //   }

  //   for (int s_row = 0; s_row < strideH; s_row++) {
  //     for (int s_col = 0; s_col < strideW; s_col++) {
  //       int lH = ((limitH - s_row + strideH - 1) / strideH);
  //       int lW = ((limitW - s_col + strideW - 1) / strideW);
  //       int lFH = ((FH - s_row + strideH - 1) / strideH);
  //       int lFW = ((FW - s_col + strideW - 1) / strideW);
  //       vector<vector<vector<vector<int64_t>>>> lFilters(CO);
  //       for (int out_c = 0; out_c < CO; out_c++) {
  //         vector<vector<vector<int64_t>>> tmp_img(CI);
  //         for (int inp_c = 0; inp_c < CI; inp_c++) {
  //           vector<vector<int64_t>> tmp_chan(lFH, vector<int64_t>(lFW));
  //           for (int row = 0; row < lFH; row++) {
  //             for (int col = 0; col < lFW; col++) {
  //               int idxFH = row * strideH + s_row;
  //               int idxFW = col * strideW + s_col;
  //               tmp_chan.at(row).at(col) = neg_mod(
  //                   filterArr[idxFH][idxFW][inp_c][out_c], (int64_t)prime_mod);
  //             }
  //           }
  //           tmp_img[inp_c] = tmp_chan;
  //         }
  //         lFilters[out_c] = tmp_img;
  //       }

  //       if (lFH > 0 && lFW > 0) {
  //         non_strided_conv(lH, lW, CI, lFH, lFW, CO, nullptr, &lFilters, outArr[0],
  //                          counts,
  //                          verbose);
  //       }
  //     }
  //   }
    data.image_h = H;
    data.image_w = W;
    data.inp_chans = CI;
    data.out_chans = CO;
    data.filter_h = FH;
    data.filter_w = FW;
    data.pad_t = zPadHLeft;
    data.pad_b = zPadHRight;
    data.pad_l = zPadWLeft;
    data.pad_r = zPadWRight;
    data.stride_h = strideH;
    data.stride_w = strideW;

    // // The filter values should be small enough to not overflow int64_t
    // vector<vector<vector<int64_t>>> local_result = ideal_functionality(image, filters);

    // for (int idx = 0; idx < newH * newW; idx++) {
    //   for (int chan = 0; chan < CO; chan++) {
    //     outArr[0][idx / newW][idx % newW][chan] =
    //         neg_mod((int64_t)local_result[chan].at(idx / newW).at(idx % newW) +
    //                     (int64_t)outArr[0][idx / newW][idx % newW][chan],
    //                 prime_mod);
    //   }
    // }
    // if (verify_output) verify(H, W, CI, CO, image, &filters, outArr);

    // printf("Conv-SCI HE #ops - mul: %i, add: %i, mod: %i, rot: %i \n", counts[0], counts[1], counts[2], counts[3]);
  // }
    if(data.print_cnts){
      cout << "[convolution] HE operation counts: " << endl;
      cout << "+ mul: " << counts[0] << endl;
      cout << "+ add: " << counts[1] << endl;
      cout << "+ mod: " << counts[2] << endl;
      cout << "+ rot: " << counts[3] << endl;
    }
}

/* convolution_MR
*/
void ConvField::convolution_MR(
    int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW,
    int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft,
    int32_t zPadWRight, int32_t strideH, int32_t strideW,
    const vector<vector<vector<vector<int64_t>>>> &inputArr,
    const vector<vector<vector<vector<int64_t>>>> &filterArr,
    vector<vector<vector<vector<int64_t>>>> &outArr, 
    vector<bool> options) {

  bool verify_output;
  bool verbose;

  if (options.size() > 0) verify_output   = options[0]; else verify_output   = false;
  if (options.size() > 1) verbose         = options[1]; else verbose         = false;
  if (options.size() > 2) data.use_heliks = options[2]; else data.use_heliks = true ;
  if (options.size() > 3) data.print_cnts = options[3]; else data.print_cnts = false;

  if(verbose){
    cout << "[convolution] Options: " << endl;
    cout << "+ verify_output: " << verify_output << endl;
    cout << "+ verbose      : " << verbose << endl;
    cout << "+ use_heliks   : " << data.use_heliks << endl;
    cout << "+ print_cnts   : " << data.print_cnts << endl;
  }

  int64_t prime_mod = data.prime_mod;

  vector<int> counts(10);
  int paddedH = H + zPadHLeft + zPadHRight;
  int paddedW = W + zPadWLeft + zPadWRight;
  int newH = 1 + (paddedH - FH) / strideH;
  int newW = 1 + (paddedW - FW) / strideW;
  int limitH = FH + ((paddedH - FH) / strideH) * strideH;
  int limitW = FW + ((paddedW - FW) / strideW) * strideW;

  for (int i = 0; i < newH; i++) {
    for (int j = 0; j < newW; j++) {
      for (int k = 0; k < CO; k++) {
        outArr[0][i][j][k] = 0;
      }
    }
  }

  vector<vector<vector<int64_t>>> image;
  vector<vector<vector<vector<int64_t>>>> filters;

  image.resize(CI);
  for (int chan = 0; chan < CI; chan++) {
    vector<vector<int64_t>> tmp_chan(H, vector<int64_t>(W));
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        tmp_chan.at(h).at(w) =
            neg_mod((int64_t)inputArr[0][h][w][chan], (int64_t)prime_mod);
      }
    }
    image[chan] = tmp_chan;
  }

  filters.resize(CO);
  for (int out_c = 0; out_c < CO; out_c++) {
    vector<vector<vector<int64_t>>> tmp_img(CI);
    for (int inp_c = 0; inp_c < CI; inp_c++) {
      vector<vector<int64_t>> tmp_chan(FH, vector<int64_t>(FW));
      for (int idx = 0; idx < FH * FW; idx++) {
        int64_t val = (int64_t)filterArr[idx / FW][idx % FW][inp_c][out_c];
        if (val > int64_t(prime_mod / 2)) {
          val = val - prime_mod;
        }
        tmp_chan.at(idx / FW).at(idx % FW) = val;
      }
      tmp_img[inp_c] = tmp_chan;
    }
    filters[out_c] = tmp_img;
  }

  // if (party == BOB) {
    for (int s_row = 0; s_row < strideH; s_row++) {
      for (int s_col = 0; s_col < strideW; s_col++) {
        int lH = ((limitH - s_row + strideH - 1) / strideH);
        int lW = ((limitW - s_col + strideW - 1) / strideW);
        vector<vector<vector<int64_t>>> lImage(CI);
        for (int chan = 0; chan < CI; chan++) {
          vector<vector<int64_t>> tmp_chan(lH, vector<int64_t>(lW));
          // lImage[chan] = new int64_t[lH*lW];
          for (int row = 0; row < lH; row++) {
            for (int col = 0; col < lW; col++) {
              int idxH = row * strideH + s_row - zPadHLeft;
              int idxW = col * strideW + s_col - zPadWLeft;
              if ((idxH < 0 || idxH >= H) || (idxW < 0 || idxW >= W)) {
                tmp_chan.at(row).at(col) = 0;
              } else {
                tmp_chan.at(row).at(col) =
                    neg_mod(inputArr[0][idxH][idxW][chan], (int64_t)prime_mod);
              }
            }
          }
          lImage[chan] = tmp_chan;
        }
        int lFH = ((FH - s_row + strideH - 1) / strideH);
        int lFW = ((FW - s_col + strideW - 1) / strideW);
        vector<vector<vector<vector<int64_t>>>> lFilters(CO);
        for (int out_c = 0; out_c < CO; out_c++) {
          vector<vector<vector<int64_t>>> tmp_img(CI);
          for (int inp_c = 0; inp_c < CI; inp_c++) {
            vector<vector<int64_t>> tmp_chan(lFH, vector<int64_t>(lFW));
            for (int row = 0; row < lFH; row++) {
              for (int col = 0; col < lFW; col++) {
                int idxFH = row * strideH + s_row;
                int idxFW = col * strideW + s_col;
                tmp_chan.at(row).at(col) = neg_mod(
                    filterArr[idxFH][idxFW][inp_c][out_c], (int64_t)prime_mod);
              }
            }
            tmp_img[inp_c] = tmp_chan;
          }
          lFilters[out_c] = tmp_img;
        }

        if (lFH > 0 && lFW > 0) {
          non_strided_conv_MR(lH, lW, CI, lFH, lFW, CO, nullptr, &lFilters,
                              outArr[0], 
                              counts, 
                              verbose);
        }

      }
    }
    for (int idx = 0; idx < newH * newW; idx++) {
      for (int chan = 0; chan < CO; chan++) {
        outArr[0][idx / newW][idx % newW][chan] = neg_mod(
            (int64_t)outArr[0][idx / newW][idx % newW][chan], prime_mod);
      }
    }

    data.image_h = H;
    data.image_w = W;
    data.inp_chans = CI;
    data.out_chans = CO;
    data.filter_h = FH;
    data.filter_w = FW;
    data.pad_t = zPadHLeft;
    data.pad_b = zPadHRight;
    data.pad_l = zPadWLeft;
    data.pad_r = zPadWRight;
    data.stride_h = strideH;
    data.stride_w = strideW;

    // // The filter values should be small enough to not overflow int64_t
    // vector<vector<vector<int64_t>>> local_result = ideal_functionality(image, filters);

    // for (int idx = 0; idx < newH * newW; idx++) {
    //   for (int chan = 0; chan < CO; chan++) {
    //     outArr[0][idx / newW][idx % newW][chan] =
    //         neg_mod((int64_t)local_result[chan].at(idx / newW).at(idx % newW) +
    //                     (int64_t)outArr[0][idx / newW][idx % newW][chan],
    //                 prime_mod);
    //   }
    // }
    // if (verify_output) verify(H, W, CI, CO, image, &filters, outArr);

    // printf("Conv-SB HE #ops - mul: %i, add: %i, mod: %i, rot: %i \n", counts[0], counts[1], counts[2], counts[3]);
  
    if(data.print_cnts){
      cout << "[convolution_MR] HE operation counts: " << endl;
      cout << "+ mul: " << counts[0] << endl;
      cout << "+ add: " << counts[1] << endl;
      cout << "+ mod: " << counts[2] << endl;
      cout << "+ rot: " << counts[3] << endl;
    }
}

/* convolution_heliks
*/
void ConvField::convolution_heliks(
    int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW,
    int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft,
    int32_t zPadWRight, int32_t strideH, int32_t strideW,
    const vector<vector<vector<vector<int64_t>>>> &inputArr,
    const vector<vector<vector<vector<int64_t>>>> &filterArr,
    vector<vector<vector<vector<int64_t>>>> &outArr, 
    vector<bool> options) {

  bool verify_output;
  bool verbose;

  if (options.size() > 0) verify_output   = options[0]; else verify_output   = false;
  if (options.size() > 1) verbose         = options[1]; else verbose         = false;
  if (options.size() > 2) data.use_heliks = options[2]; else data.use_heliks = true ;
  if (options.size() > 3) data.print_cnts = options[3]; else data.print_cnts = false;

  if(verbose){
    cout << "[convolution] Options: " << endl;
    cout << "+ verify_output: " << verify_output << endl;
    cout << "+ verbose      : " << verbose << endl;
    cout << "+ use_heliks   : " << data.use_heliks << endl;
    cout << "+ print_cnts   : " << data.print_cnts << endl;
  }

  int64_t prime_mod = data.prime_mod;

  vector<int> counts(10);

  int paddedH = H + zPadHLeft + zPadHRight;
  int paddedW = W + zPadWLeft + zPadWRight;
  int newH = 1 + (paddedH - FH) / strideH;
  int newW = 1 + (paddedW - FW) / strideW;
  int limitH = FH + ((paddedH - FH) / strideH) * strideH;
  int limitW = FW + ((paddedW - FW) / strideW) * strideW;

  for (int i = 0; i < newH; i++) {
    for (int j = 0; j < newW; j++) {
      for (int k = 0; k < CO; k++) {
        outArr[0][i][j][k] = 0;
      }
    }
  }

  vector<vector<vector<int64_t>>> image;
  vector<vector<vector<vector<int64_t>>>> filters;

  image.resize(CI);
  for (int chan = 0; chan < CI; chan++) {
    vector<vector<int64_t>> tmp_chan(H, vector<int64_t>(W));
    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        tmp_chan.at(h).at(w) =
            neg_mod((int64_t)inputArr[0][h][w][chan], (int64_t)prime_mod);
      }
    }
    image[chan] = tmp_chan;
  }

  filters.resize(CO);
  for (int out_c = 0; out_c < CO; out_c++) {
    vector<vector<vector<int64_t>>> tmp_img(CI);
    for (int inp_c = 0; inp_c < CI; inp_c++) {
      vector<vector<int64_t>> tmp_chan(FH, vector<int64_t>(FW));
      for (int idx = 0; idx < FH * FW; idx++) {
        int64_t val = (int64_t)filterArr[idx / FW][idx % FW][inp_c][out_c];
        if (val > int64_t(prime_mod / 2)) {
          val = val - prime_mod;
        }
        tmp_chan.at(idx / FW).at(idx % FW) = val;
      }
      tmp_img[inp_c] = tmp_chan;
    }
    filters[out_c] = tmp_img;
  }

  for (int s_row = 0; s_row < strideH; s_row++) {
    for (int s_col = 0; s_col < strideW; s_col++) {
      int lH = ((limitH - s_row + strideH - 1) / strideH);
      int lW = ((limitW - s_col + strideW - 1) / strideW);
      vector<vector<vector<int64_t>>> lImage(CI);
      for (int chan = 0; chan < CI; chan++) {
        vector<vector<int64_t>> tmp_chan(lH, vector<int64_t>(lW));
        // lImage[chan] = new int64_t[lH*lW];
        for (int row = 0; row < lH; row++) {
          for (int col = 0; col < lW; col++) {
            int idxH = row * strideH + s_row - zPadHLeft;
            int idxW = col * strideW + s_col - zPadWLeft;
            if ((idxH < 0 || idxH >= H) || (idxW < 0 || idxW >= W)) {
              tmp_chan.at(row).at(col) = 0;
            } else {
              tmp_chan.at(row).at(col) =
                  neg_mod(inputArr[0][idxH][idxW][chan], (int64_t)prime_mod);
            }
          }
        }
        lImage[chan] = tmp_chan;
      }
      int lFH = ((FH - s_row + strideH - 1) / strideH);
      int lFW = ((FW - s_col + strideW - 1) / strideW);
      vector<vector<vector<vector<int64_t>>>> lFilters(CO);
      for (int out_c = 0; out_c < CO; out_c++) {
        vector<vector<vector<int64_t>>> tmp_img(CI);
        for (int inp_c = 0; inp_c < CI; inp_c++) {
          vector<vector<int64_t>> tmp_chan(lFH, vector<int64_t>(lFW));
          for (int row = 0; row < lFH; row++) {
            for (int col = 0; col < lFW; col++) {
              int idxFH = row * strideH + s_row;
              int idxFW = col * strideW + s_col;
              tmp_chan.at(row).at(col) = neg_mod(
                  filterArr[idxFH][idxFW][inp_c][out_c], (int64_t)prime_mod);
            }
          }
          tmp_img[inp_c] = tmp_chan;
        }
        lFilters[out_c] = tmp_img;
      }
      if (lFH > 0 && lFW > 0) {
        non_strided_conv(lH, lW, CI, lFH, lFW, CO, &lImage, nullptr,
                          outArr[0],
                          counts, 
                          verbose);
      }
    }
  }
  // for (int idx = 0; idx < newH * newW; idx++) {
  //   for (int chan = 0; chan < CO; chan++) {
  //     outArr[0][idx / newW][idx % newW][chan] = neg_mod(
  //         (int64_t)outArr[0][idx / newW][idx % newW][chan], prime_mod);
  //   }
  // }

  //   if (verify_output) verify(H, W, CI, CO, image, nullptr, outArr);


  //   for (int s_row = 0; s_row < strideH; s_row++) {
  //     for (int s_col = 0; s_col < strideW; s_col++) {
  //       int lH = ((limitH - s_row + strideH - 1) / strideH);
  //       int lW = ((limitW - s_col + strideW - 1) / strideW);
  //       int lFH = ((FH - s_row + strideH - 1) / strideH);
  //       int lFW = ((FW - s_col + strideW - 1) / strideW);
  //       vector<vector<vector<vector<int64_t>>>> lFilters(CO);
  //       for (int out_c = 0; out_c < CO; out_c++) {
  //         vector<vector<vector<int64_t>>> tmp_img(CI);
  //         for (int inp_c = 0; inp_c < CI; inp_c++) {
  //           vector<vector<int64_t>> tmp_chan(lFH, lFW);
  //           for (int row = 0; row < lFH; row++) {
  //             for (int col = 0; col < lFW; col++) {
  //               int idxFH = row * strideH + s_row;
  //               int idxFW = col * strideW + s_col;
  //               tmp_chan.at(row).at(col) = neg_mod(
  //                   filterArr[idxFH][idxFW][inp_c][out_c], (int64_t)prime_mod);
  //             }
  //           }
  //           tmp_img[inp_c] = tmp_chan;
  //         }
  //         lFilters[out_c] = tmp_img;
  //       }

  //       if (lFH > 0 && lFW > 0) {
  //         non_strided_conv_heliks(lH, lW, CI, lFH, lFW, CO, nullptr, &lFilters,
  //                             outArr[0], 
  //                             counts, 
  //                             verbose);
  //       }
  //     }
  //   }
  data.image_h = H;
  data.image_w = W;
  data.inp_chans = CI;
  data.out_chans = CO;
  data.filter_h = FH;
  data.filter_w = FW;
  data.pad_t = zPadHLeft;
  data.pad_b = zPadHRight;
  data.pad_l = zPadWLeft;
  data.pad_r = zPadWRight;
  data.stride_h = strideH;
  data.stride_w = strideW;

    // // The filter values should be small enough to not overflow int64_t
    // vector<vector<vector<int64_t>>> local_result = ideal_functionality(image, filters);

    // for (int idx = 0; idx < newH * newW; idx++) {
    //   for (int chan = 0; chan < CO; chan++) {
    //     outArr[0][idx / newW][idx % newW][chan] =
    //         neg_mod((int64_t)local_result[chan].at(idx / newW).at(idx % newW) +
    //                     (int64_t)outArr[0][idx / newW][idx % newW][chan],
    //                 prime_mod);
    //   }
    // }
    // if (verify_output) verify(H, W, CI, CO, image, &filters, outArr);

    // printf("Conv-NTT HE #ops - mul: %i, add: %i, mod: %i, rot: %i \n", counts[0], counts[1], counts[2], counts[3]);

  if(data.print_cnts){
    cout << "[convolution_heliks] HE operation counts: " << endl;
    cout << "+ mul: " << counts[0] << endl;
    cout << "+ add: " << counts[1] << endl;
    cout << "+ mod: " << counts[2] << endl;
    cout << "+ rot: " << counts[3] << endl;
  }
}

// void ConvField::convolution(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, 
//     int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW,
//     const vector<vector<vector<vector<int64_t>>>> &inputArr,
//     const vector<vector<vector<vector<int64_t>>>> &filterArr,
//     vector<vector<vector<vector<int64_t>>>> &outArr, 
//     vector<bool> options){
  
//   if (options.size() > 2) data.use_heliks   = options[2]; else data.use_heliks   = false;

//   if(data.use_heliks){
//     convolution_heliks(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, 
//           zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr, options);
//   } else {
//     convolution_cf2(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, 
//           zPadWLeft, zPadWRight, strideH, strideW, inputArr, filterArr, outArr, options);
//   }
// }