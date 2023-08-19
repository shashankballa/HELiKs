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
-----------------------------------------------------------------------------

Original Author: Deevashwer Rathee
Modified by Authors of HELiKs

*/

#include "fc-field.h"
#include "utils.h"

using namespace std;
using namespace lbcrypto;

/*
  Pre-processes the input vector
*/
vector<vector<int64_t>> 
  preprocess_vec(const int64_t *input, const FCMetadata &data){
  // Create copies of the input vector to fill the ciphertext appropiately.
  // Pack using powers of two for easy rotations later
  vector<vector<int64_t>> result(data.i_tiles);
  int num_cols = data.input_size;
  int rem_cols = data.filter_w % data.input_size;

  for (size_t it = 0; it < data.i_tiles; it++)
  {
    vector<int64_t> pod_matrix(data.slot_count);
    if ((it == data.i_tiles - 1) && (rem_cols != 0)) num_cols = rem_cols;
    else num_cols = data.input_size;
    for (int col = 0; col < num_cols; col++) {
      for (int idx = 0; idx < data.pack_num; idx++) {
        pod_matrix[data.input_size * idx + col] = 
                                        input[it * data.input_size + col];
      }
    }
    result.at(it) = pod_matrix;
  }
  return result;
}

/*
  Pre-processes the input vector into a plaintext.
*/
vector<Plaintext> 
  preprocess_vec(const int64_t *input, const FCMetadata &data,
                 CryptoContext<DCRTPoly> &cc) {

  vector<vector<int64_t>> pod_matrices = preprocess_vec(input, data);

  vector<Plaintext> pts(data.i_tiles);
  for (size_t i = 0; i < data.i_tiles; i++)
    pts.at(i) = cc->MakePackedPlaintext(pod_matrices.at(i));
    // batch_encoder.encode(pod_matrices.at(i), pts.at(i));
  return pts;
}

/*
  Pre-processes the input vector into a ciphertext.
*/
vector<Ciphertext<DCRTPoly>> 
  preprocess_enc_vec(const int64_t *input, const FCMetadata &data,
                 CryptoContext<DCRTPoly> &cc, 
                 PublicKey<DCRTPoly> &publicKey) {

  vector<Plaintext> pts = preprocess_vec(input, data, cc);

  vector<Ciphertext<DCRTPoly>> cts(data.i_tiles);
  for (size_t i = 0; i < data.i_tiles; i++)
    cts.at(i) = cc->Encrypt(publicKey, pts.at(i));
  return cts;
}

/*
  Pre-processes the input matrix into plaintexts.
*/
vector<vector<vector<Plaintext>>>
  preprocess_matrix(const int64_t *const *matrix, const FCMetadata &data,
                    CryptoContext<DCRTPoly> &cc) {

  assert(data.pre_comp_ntt == false);

  int num_cols = data.input_size;
  int rem_cols = data.filter_w % data.input_size;

  int num_rows = data.output_size;
  int rem_rows = data.filter_h % data.output_size;
  
  vector<vector<vector<vector<int64_t>>>> 
    result_vec(data.o_tiles, vector<vector<vector<int64_t>>>(data.i_tiles));

  for(int ot = 0; ot < data.o_tiles; ot++){
    if ((ot == data.o_tiles - 1) && (rem_rows != 0)) num_rows = rem_rows;
    else num_rows = data.output_size;
    for(int it = 0; it < data.i_tiles; it++){
      // Pack the filter in alternating order of needed ciphertexts. This way we
      // rotate the input once per ciphertext
      vector<vector<int64_t>> mat_pack(data.num_pt,
                                        vector<int64_t>(data.slot_count));
      if ((it == data.i_tiles - 1) && (rem_cols != 0)) num_cols = rem_cols;
      else num_cols = data.input_size;
      for (int row = 0; row < num_rows; row++) {
        int pt_idx = row / data.num_pt;
        for (int col = 0; col < num_cols; col++) {
          mat_pack[row % data.num_pt][col + data.input_size * pt_idx] =
              matrix[row][col];
        }
      }

      // Take the packed ciphertexts above and repack them in a diagonal ordering.
      int mod_mask = (data.num_pt - 1);
      int wrap_thresh = min((int)(data.slot_count >> 1), data.input_size);
      int wrap_mask = wrap_thresh - 1;
      vector<vector<int64_t>> mat_diag(data.num_pt,
                                        vector<int64_t>(data.slot_count));
      for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
        for (int col = 0; col < data.slot_count; col++) {
          int pt_diag_l = (col - pt_idx) & wrap_mask & mod_mask;
          int pt_diag_h = (col ^ pt_idx) & (data.slot_count / 2) & mod_mask;
          int pt_diag = (pt_diag_h + pt_diag_l);

          int col_diag_l = (col - pt_diag_l) & wrap_mask;
          int col_diag_h = wrap_thresh * (col / wrap_thresh) ^ pt_diag_h;
          int col_diag = col_diag_h + col_diag_l;

          mat_diag[pt_diag][col_diag] = mat_pack[pt_idx][col];
        }
      }
      if(data.mul_then_rot){
    // Reshape the packed ciphertexts to (2, slot_count/2) and rotate them.
        for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
          vector<vector<int64_t>> diag2 = {slice1D(mat_diag.at(pt_idx), 0, data.slot_count/2 - 1),
                                            slice1D(mat_diag.at(pt_idx), data.slot_count/2,   0  )};
          mat_diag.at(pt_idx) = rowEncode2D(rotate2D(diag2, 0, -pt_idx));
        }
      }
      result_vec.at(ot).at(it) = mat_diag;
    }
  }

  vector<vector<vector<Plaintext>>> 
    result(data.o_tiles, vector<vector<Plaintext>>(data.i_tiles));

  for(int ot = 0; ot < data.o_tiles; ot++){
    for(int it = 0; it < data.i_tiles; it++){
      vector<Plaintext> enc_mat(data.num_pt);
      for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
        enc_mat.at(pt_idx) = 
            cc->MakePackedPlaintext(result_vec.at(ot).at(it).at(pt_idx));
      }
      result.at(ot).at(it) = enc_mat;
    }
  }
  return result;
}


/* not using
  Pre-processes the input matrix into Plaintexts such that multiply can be 
  performed before the rotation in fc_online and ransforms the plaintexts to 
  NTT before returning the results. 
vector<vector<vector<Plaintext>>>
  preprocess_matrix(const int64_t *const *matrix, const FCMetadata &data,
                    BatchEncoder &batch_encoder, Evaluator &evaluator,
                    parms_id_type parms_id) {

  int num_cols = data.input_size;
  int rem_cols = data.filter_w % data.input_size;

  int num_rows = data.output_size;
  int rem_rows = data.filter_h % data.output_size;
  
  vector<vector<vector<vector<int64_t>>>> 
    result_vec(data.o_tiles, vector<vector<vector<int64_t>>>(data.i_tiles));

  for(int ot = 0; ot < data.o_tiles; ot++){
    if ((ot == data.o_tiles - 1) && (rem_rows != 0)) num_rows = rem_rows;
    else num_rows = data.output_size;
    for(int it = 0; it < data.i_tiles; it++){
      // Pack the filter in alternating order of needed ciphertexts. This way we
      // rotate the input once per ciphertext
      vector<vector<int64_t>> mat_pack(data.num_pt,
                                        vector<int64_t>(data.slot_count));
      if ((it == data.i_tiles - 1) && (rem_cols != 0)) num_cols = rem_cols;
      else num_cols = data.input_size;
      for (int row = 0; row < num_rows; row++) {
        int pt_idx = row / data.num_pt;
        for (int col = 0; col < num_cols; col++) {
          mat_pack[row % data.num_pt][col + data.input_size * pt_idx] =
              matrix[ot*data.output_size + row][it*data.input_size + col];
        }
      }

      // Take the packed ciphertexts above and repack them in a diagonal ordering.
      int mod_mask = (data.num_pt - 1);
      int wrap_thresh = min((int)(data.slot_count >> 1), data.input_size);
      int wrap_mask = wrap_thresh - 1;
      vector<vector<int64_t>> mat_diag(data.num_pt,
                                        vector<int64_t>(data.slot_count));
      for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
        for (int col = 0; col < data.slot_count; col++) {
          int pt_diag_l = (col - pt_idx) & wrap_mask & mod_mask;
          int pt_diag_h = (col ^ pt_idx) & (data.slot_count / 2) & mod_mask;
          int pt_diag = (pt_diag_h + pt_diag_l);

          int col_diag_l = (col - pt_diag_l) & wrap_mask;
          int col_diag_h = wrap_thresh * (col / wrap_thresh) ^ pt_diag_h;
          int col_diag = col_diag_h + col_diag_l;

          mat_diag[pt_diag][col_diag] = mat_pack[pt_idx][col];
        }
      }

      if(data.mul_then_rot){
    // Reshape the packed ciphertexts to (2, slot_count/2) and rotate them.
        for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
          vector<vector<int64_t>> diag2 = {slice1D(mat_diag.at(pt_idx), 0, data.slot_count/2 - 1),
                                            slice1D(mat_diag.at(pt_idx), data.slot_count/2,   0  )};
          mat_diag.at(pt_idx) = rowEncode2D(rotate2D(diag2, 0, -pt_idx));
        }
      }
      result_vec.at(ot).at(it) = mat_diag;
    }
  }

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<vector<vector<Plaintext>>> 
    result(data.o_tiles, vector<vector<Plaintext>>(data.i_tiles));

  for(int ot = 0; ot < data.o_tiles; ot++){
    for(int it = 0; it < data.i_tiles; it++){
      vector<Plaintext> enc_mat(data.num_pt);
      for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
        batch_encoder.encode(result_vec.at(ot).at(it).at(pt_idx), enc_mat.at(pt_idx));
        if(data.print_times) sw.lap(6);
        if(data.pre_comp_ntt)
    // trasnform the plaintexts to NTT
          evaluator.transform_to_ntt_inplace(enc_mat[pt_idx], parms_id);
          if(data.print_times) sw.lap(3);
      }
      result.at(ot).at(it) = enc_mat;
    }
  }
  
  if(data.print_times){
    cout << "[preprocess_matrix] Times:" << endl;
    cout << "+ mul: " << sw.lapTimes.at(0) << endl;
    cout << "+ rot: " << sw.lapTimes.at(1) << endl;
    cout << "+ add: " << sw.lapTimes.at(2) << endl;
    cout << "+ mds: " << sw.lapTimes.at(5) << endl;
    cout << "+ ntt: " << sw.lapTimes.at(3) << endl;
    cout << "+ int: " << sw.lapTimes.at(4) << endl;
    cout << "+ ecd: " << sw.lapTimes.at(6) << endl;
  }

  return result;
}

*/


/*
  Generates a masking vector of random noise that will be applied to parts of 
  the ciphertext that contain leakage.
*/
vector<vector<int64_t>>
  fc_preprocess_noise(const int64_t *secret_share,
                               const FCMetadata &data) {

  // Sample randomness into vector
  vector<vector<int64_t>> 
    noise (data.o_tiles, vector<int64_t>(data.slot_count));
  if(!data.enable_ss) return noise;
  
  noise = gen2D_UID_int64(data.o_tiles, data.slot_count, 
    data.min, data.max);
  // Puncture the vector with secret shares where an actual fc result value
  // lives

  int num_rows = data.output_size;
  int rem_rows = data.filter_h % data.output_size;

  for(int ot = 0; ot < data.o_tiles; ot++){
    if ((ot == data.o_tiles - 1) && (rem_rows != 0)) num_rows = rem_rows;
    else num_rows = data.output_size;
    for (int row = 0; row < num_rows; row++) {
      int curr_set = row / data.num_pt;
      noise[ot][(row % data.num_pt) + data.input_size * curr_set] =
          secret_share[ot*data.output_size + row];
    }
  }
  return noise;
}

/*
  Generates a plaintext of random noise that will be applied to parts of the 
  ciphertext that contain leakage.
*/
vector<Plaintext> 
  fc_preprocess_noise(const int64_t *secret_share,
                      const FCMetadata &data,
                      CryptoContext<DCRTPoly> &cc) {
  // Sample randomness into vector
  vector<vector<int64_t>> noise = fc_preprocess_noise(secret_share, data);
  vector<Plaintext> noise_pts(data.o_tiles);
  for(int ot = 0; ot < data.o_tiles; ot++)
    noise_pts.at(ot) = cc->MakePackedPlaintext(noise.at(ot));
  return noise_pts;
}

/* not using 
  Generates a ciphertext of random noise that will be applied to parts of the 
  ciphertext that contain leakage.
  
  NOTE: The result need not be encrypted and can be added to the ciphertext 
  as a plaintext.
vector<Ciphertext> 
  fc_preprocess_noise(const int64_t *secret_share,
                      const FCMetadata &data, Encryptor &encryptor,
                      BatchEncoder &batch_encoder) {

  vector<Ciphertext> noise_cts(data.o_tiles);
  auto noise_pts = fc_preprocess_noise(secret_share, data, batch_encoder);
  for(int ot = 0; ot < data.o_tiles; ot++)
    encryptor.encrypt(noise_pts.at(ot), noise_cts.at(ot));
  return noise_cts;
}
*/

/*
  Performs a rotation of the ciphertext by the given amount. The rotation is 
  performed using the Near Adjacent Form (NAF) of the rotation amount.
*/
Ciphertext<DCRTPoly> rotate_arb(Ciphertext<DCRTPoly> &ct, int rot, 
                                CryptoContext<DCRTPoly> &cc){
  
  Ciphertext<DCRTPoly> result = ct;

  int ring_dim = cc->GetCryptoParameters()->GetElementParams()->GetRingDimension();
  rot = rot % ring_dim;

  if(rot == 0) return result;

 // Convert the steps to Near Adjacent Form (NAF)
  vector<int> naf_steps = naf(rot);

  // cout << "+ [rotate_arb] ring_dim:" << ring_dim << "; rot: " << rot << "; naf_steps: ";
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

/*
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Original function in CryptFlow2 with trivial tiling.
*/
vector<Ciphertext<DCRTPoly>> fc_online_cf(vector<Ciphertext<DCRTPoly>> &cts, 
                        vector<vector<vector<Plaintext>>> &enc_mats,
                        const FCMetadata &data, 
                        CryptoContext<DCRTPoly> &cc,
                        Ciphertext<DCRTPoly> &zero,
                        vector<Ciphertext<DCRTPoly>> &noise_ct
                        , PrivateKey<DCRTPoly> &secretKey
                        ) {
  
  assert(data.mul_then_rot == false 
      && data.rot_one_step == false 
      && data.pre_comp_ntt == false);

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<Ciphertext<DCRTPoly>> tiled_result(data.o_tiles);

  Plaintext ct_dec;
  cc->Decrypt(secretKey, cts.at(0), &ct_dec);
  auto ct_vec = ct_dec->GetPackedValue();

  for(int ot = 0; ot < data.o_tiles; ot++){
    for(int it = 0; it < data.i_tiles; it++){
      auto ct = cts.at(it);
      auto enc_mat = enc_mats.at(ot).at(it);
      Ciphertext<DCRTPoly> result = zero;
      // For each matrix ciphertext, rotate the input vector once and multiply + add
      for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
        Ciphertext<DCRTPoly> tmp = rotate_arb(ct, pt_idx, cc);
        if(data.print_times) sw.lap(1);
        tmp = cc->EvalMult(tmp, enc_mat[pt_idx]);
        if(data.print_times) sw.lap(0);
        result = cc->EvalAdd(result, tmp);
        if(data.print_times) sw.lap(2);
      }

      cc->ModReduceInPlace(result);
  //     evaluator.mod_switch_to_next_inplace(result);
  //     if(data.print_times) sw.lap(5);

      if(!data.skip_he_ras){
        // Rotate all partial sums together
        for (int rot = data.num_pt; rot < data.input_size; rot *= 2) {
          Ciphertext<DCRTPoly> tmp = cc->EvalRotate(result, rot);
          if(data.print_times) sw.lap(1);

          result = cc->EvalAdd(result, tmp);
          if(data.print_times) sw.lap(2);
        }
      }
      if(it == 0){
        tiled_result.at(ot) = result;
      } else {
        tiled_result.at(ot) = cc->EvalAdd(tiled_result.at(ot), result);
        if(data.print_times) sw.lap(2);
      }
    }

    if(data.enable_ss){
    // Add noise to cover leakage
      tiled_result.at(ot) = cc->EvalAdd(tiled_result.at(ot), noise_ct.at(ot));
      if(data.print_times) sw.lap(2);
  //     evaluator.mod_switch_to_next_inplace(noise_ct.at(ot));
    }
  }

  if(data.print_times){
    cout << "[fc_online_cf2] Times:" << endl;
    cout << "+ mul: " << sw.lapTimes.at(0) << endl;
    cout << "+ rot: " << sw.lapTimes.at(1) << endl;
    cout << "+ add: " << sw.lapTimes.at(2) << endl;
    cout << "+ mds: " << sw.lapTimes.at(5) << endl;
    cout << "+ ntt: " << sw.lapTimes.at(3) << endl;
    cout << "+ int: " << sw.lapTimes.at(4) << endl;
  }

  return tiled_result;
}


/* fc_online_v1
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Algorithm from HELiKs with multiplication before rotation (sec 4.2) and 
    trivial tiling.
*/
vector<Ciphertext<DCRTPoly>> fc_online_v1(vector<Ciphertext<DCRTPoly>> &cts, 
                        vector<vector<vector<Plaintext>>> &enc_mats,
                        const FCMetadata &data, 
                        CryptoContext<DCRTPoly> &cc,
                        vector<Plaintext> &noise_pt) {

  assert(data.mul_then_rot == true 
      && data.rot_one_step == false 
      && data.pre_comp_ntt == false);

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<Ciphertext<DCRTPoly>> tiled_result(data.o_tiles);

  for(int ot = 0; ot < data.o_tiles; ot++){
    for(int it = 0; it < data.i_tiles; it++){
      auto ct = cts.at(it);
      auto enc_mat = enc_mats.at(ot).at(it);

      Ciphertext<DCRTPoly> result = cc->EvalMult(ct, enc_mat[0]);
      if(data.print_times) sw.lap(0);

      for (int pt_idx = 1; pt_idx < data.num_pt; pt_idx++) {

        Ciphertext<DCRTPoly> tmp = cc->EvalMult(ct, enc_mat[pt_idx]);
        if(data.print_times) sw.lap(0);

        tmp = rotate_arb(tmp, pt_idx, cc);
        if(data.print_times) sw.lap(1);

        result = cc->EvalAdd(result, tmp);
        if(data.print_times) sw.lap(2);
      }

      cc->ModReduceInPlace(result);
      // evaluator.mod_switch_to_next_inplace(result);
      // if(data.print_times) sw.lap(5);

      if(!data.skip_he_ras){
        // Rotate all partial sums together
        for (int rot = data.num_pt; rot < data.input_size; rot *= 2) {
          Ciphertext<DCRTPoly> tmp = rotate_arb(result, rot, cc);
          if(data.print_times) sw.lap(1);
          result = cc->EvalAdd(result, tmp);
          if(data.print_times) sw.lap(2);
        }
      }
      if(it == 0){
        tiled_result.at(ot) = result;
      } else {
        tiled_result.at(ot) = cc->EvalAdd(tiled_result.at(ot), result);
        if(data.print_times) sw.lap(2);
      }
    }

    if(data.enable_ss){
      tiled_result.at(ot) = cc->EvalAdd(tiled_result.at(ot), 
                                        noise_pt.at(ot));
      if(data.print_times) sw.lap(2);
    }
  }
  
  if(data.print_times){
    cout << "[fc_online_v1] Times:" << endl;
    cout << "+ mul: " << sw.lapTimes.at(0) << endl;
    cout << "+ rot: " << sw.lapTimes.at(1) << endl;
    cout << "+ add: " << sw.lapTimes.at(2) << endl;
    cout << "+ mds: " << sw.lapTimes.at(5) << endl;
    cout << "+ ntt: " << sw.lapTimes.at(3) << endl;
    cout << "+ int: " << sw.lapTimes.at(4) << endl;
  }

  return tiled_result;
}

/* fc_online_v2
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Algorithm from HELiKs with multiplication before rotation (sec 4.2),
    1-step rotations (sec 4.3) and tiling (sec 4.5).
*/
vector<Ciphertext<DCRTPoly>> fc_online_v2(vector<Ciphertext<DCRTPoly>> &cts, 
                    vector<vector<vector<Plaintext>>> &enc_mats,
                    const FCMetadata &data,
                    CryptoContext<DCRTPoly> &cc,
                    vector<Plaintext> &noise_pt) {

  assert(data.mul_then_rot == true 
      && data.rot_one_step == true 
      && data.pre_comp_ntt == false);

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<Ciphertext<DCRTPoly>> tiled_result(data.o_tiles);

  for(int ot = 0; ot < data.o_tiles; ot++){

    vector<Ciphertext<DCRTPoly>> psums(data.num_pt);

    for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
      psums.at(pt_idx) = cc->EvalMult(cts.at(0), enc_mats[ot][0][pt_idx]);
      if(data.print_times) sw.lap(0);
      for(int it = 1; it < data.i_tiles; it++){
          auto tmp = cc->EvalMult(cts.at(it), enc_mats[ot][it][pt_idx]);
          if(data.print_times) sw.lap(0);
          psums.at(pt_idx) = cc->EvalAdd(psums.at(pt_idx), tmp);
          if(data.print_times) sw.lap(2);
      }
      cc->ModReduceInPlace(psums.at(pt_idx));
    // evaluator.mod_switch_to_next_inplace(result);
    // if(data.print_times) sw.lap(5);
    }

    Ciphertext<DCRTPoly> result = psums.at(data.num_pt - 1);

    for (int pt_idx = data.num_pt-2; pt_idx >= 0; pt_idx--) {
      result = rotate_arb(result, 1, cc);
      if(data.print_times) sw.lap(1);
      result = cc->EvalAdd(result, psums.at(pt_idx)); 
      if(data.print_times) sw.lap(2);
    }

    if(!data.skip_he_ras){
      // Rotate all partial sums together
      for (int rot = data.num_pt; rot < data.input_size; rot *= 2) {
        auto tmp = rotate_arb(result, rot, cc);
        if(data.print_times) sw.lap(1);
        result = cc->EvalAdd(result, tmp); 
        if(data.print_times) sw.lap(2);
      }
    }
    
    tiled_result.at(ot) = result;

    if(data.enable_ss){
      tiled_result.at(ot) = cc->EvalAdd(tiled_result.at(ot), noise_pt.at(ot));
      if(data.print_times) sw.lap(2);
    }
  }

  if(data.print_times){
    cout << "[fc_online_v2] Times:" << endl;
    cout << "+ mul: " << sw.lapTimes.at(0) << endl;
    cout << "+ rot: " << sw.lapTimes.at(1) << endl;
    cout << "+ add: " << sw.lapTimes.at(2) << endl;
    cout << "+ mds: " << sw.lapTimes.at(5) << endl;
    cout << "+ ntt: " << sw.lapTimes.at(3) << endl;
    cout << "+ int: " << sw.lapTimes.at(4) << endl;
  }

  return tiled_result;
}

/* fc_online_v3
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Algorithm from HELiKs with multiplication before rotation (sec 4.2), 1-step 
    rotations (sec 4.3), NTT precomputation (sec 4.4) and tiling (sec 4.5).

vector<Ciphertext> fc_online_v3(vector<Ciphertext> &cts, 
                    vector<vector<vector<Plaintext>>> &enc_mats,
                    const FCMetadata &data, Evaluator &evaluator,
                    GaloisKeys &gal_keys,
                    vector<Plaintext> &noise_pt) {

  assert(data.mul_then_rot == true 
      && data.rot_one_step == true 
      && data.pre_comp_ntt == true);

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<Ciphertext> tiled_result(data.o_tiles);

  vector<Ciphertext> ct_ntts(data.i_tiles);
  for(int it = 0; it < data.i_tiles; it++){
    evaluator.transform_to_ntt(cts.at(it), ct_ntts.at(it)); 
    if(data.print_times) sw.lap(3);
  }

  for(int ot = 0; ot < data.o_tiles; ot++){

    vector<Ciphertext> psums(data.num_pt);

    for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
      vector<Ciphertext> psums_it = ct_ntts;
      for(int it = 0; it < data.i_tiles; it++){
        if (!enc_mats[ot][it][pt_idx].is_zero()) {
          evaluator.multiply_plain_inplace(
            psums_it.at(it), enc_mats[ot][it][pt_idx]);
          if(data.print_times) sw.lap(0);
        }
      }
      evaluator.add_many(psums_it, psums.at(pt_idx));

    // evaluator.mod_switch_to_next_inplace(result);
    // if(data.print_times) sw.lap(5);

      evaluator.transform_from_ntt_inplace(psums.at(pt_idx)); 
    }

    Ciphertext result = psums.at(data.num_pt - 1);

    for (int pt_idx = data.num_pt-2; pt_idx >= 0; pt_idx--) {
      evaluator.rotate_rows_inplace(result, 1, gal_keys); 
      if(data.print_times) sw.lap(1);
      evaluator.add_inplace(result, psums.at(pt_idx)); 
      if(data.print_times) sw.lap(2);
    }

    if(!data.skip_he_ras){
      // Rotate all partial sums together
      for (int rot = data.num_pt; rot < data.input_size; rot *= 2) {
        Ciphertext tmp;
        if (rot == data.slot_count / 2) {
          evaluator.rotate_columns(result, gal_keys, tmp); 
          if(data.print_times) sw.lap(1);
        } else {
          evaluator.rotate_rows(result, rot, gal_keys, tmp); 
          if(data.print_times) sw.lap(1);
        }
        evaluator.add_inplace(result, tmp); 
        if(data.print_times) sw.lap(2);
      }
    }
    
    tiled_result.at(ot) = result;

    if(data.enable_ss){
    // Add noise to cover leakage
      evaluator.add_plain_inplace(tiled_result.at(ot), noise_pt.at(ot));
      if(data.print_times) sw.lap(2);
    }
  }

  if(data.print_times){
    cout << "[fc_online_v3] Times:" << endl;
    cout << "+ mul: " << sw.lapTimes.at(0) << endl;
    cout << "+ rot: " << sw.lapTimes.at(1) << endl;
    cout << "+ add: " << sw.lapTimes.at(2) << endl;
    cout << "+ mds: " << sw.lapTimes.at(5) << endl;
    cout << "+ ntt: " << sw.lapTimes.at(3) << endl;
    cout << "+ int: " << sw.lapTimes.at(4) << endl;
  }

  return tiled_result;
}
*/

/*
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.
*/
vector<Ciphertext<DCRTPoly>> fc_online(vector<Ciphertext<DCRTPoly>> &cts, 
                    vector<vector<vector<Plaintext>>> &enc_mats,
                    const FCMetadata &data,
                    CryptoContext<DCRTPoly> &cc,
                    Ciphertext<DCRTPoly> &zero,
                    vector<Ciphertext<DCRTPoly>> &noise_ct,
                    vector<Plaintext>  &noise_pt
                    , PrivateKey<DCRTPoly> &secretKey
                    ) {

  vector<Ciphertext<DCRTPoly>> result(data.o_tiles);

  if(data.mul_then_rot == false 
  && data.rot_one_step == false 
  && data.pre_comp_ntt == false){
    result = fc_online_cf(cts, enc_mats, data, cc, zero, noise_ct, secretKey);
  }

  if(data.mul_then_rot == true
  && data.rot_one_step == false 
  && data.pre_comp_ntt == false){
    result = fc_online_v1(cts, enc_mats, data, cc, noise_pt);
  }

  if(data.mul_then_rot == true
  && data.rot_one_step == true 
  && data.pre_comp_ntt == false){
    result = fc_online_v2(cts, enc_mats, data, cc, noise_pt);
  }

  // if(data.mul_then_rot == true
  // && data.rot_one_step == true 
  // && data.pre_comp_ntt == true){
  //   result = fc_online_v3(cts, enc_mats, data, evaluator, gal_keys, noise_pt);
  // }

  return result;
}

/*
  Rotate and sum computation to add up all the partial sums in the ciphertext
  vector after decryption. This is also applied to the server's secret share
  to get the final result.
*/
vector<int64_t> rotate_and_sum(vector<int64_t> plain, const FCMetadata &data) {

  assert(plain.size() == data.slot_count);

// Break the input vector into two halves to mimic BFV/BGV rotaion structure  
  vector<vector<int64_t>> plain_halfs = {slice1D(plain, 0, data.slot_count/2 - 1),
                                          slice1D(plain, data.slot_count/2,   0  )};

// Rotate all partial sums together
  for (int rot = data.num_pt; rot < data.input_size; rot *= 2) {
    vector<vector<int64_t>> tmp(2, vector<int64_t>(data.slot_count/2));
    if (rot == data.slot_count / 2) {
// similar to evaluator.rotate_columns(result, gal_keys, tmp)
      tmp = rotate2D(plain_halfs, 1 , 0);
    }else{
      // similar to evaluator.rotate_rows(result, rot, gal_keys, tmp)
      tmp = rotate2D(plain_halfs, 0, rot);
    }

// Print results for debugging
    // cout << "[rotate_and_sum_SB] rot " << rot << " IN 1: "; 
    // print1D(rowEncode2D(plain_halfs), nprint, false);

    // cout << "[rotate_and_sum_SB] rot " << rot << " IN 2: "; 
    // print1D(rowEncode2D(tmp), nprint, false);

// similar to evaluator.add_inplace(result, tmp)
    for(int i = 0; i < 2; i++){
      for(int j = 0; j < data.slot_count/2; j++){
        plain_halfs.at(i).at(j) = (((((int64_t) plain_halfs.at(i).at(j) 
                                    + (int64_t) tmp.at(i).at(j))
                                    % data.prime_mod) + data.prime_mod) % data.prime_mod);
      }
    }
  }

// Print results for debugging
  // cout << "[rotate_and_sum_SB] OUT : "; 
  // print1D(neg_mod_1D(rowEncode2D(plain_halfs), prime_mod), nprint, false);
  
  auto plain_vec = rowEncode2D(plain_halfs);

// Reduce the result to the range [-prime_mod/2, prime_mod/2]
  vector<int64_t> plain_vec_mod;
  for(auto val:plain_vec){
    plain_vec_mod.push_back(((val % data.prime_mod) + data.prime_mod) % data.prime_mod);
  }

  return plain_vec_mod;
}

/*
  Extracts the final result vector from the HE result after decryption and decoding.
*/
int64_t *fc_postprocess(vector<vector<int64_t>> &plains, 
                         const FCMetadata &data) {

// Rotate and sum computation to add up all the partial sums in the ciphertext
  
  if(data.skip_he_ras)
    for(int ot = 0; ot < data.o_tiles; ot++)
      plains.at(ot) = rotate_and_sum(plains.at(ot), data);

  int64_t *result = new int64_t[data.filter_h];
    
  int num_rows = data.output_size;
  int rem_rows = data.filter_h % data.output_size;

  for(int ot = 0; ot < data.o_tiles; ot++){
    if ((ot == data.o_tiles - 1) && (rem_rows != 0)) num_rows = rem_rows;
    else num_rows = data.output_size;
    for (int row = 0; row < num_rows; row++) {
      int curr_set = row / data.num_pt;
      result[ot * data.output_size + row] =
        plains.at(ot)[(row % data.num_pt) + data.input_size * curr_set];
    }
  }
  return result;
}

/*
  Extracts the final result vector from the HE result after decryption.
*/
int64_t *fc_postprocess(vector<Plaintext> &pt, const FCMetadata &data) {

  vector<vector<int64_t>> 
    result(data.o_tiles, vector<int64_t>(data.slot_count));
  for(int ot = 0; ot < data.o_tiles; ot++){
    result.at(ot) = pt.at(ot)->GetPackedValue();
  }
  return fc_postprocess(result, data);
}

/*
  Extracts the final result vector from the HE result.
*/
int64_t *fc_postprocess(vector<Ciphertext<DCRTPoly>> &ct, 
                        const FCMetadata &data,
                        CryptoContext<DCRTPoly> &cc, 
                        PrivateKey<DCRTPoly> &secretKey) {
  vector<Plaintext> result(data.o_tiles);
  for(int ot = 0; ot < data.o_tiles; ot++){
    cc->Decrypt(secretKey, ct.at(ot), &result.at(ot));
  }
  return fc_postprocess(result, data);
}

/*
  Initializes the FCFIeld object with parameters from CryptFlow2.
*/
FCField::FCField(int ring_dim, int mult_depth, int64_t prime_mod, bool verbose) {
  CCParams<CryptoContextBGVRNS> parameters;

  data.min = -1; // -(prime_mod / 2);
  data.max = +1; // (prime_mod / 2) - 1;
  data.prime_mod = prime_mod;
  
  parameters.SetPlaintextModulus(data.prime_mod);
  parameters.SetMultiplicativeDepth(mult_depth);
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

  if(verbose){
    cout << "[FCField] HE Parameters: " << endl;
    cout << "+ ring_dim   : " << data.slot_count << endl;
    cout << "+ mult_depth : " << mult_depth << endl;
    cout << "+ prime_mod  : " << data.prime_mod << endl;
  }
}

/* not used
  Initializes the FCFIeld object with selected parameters (coeff_modulus and 
  slot_count).

FCField::FCField(int party, NetIO *io, vector<int> coeff_modulus, 
                 size_t slot_count, bool verbose) {
  this->party = party;
  this->io = io;
  this->slot_count = slot_count;
  generate_context(slot_count, coeff_modulus, context, evaluator, 
                   encoder, verbose);
  exchange_keys(party, io, slot_count, context, encoder, encryptor, 
                decryptor, gal_keys, zero, verbose);
}

FCField::~FCField() {
  free_keys(party, encryptor, decryptor, evaluator, encoder, gal_keys, zero);
}
*/

void FCField::configure(bool verbose) {

  data.filter_size = data.filter_h * data.filter_w;
  data.input_size  = ceil2Pow(data.filter_w);
  data.output_size = ceil2Pow(data.filter_h);

  data.i_tiles = ceil((data.input_size * 2.0) / data.slot_count);
  data.o_tiles = ceil(((double) data.output_size)/ data.slot_count);

  if(data.i_tiles > 1)
    data.input_size = data.slot_count / 2;

  if(data.o_tiles > 1)
    data.output_size = data.slot_count;


  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = data.slot_count / data.input_size;

  // How many total ciphertexts we'll need
  data.num_pt = ceil((float) data.output_size / data.pack_num);

  if(verbose){
    cout << "[configure] FCMetadata:" << endl;
    cout << "+ filter_size: " << data.filter_size << endl;
    cout << "+ filter_h   : " << data.filter_h << endl;
    cout << "+ filter_w   : " << data.filter_w << endl;
    cout << "+ input_size : " << data.input_size << endl;
    cout << "+ output_size: " << data.output_size << endl;
    cout << "+ pack_num   : " << data.pack_num << endl;
    cout << "+ num_pt     : " << data.num_pt << endl;
    cout << "+ i_tiles    : " << data.i_tiles << endl;
    cout << "+ o_tiles    : " << data.o_tiles << endl;
  }
}

vector<int64_t> FCField::ideal_functionality(int64_t *vec,
                                              int64_t **matrix) {
  vector<int64_t> result(data.filter_h);
  for (int row = 0; row < data.filter_h; row++) {
    for (int idx = 0; idx < data.filter_w; idx++) {
      int64_t partial = vec[idx] * matrix[row][idx];
      result[row] = result[row] + partial;
    }
  }
  return result;
}

/*
  Performs matrix multiplication on the shares of input matrices A and B
  and returns the result in C.

  There are 2 parties (p1 and p2) that each hold a share of the input matrices,
  i.e. p1 holds A1 and B1, and p2 holds A2 and B2 such that:
      A = A1 + A2 % primemod and B = B1 + B2 % primemod

  Each party recieves the result of the computation in C1 and C2, such that:
      A @ B = C = C1 + C2 % primemod

  Note: A2 is set to 0 since this is assumed to be solely p1's private data.

  options is a vector of bools that control the behavior of the computation.
*/
void FCField::matrix_multiplication(int32_t num_rows, int32_t common_dim,
                                    int32_t num_cols,
                                    vector<vector<int64_t>> &A,
                                    vector<vector<int64_t>> &B,
                                    vector<vector<int64_t>> &C,
                                    vector<bool> options) {

  assert(num_cols == 1);
  data.filter_h = num_rows;
  data.filter_w = common_dim;

  bool verify_output, verbose, use_tiling;

// Set options
  int num_opts = options.size();
  if(num_opts > 0) verify_output     = options.at(0); else verify_output     = false;
  if(num_opts > 1) verbose           = options.at(1); else verbose           = false;
  if(num_opts > 2) data.mul_then_rot = options.at(2); else data.mul_then_rot = false;
  if(num_opts > 3) data.rot_one_step = options.at(3); else data.rot_one_step = false;
  if(num_opts > 4) data.pre_comp_ntt = options.at(4); else data.pre_comp_ntt = false;
  if(num_opts > 5) data.use_symm_key = options.at(5); else data.use_symm_key = false;
  if(num_opts > 6) data.skip_he_ras  = options.at(6); else data.skip_he_ras  = true;
  if(num_opts > 7) data.enable_ss    = options.at(7); else data.enable_ss    = true;
  if(num_opts > 8) data.print_times  = options.at(8); else data.print_times  = true;
  if(num_opts > 9) use_tiling        = options.at(9); else use_tiling        = false;

  if(verbose){
    cout << "[matrix_multiplication] Options: " << boolalpha << endl;
    cout << "+ verify_output: " << verify_output     << endl;
    cout << "+ verbose      : " << verbose           << endl;
    cout << "+ mul_then_rot : " << data.mul_then_rot << endl;
    cout << "+ rot_one_step : " << data.rot_one_step << endl;
    cout << "+ pre_comp_ntt : " << data.pre_comp_ntt << endl;
    cout << "+ use_symm_key : " << data.use_symm_key << endl;
    cout << "+ skip_he_ras  : " << data.skip_he_ras  << endl;
    cout << "+ enable_ss    : " << data.enable_ss    << endl;
    cout << "+ print_times  : " << data.print_times  << endl;
    cout << "+ use_tiling   : " << use_tiling        << endl;
  }

/*
  SEALContext *context_;
  Encryptor *encryptor_;
  Decryptor *decryptor_;
  Evaluator *evaluator_;
  BatchEncoder *encoder_;
  GaloisKeys *gal_keys_;
  Ciphertext *zero_;

  size_t slot_count_ = 
    min(
      (int) SEAL_POLY_MOD_DEGREE_MAX,
      max(
        (int) POLY_MOD_DEGREE, 
        max(
          2 * next_pow2(common_dim), 
          next_pow2(num_rows))));

  if ((!use_tiling) && (this->slot_count != slot_count_)) {
      this->slot_count = slot_count_;
      vector<int> coeff_modulus = data.mul_then_rot ? 
                                  GET_COEFF_MOD_HLK() : GET_COEFF_MOD_CF2();
      generate_new_keys(party, io, slot_count_, coeff_modulus, context_, encryptor_, 
                        decryptor_, evaluator_, encoder_, gal_keys_, zero_);
  } else {
    context_ = this->context;
    encryptor_ = this->encryptor;
    decryptor_ = this->decryptor;
    evaluator_ = this->evaluator;
    encoder_ = this->encoder;
    gal_keys_ = this->gal_keys;
    zero_ = this->zero;
  }
*/
  
  int64_t prime_mod = data.prime_mod;
  
  configure(verbose);
  
  auto sw = StopWatch();
  double prep_mat_time{}, prep_noise_time{}, processing_time{};

  // if (party == BOB) { // BOB is Client (p2)
    vector<int64_t> vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      vec[i] = B[i][0]; // B2
    }
    if (verbose)
      cout << "[Client] Vector (B) Generated" << endl;

    auto cts = preprocess_enc_vec(vec.data(), data, cc, keys.publicKey);
    
/*
    for (size_t i = 0; i < data.i_tiles; i++)
    {
      if(data.use_symm_key){
        auto ct = encryptor_->encrypt_symmetric(vec_pt.at(i));
        send_ciphertext(io, ct);
      } else {
        auto ct = encryptor_->encrypt(vec_pt.at(i));
        send_ciphertext(io, ct);
      }
    }
*/
    
    if (verbose)
      cout << "[Client] Vector (B2) processed and sent" << endl;

/*
    vector<Ciphertext> enc_result(data.o_tiles);
    for(int ot = 0; ot < data.o_tiles; ot++){
      recv_ciphertext(io, *context_, enc_result.at(ot));
    }

    // Client receives the result C2 = A @ B2 + R

    auto HE_result = fc_postprocess(enc_result, data, *encoder_, *decryptor_);
    if (verbose)
      cout << "[Client] Result (C2  = A @ B2 + R) received and decrypted" << endl;

    for (int i = 0; i < num_rows; i++) {
      C[i][0] = HE_result[i]; // C2
    }
    if (verify_output)
      verify(&vec, nullptr, C); // Client passes B2, 0, C2

    delete[] HE_result;
  } 
  
  else{ // party == ALICE (Server/p1)
*/
    
    if(data.print_times) sw.lap();

    vector<int64_t *> matrix_mod_p(num_rows);
    vector<int64_t *> matrix(num_rows);
    for (int i = 0; i < num_rows; i++) {
      matrix_mod_p[i] = new int64_t[common_dim];
      matrix[i] = new int64_t[common_dim];
      for (int j = 0; j < common_dim; j++) {
        matrix_mod_p[i][j] = neg_mod((int64_t)A[i][j], (int64_t)prime_mod);
        int64_t val = (int64_t)A[i][j];
        if (val > int64_t(prime_mod/2)) {
          val = val - prime_mod;
        }
        matrix[i][j] = val;
        // matrix[i][j] = A[i][j];
      }
    }

/*
  The server's share of the input matrix (A). This computation is independent 
  of the input vector (B1) and can be performed before the online phase. 
  Once computed, the result can be stored for multiple online queries with 
  the same matrix.
*/
    
    auto encoded_mat = preprocess_matrix(matrix_mod_p.data(), data, cc);

    if(data.print_times){
      prep_mat_time = sw.lap();
      cout << "[Server] preprocess_matrix runtime: " << prep_mat_time << endl;
    }
    
    if (verbose)
      cout << "[Server] Matrix (A) generated and processed" << endl;

// The server's random mask (R) for secret sharing the HE result

    int data_min = data.enable_ss ? data.min : 0;
    int data_max = data.enable_ss ? data.max : 0;
    vector<int64_t> sec_share_vec = gen2D_UID_int64(1, num_rows, data_min, data_max).at(0);

    cout << "[Server] sec_share_vec: "; print1D(sec_share_vec, 1, 10);

    int64_t *secret_share = sec_share_vec.data();

    auto random_noise = fc_preprocess_noise(secret_share, data);

    vector<Plaintext> noise_pt(data.o_tiles);
    for (size_t ot = 0; ot < data.o_tiles; ot++)
      noise_pt.at(ot) = cc->MakePackedPlaintext(random_noise.at(ot));

    vector<Ciphertext<DCRTPoly>> noise_ct(data.o_tiles);
    if(data.mul_then_rot == false)
      for(int ot = 0; ot < data.o_tiles; ot++)
        noise_ct.at(ot) = cc->Encrypt(keys.publicKey, noise_pt.at(ot));

    if(data.skip_he_ras)
      secret_share = fc_postprocess(random_noise, data);

    if(data.print_times){
      prep_noise_time = sw.lap();
      cout << "[Server] fc_preprocess_noise runtime: " << prep_noise_time << endl;
      cout << "[Server] Total offline pre-processing time: ";
      cout << prep_mat_time + prep_noise_time << endl;
    }

    if (verbose)
      cout << "[Server] Random mask (R) generated and processed" << endl;

/*
// Server receives the client's input vector (B1)
    vector<Ciphertext> cts(data.i_tiles);
    for (int it = 0; it < data.i_tiles; it++) {
      recv_ciphertext(io, *context_, cts.at(it));
    }

    if (verbose)
      cout << "[Server] Vector (B1) received" << endl;

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, cts[0], "before FC Online");
#endif
*/

// Server computes C1 = A @ B1 + R
    auto res_ct = fc_online(cts, encoded_mat, data, cc,
                               zero, noise_ct, noise_pt
                              , keys.secretKey
                               );

/*
#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result[0], "after FC Online");
#endif
    for(int ot = 0; ot < data.o_tiles; ot++){
      parms_id_type parms_id = HE_result.at(ot).parms_id();
      shared_ptr<const SEALContext::ContextData> context_data =
          context_->get_context_data(parms_id);
      flood_ciphertext(HE_result.at(ot), context_data, SMUDGING_BITLEN);
    }

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result[0], "after noise flooding");
#endif

    for(int ot = 0; ot < data.o_tiles; ot++)
      evaluator_->mod_switch_to_next_inplace(HE_result.at(ot));

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result[0], "after mod-switch");
#endif
*/ 
    
    if(data.print_times){
      processing_time = sw.lap();
      cout << "[Server] Total online processing time: " << processing_time << endl;
    }

/*
    for(int ot = 0; ot < data.o_tiles; ot++)
      send_ciphertext(io, HE_result.at(ot));
*/

    if (verbose)
      cout << "[Server] Result (C2  = A @ B2 + R) computed and sent" << endl;

// Server computes A @ B1 in the plain with its share of the input (B1)
    auto exp_res = ideal_functionality(vec.data(), matrix.data());

    auto res_vec = fc_postprocess(res_ct, data, cc, keys.secretKey);
    if (verbose)
      cout << "[Client] Result (C2  = A @ B2 + R) received and decrypted" << endl;

    vector<int64_t> act_res(num_rows);
// Server's secret share of the result C1 = A @ B1 - R
    bool pass = true;
    for (int i = 0; i < num_rows; i++) {
      C[i][0] = neg_mod((int64_t)res_vec[i] - (int64_t)secret_share[i],
                        (int64_t)prime_mod);
      if (C[i][0] > int64_t(prime_mod/2)) {
        C[i][0] = C[i][0] - prime_mod;
      }
      act_res[i] = C[i][0];
      if (act_res[i] != exp_res[i]) {
        pass = false;
      }
    }

    cout << "[matmul] act_res: "; print1D(act_res, 2, 5);
    cout << "[matmul] exp_res: "; print1D(exp_res, 2, 5);
    cout << "[matmul] pass: " << boolalpha << pass << endl;

/*
//     if (verify_output)
//       verify(&vec, &matrix, C); // Server passes B1, A, C1

//     for (int i = 0; i < num_rows; i++) {
//       delete[] matrix_mod_p[i];
//       delete[] matrix[i];
//     }
//     delete[] secret_share;
//   }
//   if (slot_count > POLY_MOD_DEGREE) {
//     free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_,
//               zero_);
//   }
*/
}

/*
void FCField::verify(vector<int64_t> *vec, vector<int64_t *> *matrix,
                     vector<vector<int64_t>> &C) {
  if (party == BOB) {
    io->send_data(vec->data(), data.filter_w * sizeof(int64_t));
    io->flush();
    for (int i = 0; i < data.filter_h; i++) {
      io->send_data(C[i].data(), sizeof(int64_t));
    }
  } else // party == ALICE
  {
    vector<int64_t> vec_0(data.filter_w);
    io->recv_data(vec_0.data(), data.filter_w * sizeof(int64_t));
    for (int i = 0; i < data.filter_w; i++) {
      vec_0[i] = (vec_0[i] + (*vec)[i]) % prime_mod;
    }
    auto result = ideal_functionality(vec_0.data(), matrix->data());

    vector<vector<int64_t>> C_0(data.filter_h);
    for (int i = 0; i < data.filter_h; i++) {
      C_0[i].resize(1);
      io->recv_data(C_0[i].data(), sizeof(int64_t));
      C_0[i][0] = (C_0[i][0] + C[i][0]) % prime_mod;
    }
    bool pass = true;
    for (int i = 0; i < data.filter_h; i++) {
      if (neg_mod(result[i], (int64_t)prime_mod) != (int64_t)C_0[i][0]) {
        pass = false;
      }
    }
    if (pass)
      cout << "[Server] " << GREEN << "Successful Operation" << RESET << endl;
    else {
      cout << "[Server] " << RED << "Failed Operation" << RESET << endl;
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
}
*/