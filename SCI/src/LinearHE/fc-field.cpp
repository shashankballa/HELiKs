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

#include "LinearHE/fc-field.h"

using namespace std;
using namespace seal;
using namespace sci;

/*
  Pre-processes the input vector
*/
vector<vector<uint64_t>> 
  preprocess_vec(const uint64_t *input, const FCMetadata &data){
  // Create copies of the input vector to fill the ciphertext appropiately.
  // Pack using powers of two for easy rotations later
  vector<vector<uint64_t>> result(data.i_tiles);
  int num_cols = data.input_size;
  int rem_cols = data.filter_w % data.input_size;

  for (size_t it = 0; it < data.i_tiles; it++)
  {
    vector<uint64_t> pod_matrix(data.slot_count, 0ULL);
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
  preprocess_vec(const uint64_t *input, const FCMetadata &data,
                 BatchEncoder &batch_encoder) {

  vector<vector<uint64_t>> pod_matrices = preprocess_vec(input, data);

  vector<Plaintext> pts(data.i_tiles);
  for (size_t i = 0; i < data.i_tiles; i++)
    batch_encoder.encode(pod_matrices.at(i), pts.at(i));
  return pts;
}

/*
  Pre-processes the input vector into a ciphertext.
*/
vector<Ciphertext> 
  preprocess_vec(const uint64_t *input, const FCMetadata &data,
                 Encryptor &encryptor, BatchEncoder &batch_encoder) {

  vector<Plaintext> pts = preprocess_vec(input, data, batch_encoder);

  vector<Ciphertext> cts(data.i_tiles);
  for (size_t i = 0; i < data.i_tiles; i++)
    encryptor.encrypt(pts.at(i), cts.at(i));
  return cts;
}

/*
  Pre-processes the input matrix into plaintexts.
*/
vector<vector<vector<Plaintext>>>
  preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data,
                    BatchEncoder &batch_encoder) {

  assert(data.mul_then_rot == false
      && data.pre_comp_ntt == false);

  int num_cols = data.input_size;
  int rem_cols = data.filter_w % data.input_size;

  int num_rows = data.output_size;
  int rem_rows = data.filter_h % data.output_size;
  
  vector<vector<vector<vector<uint64_t>>>> 
    result_vec(data.o_tiles, vector<vector<vector<uint64_t>>>(data.i_tiles));

  for(int ot = 0; ot < data.o_tiles; ot++){
    if ((ot == data.o_tiles - 1) && (rem_rows != 0)) num_rows = rem_rows;
    else num_rows = data.output_size;
    for(int it = 0; it < data.i_tiles; it++){
      // Pack the filter in alternating order of needed ciphertexts. This way we
      // rotate the input once per ciphertext
      vector<vector<uint64_t>> mat_pack(data.num_pt,
                                        vector<uint64_t>(data.slot_count, 0ULL));
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
      vector<vector<uint64_t>> mat_diag(data.num_pt,
                                        vector<uint64_t>(data.slot_count, 0ULL));
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
      result_vec.at(ot).at(it) = mat_diag;
    }
  }

  vector<vector<vector<Plaintext>>> 
    result(data.o_tiles, vector<vector<Plaintext>>(data.i_tiles));

  for(int ot = 0; ot < data.o_tiles; ot++){
    for(int it = 0; it < data.i_tiles; it++){
      vector<Plaintext> enc_mat(data.num_pt);
      for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
        batch_encoder.encode(result_vec.at(ot).at(it).at(pt_idx), 
                             enc_mat.at(pt_idx));
      }
      result.at(ot).at(it) = enc_mat;
    }
  }
  return result;
}

/* 
  Pre-processes the input matrix into Plaintexts such that multiply can be 
  performed before the rotation in fc_online and ransforms the plaintexts to 
  NTT before returning the results. 
*/
vector<vector<vector<Plaintext>>>
  preprocess_matrix(const uint64_t *const *matrix, const FCMetadata &data,
                    BatchEncoder &batch_encoder, Evaluator &evaluator,
                    parms_id_type parms_id) {

  int num_cols = data.input_size;
  int rem_cols = data.filter_w % data.input_size;

  int num_rows = data.output_size;
  int rem_rows = data.filter_h % data.output_size;
  
  vector<vector<vector<vector<uint64_t>>>> 
    result_vec(data.o_tiles, vector<vector<vector<uint64_t>>>(data.i_tiles));

  for(int ot = 0; ot < data.o_tiles; ot++){
    if ((ot == data.o_tiles - 1) && (rem_rows != 0)) num_rows = rem_rows;
    else num_rows = data.output_size;
    for(int it = 0; it < data.i_tiles; it++){
      // Pack the filter in alternating order of needed ciphertexts. This way we
      // rotate the input once per ciphertext
      vector<vector<uint64_t>> mat_pack(data.num_pt,
                                        vector<uint64_t>(data.slot_count, 0ULL));
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
      vector<vector<uint64_t>> mat_diag(data.num_pt,
                                        vector<uint64_t>(data.slot_count, 0ULL));
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
          vector<vector<uint64_t>> diag2 = {slice1D(mat_diag.at(pt_idx), 0, data.slot_count/2 - 1),
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
        if(data.pre_comp_ntt){
// trasnform the plaintexts to NTT
          evaluator.transform_to_ntt_inplace(enc_mat[pt_idx], parms_id);
          if(data.print_times) sw.lap(3);
        }
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

/*
  Generates a masking vector of random noise that will be applied to parts of 
  the ciphertext that contain leakage.
*/
vector<vector<uint64_t>>
  fc_preprocess_noise(const uint64_t *secret_share,
                               const FCMetadata &data) {

  // Sample randomness into vector
  vector<vector<uint64_t>> 
    noise (data.o_tiles, vector<uint64_t>(data.slot_count, 0ULL));
  if(!data.enable_ss) return noise;
  
  PRG128 prg;
  for(int ot = 0; ot < data.o_tiles; ot++)
    prg.random_mod_p<uint64_t>(noise.at(ot).data(), 
                              data.slot_count, prime_mod);

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
  fc_preprocess_noise(const uint64_t *secret_share,
                      const FCMetadata &data,
                      BatchEncoder &batch_encoder) {
  // Sample randomness into vector
  vector<vector<uint64_t>> noise = fc_preprocess_noise(secret_share, data);
  vector<Plaintext> noise_pts(data.o_tiles);
  for(int ot = 0; ot < data.o_tiles; ot++)
    batch_encoder.encode(noise.at(ot), noise_pts.at(ot));
  return noise_pts;
}

/*
  Generates a ciphertext of random noise that will be applied to parts of the 
  ciphertext that contain leakage.
  
  NOTE: The result need not be encrypted and can be added to the ciphertext 
  as a plaintext.
*/
vector<Ciphertext> 
  fc_preprocess_noise(const uint64_t *secret_share,
                      const FCMetadata &data, Encryptor &encryptor,
                      BatchEncoder &batch_encoder) {

  vector<Ciphertext> noise_cts(data.o_tiles);
  auto noise_pts = fc_preprocess_noise(secret_share, data, batch_encoder);
  for(int ot = 0; ot < data.o_tiles; ot++)
    encryptor.encrypt(noise_pts.at(ot), noise_cts.at(ot));
  return noise_cts;
}

/*
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Original function in CryptFlow2 with trivial tiling.
*/
vector<Ciphertext> fc_online_cf(vector<Ciphertext> &cts, 
                        vector<vector<vector<Plaintext>>> &enc_mats,
                        const FCMetadata &data, Evaluator &evaluator,
                        GaloisKeys &gal_keys, Ciphertext &zero,
                        vector<Ciphertext> &noise_ct) {
  
  assert(data.mul_then_rot == false 
      && data.rot_one_step == false 
      && data.pre_comp_ntt == false);

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<Ciphertext> tiled_result(data.o_tiles);

  for(int ot = 0; ot < data.o_tiles; ot++){
    for(int it = 0; it < data.i_tiles; it++){
      auto ct = cts.at(it);
      auto enc_mat = enc_mats.at(ot).at(it);
      Ciphertext result = zero;
      // For each matrix ciphertext, rotate the input vector once and multiply + add
      Ciphertext tmp;
      for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
        if (!enc_mat[pt_idx].is_zero()) {
          evaluator.rotate_rows(ct, pt_idx, gal_keys, tmp);
          if(data.print_times) sw.lap(1);
          evaluator.multiply_plain_inplace(tmp, enc_mat[pt_idx]);
          if(data.print_times) sw.lap(0);
          evaluator.add_inplace(result, tmp);
          if(data.print_times) sw.lap(2);
        }
      }
      evaluator.mod_switch_to_next_inplace(result);
      if(data.print_times) sw.lap(5);
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
      if(it == 0){
        tiled_result.at(ot) = result;
      } else {
        evaluator.add_inplace(tiled_result.at(ot), result);
        if(data.print_times) sw.lap(2);
      }
    }

    if(data.enable_ss){
    // Add noise to cover leakage
      evaluator.mod_switch_to_next_inplace(noise_ct.at(ot));
      if(data.print_times) sw.lap(5);
      evaluator.add_inplace(tiled_result.at(ot), noise_ct.at(ot));
      if(data.print_times) sw.lap(2);
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

/*
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Algorithm from HELiKs with multiplication before rotation (sec 4.2) and 
    trivial tiling.
*/
vector<Ciphertext> fc_online_v1(vector<Ciphertext> &cts, 
                        vector<vector<vector<Plaintext>>> &enc_mats,
                        const FCMetadata &data, Evaluator &evaluator,
                        GaloisKeys &gal_keys,
                        vector<Plaintext> &noise_pt
#ifdef HE_DEBUG
                        , Decryptor *decryptor_
#endif
                        ) {

  assert(data.mul_then_rot == true 
      && data.rot_one_step == false 
      && data.pre_comp_ntt == false);

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<Ciphertext> tiled_result(data.o_tiles);

  for(int ot = 0; ot < data.o_tiles; ot++){
    for(int it = 0; it < data.i_tiles; it++){
      auto ct = cts.at(it);
      auto enc_mat = enc_mats.at(ot).at(it);

      Ciphertext result;

      vector<Ciphertext> pSums;
      if (!enc_mat[0].is_zero()) {
        Ciphertext tmp = ct;
        evaluator.multiply_plain_inplace(tmp, enc_mat[0]); 
        if(data.print_times) sw.lap(0);
        pSums.push_back(tmp);
      }

      for (int pt_idx = 1; pt_idx < data.num_pt; pt_idx++) {
        if (!enc_mat[pt_idx].is_zero()) {
          Ciphertext tmp = ct;
          evaluator.multiply_plain_inplace(tmp, enc_mat[pt_idx]); 
          if(data.print_times) sw.lap(0);
          evaluator.rotate_rows_inplace(tmp, pt_idx, gal_keys); 
          if(data.print_times) sw.lap(1);
          pSums.push_back(tmp);
        }
      }

      if(data.num_pt==1){
        result = pSums[0];
      } else {
        evaluator.add_many(pSums, result); 
        if(data.print_times) sw.lap(2);
      }
      
#ifdef HE_DEBUG
      auto noise_before = decryptor_->invariant_noise_budget(result);
#endif

      evaluator.mod_switch_to_next_inplace(result);
      if(data.print_times) sw.lap(5);

#ifdef HE_DEBUG
      auto noise_after = decryptor_->invariant_noise_budget(result);
      if(noise_before - noise_after > 0){
        cout << "[fc_online_v1] noise budget loss: " 
             << noise_before - noise_after << endl;
      }
#endif

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
      if(it == 0){
        tiled_result.at(ot) = result;
      } else {
        evaluator.add_inplace(tiled_result.at(ot), result);
        if(data.print_times) sw.lap(2);
      }
    }

    if(data.enable_ss){
    // Add noise to cover leakage
      evaluator.add_plain_inplace(tiled_result.at(ot), 
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

/*
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Algorithm from HELiKs with multiplication before rotation (sec 4.2),
    1-step rotations (sec 4.3) and tiling (sec 4.5).
*/
vector<Ciphertext> fc_online_v2(vector<Ciphertext> &cts, 
                    vector<vector<vector<Plaintext>>> &enc_mats,
                    const FCMetadata &data, Evaluator &evaluator,
                    GaloisKeys &gal_keys,
                    vector<Plaintext> &noise_pt
#ifdef HE_DEBUG
                    , Decryptor *decryptor_
#endif
                    ) {

  assert(data.mul_then_rot == true 
      && data.rot_one_step == true 
      && data.pre_comp_ntt == false);

  auto sw = StopWatch();
  if(data.print_times) sw.lap();

  vector<Ciphertext> tiled_result(data.o_tiles);

  for(int ot = 0; ot < data.o_tiles; ot++){

    vector<Ciphertext> psums(data.num_pt);

    for (int pt_idx = 0; pt_idx < data.num_pt; pt_idx++) {
      vector<Ciphertext> psums_it = cts;
      for(int it = 0; it < data.i_tiles; it++){
        if (!enc_mats[ot][it][pt_idx].is_zero()) {
          evaluator.multiply_plain_inplace(
            psums_it.at(it), enc_mats[ot][it][pt_idx]);
          if(data.print_times) sw.lap(0);
        }
      }
      evaluator.add_many(psums_it, psums.at(pt_idx));
      if(data.print_times) sw.lap(2);
      
#ifdef HE_DEBUG
      auto noise_before = decryptor_->invariant_noise_budget(psums.at(pt_idx));
#endif

      evaluator.mod_switch_to_next_inplace(psums.at(pt_idx));
      if(data.print_times) sw.lap(5);

#ifdef HE_DEBUG
      auto noise_after = decryptor_->invariant_noise_budget(psums.at(pt_idx));
      if(noise_before - noise_after > 0){
        cout << "[fc_online_v2] pt_idx: " << pt_idx << "; noise budget loss: " 
             << noise_before - noise_after << endl;
      }
#endif

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

/*
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.

  ~ Algorithm from HELiKs with multiplication before rotation (sec 4.2), 1-step 
    rotations (sec 4.3), NTT precomputation (sec 4.4) and tiling (sec 4.5).
*/
vector<Ciphertext> fc_online_v3(vector<Ciphertext> &cts, 
                    vector<vector<vector<Plaintext>>> &enc_mats,
                    const FCMetadata &data, Evaluator &evaluator,
                    GaloisKeys &gal_keys,
                    vector<Plaintext> &noise_pt
#ifdef HE_DEBUG
                    , Decryptor *decryptor_
#endif
                    ) {

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
      if(data.print_times) sw.lap(2);

      evaluator.transform_from_ntt_inplace(psums.at(pt_idx)); 
      if(data.print_times) sw.lap(4);

#ifdef HE_DEBUG
      auto noise_before = decryptor_->invariant_noise_budget(psums.at(pt_idx));
#endif

      evaluator.mod_switch_to_next_inplace(psums.at(pt_idx));
      if(data.print_times) sw.lap(5);

#ifdef HE_DEBUG
      auto noise_after = decryptor_->invariant_noise_budget(psums.at(pt_idx));
      if(noise_before - noise_after > 0){
        cout << "[fc_online_v3] pt_idx: " << pt_idx << "; noise budget loss: " 
             << noise_before - noise_after << endl;
      }
#endif

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

/*
  Performs matrix multiplication of the recieved ciphertext and the server's
  weight matrix. The result is masked by a secret share and returned as a 
  ciphertext.
*/
vector<Ciphertext> fc_online(vector<Ciphertext> &cts, 
                    vector<vector<vector<Plaintext>>> &enc_mats,
                    const FCMetadata &data, Evaluator &evaluator,
                    GaloisKeys &gal_keys,
                    Ciphertext &zero,
                    vector<Ciphertext> &noise_ct,
                    vector<Plaintext>  &noise_pt
#ifdef HE_DEBUG
                    , Decryptor *decryptor_
#endif
                    ) {

  vector<Ciphertext> result(data.o_tiles);

  if(data.mul_then_rot == false 
  && data.rot_one_step == false 
  && data.pre_comp_ntt == false){
    result = fc_online_cf(cts, enc_mats, data, evaluator, gal_keys, zero, noise_ct);
  }

  if(data.mul_then_rot == true
  && data.rot_one_step == false 
  && data.pre_comp_ntt == false){
    result = fc_online_v1(cts, enc_mats, data, evaluator, gal_keys, noise_pt
#ifdef HE_DEBUG
                          , decryptor_
#endif
                          );
  }

  if(data.mul_then_rot == true
  && data.rot_one_step == true 
  && data.pre_comp_ntt == false){
    result = fc_online_v2(cts, enc_mats, data, evaluator, gal_keys, noise_pt
#ifdef HE_DEBUG
                          , decryptor_                    
#endif
                          );
  }

  if(data.mul_then_rot == true
  && data.rot_one_step == true 
  && data.pre_comp_ntt == true){
    result = fc_online_v3(cts, enc_mats, data, evaluator, gal_keys, noise_pt
#ifdef HE_DEBUG
                          , decryptor_
#endif
                          );
  }

  return result;
}

/*
  Rotate and sum computation to add up all the partial sums in the ciphertext
  vector after decryption. This is also applied to the server's secret share
  to get the final result.
*/
vector<uint64_t> rotate_and_sum(vector<uint64_t> plain, const FCMetadata &data) {

  assert(plain.size() == data.slot_count);

// Break the input vector into two halves to mimic BFV/BGV rotaion structure  
  vector<vector<uint64_t>> plain_halfs = {slice1D(plain, 0, data.slot_count/2 - 1),
                                          slice1D(plain, data.slot_count/2,   0  )};

// Rotate all partial sums together
  for (int rot = data.num_pt; rot < data.input_size; rot *= 2) {
    vector<vector<uint64_t>> tmp(2, vector<uint64_t>(data.slot_count/2, 0ULL));
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
                                    % prime_mod) + prime_mod) % prime_mod);
      }
    }
  }

// Print results for debugging
  // cout << "[rotate_and_sum_SB] OUT : "; 
  // print1D(neg_mod_1D(rowEncode2D(plain_halfs), prime_mod), nprint, false);
  
  auto plain_vec = rowEncode2D(plain_halfs);

// Reduce the result to the range [-prime_mod/2, prime_mod/2]
  vector<uint64_t> plain_vec_mod;
  for(auto val:plain_vec){
    plain_vec_mod.push_back(((val % prime_mod) + prime_mod) % prime_mod);
  }

  return plain_vec_mod;
}

/*
  Extracts the final result vector from the HE result after decryption and decoding.
*/
uint64_t *fc_postprocess(vector<vector<uint64_t>> &plains, 
                         const FCMetadata &data) {

// Rotate and sum computation to add up all the partial sums in the ciphertext
  
  if(data.skip_he_ras)
    for(int ot = 0; ot < data.o_tiles; ot++)
      plains.at(ot) = rotate_and_sum(plains.at(ot), data);

  uint64_t *result = new uint64_t[data.filter_h];
    
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
uint64_t *fc_postprocess(vector<Plaintext> &pt, const FCMetadata &data,
                         BatchEncoder &batch_encoder) {

  vector<vector<uint64_t>> 
    result(data.o_tiles, vector<uint64_t>(data.slot_count, 0ULL));
  for(int ot = 0; ot < data.o_tiles; ot++){
    batch_encoder.decode(pt.at(ot), result.at(ot));
  }
  return fc_postprocess(result, data);
}

/*
  Extracts the final result vector from the HE result.
*/
uint64_t *fc_postprocess(vector<Ciphertext> &ct, const FCMetadata &data,
                         BatchEncoder &batch_encoder, Decryptor &decryptor) {
  vector<Plaintext> result(data.o_tiles);
  for(int ot = 0; ot < data.o_tiles; ot++){
    decryptor.decrypt(ct.at(ot), result.at(ot));
  }
  return fc_postprocess(result, data, batch_encoder);
}

/*
  Initializes the FCFIeld object with parameters from CryptFlow2.
*/
FCField::FCField(int party, NetIO *io) {
  this->party = party;
  this->io = io;
  this->slot_count = POLY_MOD_DEGREE;
  generate_new_keys(party, io, slot_count, context, encryptor, decryptor,
                    evaluator, encoder, gal_keys, zero);
}

/*
  Initializes the FCFIeld object with selected parameters (coeff_modulus and 
  slot_count).
*/
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

void FCField::configure(bool verbose) {

  data.slot_count  = slot_count;

  data.filter_size = data.filter_h * data.filter_w;
  data.input_size  = next_pow2(data.filter_w);
  data.output_size = next_pow2(data.filter_h);

  data.i_tiles = ceil((data.input_size * 2.0) / data.slot_count);
  data.o_tiles = ceil(((double) data.output_size)/ data.slot_count);

  if(data.i_tiles > 1)
    data.input_size = data.slot_count / 2;

  if(data.o_tiles > 1)
    data.output_size = data.slot_count;


  // How many columns of matrix we can fit in a single ciphertext
  data.pack_num = slot_count / data.input_size;

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

vector<uint64_t> FCField::ideal_functionality(uint64_t *vec,
                                              uint64_t **matrix) {
  vector<uint64_t> result(data.filter_h, 0ULL);
  for (int row = 0; row < data.filter_h; row++) {
    for (int idx = 0; idx < data.filter_w; idx++) {
      uint64_t partial = vec[idx] * matrix[row][idx];
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
                                    vector<vector<uint64_t>> &A,
                                    vector<vector<uint64_t>> &B,
                                    vector<vector<uint64_t>> &C,
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
  if(num_opts > 6) data.skip_he_ras  = options.at(6); else data.skip_he_ras  = false;
  if(num_opts > 7) data.enable_ss    = options.at(7); else data.enable_ss    = false;
  if(num_opts > 8) data.print_times  = options.at(8); else data.print_times  = false;
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
  
  configure(verbose);
  
  auto sw = StopWatch();
  double prep_mat_time{}, prep_noise_time{}, processing_time{};

  if (party == BOB) { // BOB is Client (p2)
    vector<uint64_t> vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      vec[i] = B[i][0]; // B2
    }
    if (verbose)
      cout << "[Client] Vector (B2) Generated" << endl;

    auto vec_pt = preprocess_vec(vec.data(), data, *encoder_);

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

    if (verbose)
      cout << "[Client] Vector (B2) processed and sent" << endl;

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
    if(data.print_times) sw.lap();
/*
  All of the server's work until the client's input is recieved can be done is
  independent of the client's input and encryption keys. This includes:
    - Preprocessing the server's private matrix (A)
    - Generating the server's random mask (R) for secret sharing the HE result
*/
    vector<uint64_t> vec(common_dim);
    for (int i = 0; i < common_dim; i++) {
      vec[i] = B[i][0]; // B1
    }
    if (verbose)
      cout << "[Server] Vector (B1) Generated" << endl;
    vector<uint64_t *> matrix_mod_p(num_rows);
    vector<uint64_t *> matrix(num_rows);
    for (int i = 0; i < num_rows; i++) {
      matrix_mod_p[i] = new uint64_t[common_dim];
      matrix[i] = new uint64_t[common_dim];
      for (int j = 0; j < common_dim; j++) {
        matrix_mod_p[i][j] = neg_mod((int64_t)A[i][j], (int64_t)prime_mod);
        int64_t val = (int64_t)A[i][j];
        if (val > int64_t(prime_mod/2)) {
          val = val - prime_mod;
        }
        matrix[i][j] = val;
      }
    }

/*
  The server's share of the input matrix (A). This computation is independent 
  of the input vector (B1) and can be performed before the online phase. 
  Once computed, the result can be stored for multiple online queries with 
  the same matrix.
*/
    auto encoded_mat = preprocess_matrix(matrix_mod_p.data(), data, *encoder_, 
                                        *evaluator_, context_->first_parms_id());

    if(data.print_times){
      prep_mat_time = sw.lap();
      cout << "[Server] preprocess_matrix runtime: " << prep_mat_time << endl;
    }
    
    if (verbose)
      cout << "[Server] Matrix (A) generated and processed" << endl;

// The server's random mask (R) for secret sharing the HE result
    PRG128 prg;
    uint64_t *secret_share = new uint64_t[num_rows];
    prg.random_mod_p<uint64_t>(secret_share, num_rows, prime_mod);

    auto random_noise = fc_preprocess_noise(secret_share, data);

    vector<Plaintext> noise_pt(data.o_tiles);
    for (size_t ot = 0; ot < data.o_tiles; ot++)
      encoder_->encode(random_noise.at(ot), noise_pt.at(ot));

    vector<Ciphertext> noise_ct(data.o_tiles);
    if(data.mul_then_rot == false)
      for(int ot = 0; ot < data.o_tiles; ot++)
        encryptor_->encrypt(noise_pt.at(ot), noise_ct.at(ot));

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

// Server computes C1 = A @ B1 + R
    auto HE_result = fc_online(cts, encoded_mat, data, *evaluator_, *gal_keys_,
                               *zero_, noise_ct, noise_pt
#ifdef HE_DEBUG
                                , decryptor_
#endif
                               );

#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result[0], "after FC Online");
#endif

    if(!data.mul_then_rot){
      for(int ot = 0; ot < data.o_tiles; ot++){
        evaluator_->mod_switch_to_next_inplace(HE_result.at(ot));
      }
#ifdef HE_DEBUG
    PRINT_NOISE_BUDGET(decryptor_, HE_result[0], "after mod-switch");
#endif
    }
    
    if(data.print_times){
      processing_time = sw.lap();
      cout << "[Server] Total online processing time: " << processing_time << endl;
    }

    for(int ot = 0; ot < data.o_tiles; ot++)
      send_ciphertext(io, HE_result.at(ot));
      
    if (verbose)
      cout << "[Server] Result (C2  = A @ B2 + R) computed and sent" << endl;

// Server computes A @ B1 in the plain with its share of the input (B1)
    auto result = ideal_functionality(vec.data(), matrix.data());

// Server's secret share of the result C1 = A @ B1 - R
    for (int i = 0; i < num_rows; i++) {
      C[i][0] = neg_mod((int64_t)result[i] - (int64_t)secret_share[i],
                        (int64_t)prime_mod);
    }
    if (verify_output)
      verify(&vec, &matrix, C); // Server passes B1, A, C1

    for (int i = 0; i < num_rows; i++) {
      delete[] matrix_mod_p[i];
      delete[] matrix[i];
    }
    delete[] secret_share;
  }
  if (slot_count > POLY_MOD_DEGREE) {
    free_keys(party, encryptor_, decryptor_, evaluator_, encoder_, gal_keys_,
              zero_);
  }
}

void FCField::verify(vector<uint64_t> *vec, vector<uint64_t *> *matrix,
                     vector<vector<uint64_t>> &C) {
  if (party == BOB) {
    io->send_data(vec->data(), data.filter_w * sizeof(uint64_t));
    io->flush();
    for (int i = 0; i < data.filter_h; i++) {
      io->send_data(C[i].data(), sizeof(uint64_t));
    }
  } else // party == ALICE
  {
    vector<uint64_t> vec_0(data.filter_w);
    io->recv_data(vec_0.data(), data.filter_w * sizeof(uint64_t));
    for (int i = 0; i < data.filter_w; i++) {
      vec_0[i] = (vec_0[i] + (*vec)[i]) % prime_mod;
    }
    auto result = ideal_functionality(vec_0.data(), matrix->data());

    vector<vector<uint64_t>> C_0(data.filter_h);
    for (int i = 0; i < data.filter_h; i++) {
      C_0[i].resize(1);
      io->recv_data(C_0[i].data(), sizeof(uint64_t));
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