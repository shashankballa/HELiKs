#include <iostream>
#include <vector>
#include <string>

#include "openfhe.h"

#include "fc-field.h"
#include "conv-field.h"

using namespace std;
using namespace lbcrypto;

void matmul_test(int argc, char* argv[]){
    
    vector<int> dims = {1024, 1024};
    
    if(argc > 2) dims.at(0) = atoi(argv[2]);
    if(argc > 3) dims.at(1) = atoi(argv[3]);

    int numops = 10;
    vector<bool> options = {
                            true , // options[0] -> verify_output
                            true , // options[1] -> verbose
                            false, // options[2] -> mul_then_rot
                            false, // options[3] -> rot_one_step
                            false, // options[4] -> pre_comp_ntt
                            false, // options[5] -> use_symm_key
                            true , // options[6] -> skip_he_ras
                            true , // options[7] -> enable_ss
                            true , // options[8] -> print_times
                            false  // options[9] -> use_tiling
                            };

    string options_str;
    if(argc > 3) options_str = argv[4];

    for(int i = 0; i < options_str.size(); i++){
        options.at(i) = options_str.at(i) == '1';
    }

    bool verbose = options.at(1);
    bool mul_then_rot = options.at(2);
    bool use_tiling = options.at(9);

    int ring_dim;
    // if(argc > 5) ring_dim = atoi(argv[5]);

    int32_t num_rows   = dims.at(0);
    int32_t common_dim = dims.at(1);
    int32_t num_cols   = 1;

    cout << "[matmul_test] (" << num_rows << " x " << common_dim;
    cout << ") Options: " << options_str << endl;


    ring_dim = mul_then_rot ? 1 << 13 : 1 << 14;

    if(!use_tiling){
        ring_dim = 
            max(ring_dim,
                max(2 * ceil2Pow(common_dim), 
                    ceil2Pow(num_rows)));
    }
    
    
    int32_t prime_mod = 65537; //4293918721, 2147483649, 536903681, 65537
    if(argc > 5) prime_mod = atoi(argv[5]);

    FCField *fc_he;

    if(!mul_then_rot){
        fc_he = new FCField(ring_dim, 2, prime_mod, verbose);
    }else{
        fc_he = new FCField(ring_dim, 1, prime_mod, verbose);
    }

    auto A = gen2D_UID_int64(num_rows, common_dim, fc_he->data.min, fc_he->data.max);
    auto B = gen2D_UID_int64(common_dim, num_cols, fc_he->data.min, fc_he->data.max);

    vector<vector<int64_t>> AB(num_rows, vector<int64_t>(num_cols, 0));

    for(int i = 0; i < num_rows; i++){
        for(int j = 0; j < num_cols; j++){
            for(int k = 0; k < common_dim; k++){
                AB[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    vector<vector<int64_t>> C(num_rows, vector<int64_t>(num_cols, 0));

    fc_he->matrix_multiplication(num_rows, common_dim, num_cols, A, B, C, options);
}

void conv_test(int argc, char* argv[]){
    
    int image_h = 56;
    int inp_chans = 64;
    int filter_h = 3;
    int out_chans = 64;
    int pad_l = 0;
    int pad_r = 0;
    int stride = 2;
    int filter_precision = 12;
    int slots_count_ = 8192;
    
    if(argc > 2)  image_h   = atoi(argv[2]);
    if(argc > 3)  filter_h  = atoi(argv[3]);
    if(argc > 4)  inp_chans = atoi(argv[4]);
    if(argc > 5)  out_chans = atoi(argv[5]);
    if(argc > 6)  stride    = atoi(argv[6]);
    if(argc > 7){ pad_l     = atoi(argv[7]); pad_r = pad_l;}

    int numops = 3;
    vector<bool> options = {
                            true , // options[0] -> verify_output
                            true , // options[1] -> verbose
                            false, // options[2] -> use_heliks
                            true   // options[3] -> print_counts
                            };


    string options_str;
    if(argc > 8) options_str = argv[8];

    for(int i = 0; i < options_str.size(); i++){
        options.at(i) = options_str.at(i) == '1';
    }
    bool verify_output = options.at(0);
    bool verbose = options.at(1);
    bool use_heliks = options.at(2);

    int ring_dim = use_heliks ? 1 << 13 : 1 << 14;

    int _paddedH = image_h + pad_l + pad_r;
    int _limitH  = filter_h + ((_paddedH - filter_h) / stride) * stride;

    for (int s_row = 0; s_row < stride; s_row++) {
        for (int s_col = 0; s_col < stride; s_col++) {
            int lH  = ((_limitH - s_row + stride - 1) / stride);
            int lW  = lH;
            int lFH = ((filter_h - s_row + stride - 1) / stride);
            int lFW = lFH;

            ring_dim = max(ring_dim, 2 * ceil2Pow(lH * lW));
        }
    }

    cout << "[conv_test] Parameters: " << endl;
    cout << "+ Image  : " << inp_chans << "x" << image_h  <<  "x" << image_h << endl;
    cout << "+ Filter : " << out_chans << "x" << inp_chans 
                   << "x" <<  filter_h << "x" << filter_h << endl;
    cout << "+ Stride : " << stride    << "x" << stride   << endl;
    cout << "+ Padding: " << pad_l     << "x" << pad_r    << endl;
    

    ConvField conv_he(ring_dim, use_heliks);
    
    int32_t prime_mod = conv_he.get_prime_mod();
    int scaling_factor = 8;
    int data_max = conv_he.data.max; // (prime_mod/scaling_factor)-1;
    int data_min = conv_he.data.min; // -(prime_mod/scaling_factor);

// void Conv(ConvField &he_conv, ConvField &he_conv_SB, int32_t H, int32_t CI, int32_t FH, int32_t CO,
//           int32_t zPadHLeft, int32_t zPadHRight, int32_t strideH, bool verify_output, bool verbose) {
    
// Conv(*he_conv, *he_conv_SB, image_h, inp_chans, filter_h, out_chans, pad_l, pad_r, stride,
//     verify_output, verbose);

    int H = image_h;
    int CI = inp_chans;
    int FH = filter_h;
    int CO = out_chans;
    int zPadHLeft = pad_l;
    int zPadHRight = pad_r;
    int strideH = stride;

    int newH = 1 + (H + zPadHLeft + zPadHRight - FH) / strideH;
    int N = 1;
    int W = H;
    int FW = FH;
    int zPadWLeft = zPadHLeft;
    int zPadWRight = zPadHRight;
    int strideW = strideH;
    int newW = newH;
    vector<vector<vector<vector<int64_t>>>> inputArr(N);   // N, H, W, CI
    vector<vector<vector<vector<int64_t>>>> filterArr(FH); // FH, FW, CI, CO
    vector<vector<vector<vector<int64_t>>>> outArr(N);     // N, newH, newW, CO

    // PRG128 prg;
    for (int i = 0; i < N; i++) {
        outArr[i].resize(newH);
        for (int j = 0; j < newH; j++) {
            outArr[i][j].resize(newW);
            for (int k = 0; k < newW; k++) {
                outArr[i][j][k].resize(CO);
            }
        }
    }
    // if (party == ALICE) {
    for (int i = 0; i < FH; i++) {
        filterArr[i].resize(FW);
        for (int j = 0; j < FW; j++) {
            filterArr[i][j].resize(CI);
            for (int k = 0; k < CI; k++) {
                filterArr[i][j][k] = gen2D_UID_int64(1, CO, data_min, data_max).at(0);
                // prg.random_data(filterArr[i][j][k].data(), CO * sizeof(uint64_t));
                // for (int h = 0; h < CO; h++) {
                //     // filterArr[i][j][k][h] = (i+1)*1000 + (j+1)*100 + (k+1)*10 + (h+1);
                //     filterArr[i][j][k][h] = ((int64_t)filterArr[i][j][k][h]) >> (64 - filter_precision);
                // }
            }
        }
    }
    // }
    for (int i = 0; i < N; i++) {
        inputArr[i].resize(H);
        for (int j = 0; j < H; j++) {
            inputArr[i][j] = gen2D_UID_int64(W, CI, data_min, data_max);
            // inputArr[i][j].resize(W);
            // for (int k = 0; k < W; k++) {
            //     inputArr[i][j][k].resize(CI);
            //     prg.random_mod_p<uint64_t>(inputArr[i][j][k].data(), CI, prime_mod);
            // }
        }
    }
    // INIT_TIMER;
    // uint64_t comm_start_NTT = he_conv_SB.io->counter;
    // START_TIMER;

    conv_he.convolution(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft,
                        zPadWRight, strideH, strideW, inputArr, filterArr, outArr,
                        options);

    // STOP_TIMER("Conv-NTT " + roles[party-1]);
    // uint64_t comm_end_NTT = he_conv_SB.io->counter;
    // cout << "Conv-NTT " << roles[party-1] << " Sent: " << (comm_end_NTT - comm_start_NTT) / (1.0 * (1ULL << 20))
    //     << endl;
}

int main(int argc, char* argv[]){

    bool run_mm = true;

    if(argc > 1) run_mm = (atoi(argv[1]) == 0);

    if(run_mm){
        matmul_test(argc, argv);
    }else{
        conv_test(argc, argv);
    }

    return 0;
}