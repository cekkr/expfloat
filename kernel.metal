#include <metal_stdlib>
using namespace metal;

// Kernel per moltiplicazione matriciale ExpFloat
kernel void exp_float_matmul(
    device const char* a_exp [[ buffer(0) ]],
    device const char* b_exp [[ buffer(1) ]],
    device char* c_exp [[ buffer(2) ]],
    constant int& M [[ buffer(3) ]],
    constant int& N [[ buffer(4) ]],
    constant int& K [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    const int row = gid.y;
    const int col = gid.x;
    
    if (row < M && col < N) {
        // Per ExpFloat, la moltiplicazione è la somma degli esponenti
        // Ma l'addizione richiede conversione al dominio lineare
        
        // Array temporaneo per memorizzare i risultati parziali
        // Limitato a 32 elementi per questioni di performance
        const int LOCAL_SIZE = 32;
        float partial_results[LOCAL_SIZE];
        int count = 0;
        
        // Esegui la moltiplicazione (somma di esponenti)
        for (int i = 0; i < K; i++) {
            // Leggi gli esponenti
            char exp_a = a_exp[row * K + i];
            char exp_b = b_exp[i * N + col];
            
            // Somma esponenti (equivalente a moltiplicazione)
            int exp_sum = exp_a + exp_b;
            
            // Limita l'esponente al range valido (-128 a 127)
            exp_sum = max(-128, min(127, exp_sum));
            
            // Converti al dominio lineare (2^exp)
            // Ottimizzato per Metal
            float linear_value = exp2(float(exp_sum));
            
            // Accumula risultati parziali
            if (count < LOCAL_SIZE) {
                partial_results[count++] = linear_value;
            } else {
                // Somma e riduci l'array quando pieno
                float sum = 0.0f;
                for (int j = 0; j < LOCAL_SIZE; j++) {
                    sum += partial_results[j];
                }
                partial_results[0] = sum + linear_value;
                count = 1;
            }
        }
        
        // Somma i risultati parziali rimanenti
        float final_sum = 0.0f;
        for (int j = 0; j < count; j++) {
            final_sum += partial_results[j];
        }
        
        // Converti il risultato finale in esponente
        // log2 per tornare al formato esponente
        int final_exp = int(round(log2(final_sum)));
        
        // Limita il risultato al range valido
        final_exp = max(-128, min(127, final_exp));
        
        // Scrivi il risultato
        c_exp[row * N + col] = char(final_exp);
    }
}

// Kernel per accelerare la conversione da float a ExpFloat
kernel void float_to_exp_float(
    device const float* input [[ buffer(0) ]],
    device char* output [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]],
    uint size [[ buffer(2) ]]
) {
    if (id < size) {
        float value = input[id];
        
        // Gestisci il caso dello zero
        if (fabs(value) < 1e-38) {
            output[id] = -128; // Minimo valore rappresentabile
            return;
        }
        
        // Calcola l'esponente
        int exp = int(round(log2(fabs(value))));
        
        // Limita al range valido
        exp = max(-128, min(127, exp));
        
        // Memorizza l'esponente
        output[id] = char(exp);
    }
}

// Kernel per accelerare la conversione da ExpFloat a float
kernel void exp_float_to_float(
    device const char* input [[ buffer(0) ]],
    device float* output [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]],
    uint size [[ buffer(2) ]]
) {
    if (id < size) {
        // Leggi l'esponente
        char exp = input[id];
        
        // Converti: 1.0 * 2^exp
        float value = exp2(float(exp));
        
        // Scrivi il risultato
        output[id] = value;
    }
}

// Kernel per addizione ExpFloat
kernel void exp_float_add(
    device const char* a_exp [[ buffer(0) ]],
    device const char* b_exp [[ buffer(1) ]],
    device char* c_exp [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]],
    uint size [[ buffer(3) ]]
) {
    if (id < size) {
        // Leggi esponenti
        char exp_a = a_exp[id];
        char exp_b = b_exp[id];
        
        // Converti al dominio lineare
        float val_a = exp2(float(exp_a));
        float val_b = exp2(float(exp_b));
        
        // Somma nel dominio lineare
        float sum = val_a + val_b;
        
        // Converti risultato in esponente
        int result_exp = int(round(log2(sum)));
        
        // Limita al range valido
        result_exp = max(-128, min(127, result_exp));
        
        // Scrivi il risultato
        c_exp[id] = char(result_exp);
    }
}

// Ottimizzazione per batch processing di modelli neurali
kernel void exp_float_batch_matmul(
    device const char* weights_exp [[ buffer(0) ]],
    device const char* activations_exp [[ buffer(1) ]],
    device char* output_exp [[ buffer(2) ]],
    constant int& batch_size [[ buffer(3) ]],
    constant int& in_features [[ buffer(4) ]],
    constant int& out_features [[ buffer(5) ]],
    uint3 gid [[ thread_position_in_grid ]]
) {
    const int batch = gid.z;
    const int out_idx = gid.y;
    
    if (batch < batch_size && out_idx < out_features) {
        // Per ogni elemento del batch e output, calcola il dot product
        
        // Array per risultati parziali (limitato per performance)
        const int LOCAL_SIZE = 32;
        float partial_results[LOCAL_SIZE];
        int count = 0;
        
        for (int in_idx = 0; in_idx < in_features; in_idx++) {
            // Indici per accedere ai tensori
            int act_idx = batch * in_features + in_idx;
            int weight_idx = out_idx * in_features + in_idx;
            
            // Leggi esponenti
            char act_exp = activations_exp[act_idx];
            char weight_exp = weights_exp[weight_idx];
            
            // Somma esponenti (moltiplicazione)
            int exp_sum = act_exp + weight_exp;
            exp_sum = max(-128, min(127, exp_sum));
            
            // Converti al dominio lineare
            float linear_value = exp2(float(exp_sum));
            
            // Accumula risultati parziali
            if (count < LOCAL_SIZE) {
                partial_results[count++] = linear_value;
            } else {
                // Somma e riduci quando l'array è pieno
                float sum = 0.0f;
                for (int j = 0; j < LOCAL_SIZE; j++) {
                    sum += partial_results[j];
                }
                partial_results[0] = sum + linear_value;
                count = 1;
            }
        }
        
        // Somma i risultati parziali rimanenti
        float final_sum = 0.0f;
        for (int j = 0; j < count; j++) {
            final_sum += partial_results[j];
        }
        
        // Converti in esponente
        int final_exp = int(round(log2(final_sum)));
        final_exp = max(-128, min(127, final_exp));
        
        // Scrivi il risultato
        output_exp[batch * out_features + out_idx] = char(final_exp);
    }
}