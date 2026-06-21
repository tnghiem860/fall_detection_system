#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include "esp_err.h"
#include <stdbool.h>

// Params khớp với quantization_params.json và mean_std.json
#define AI_WINDOW_SIZE   150
#define AI_NUM_CHANNELS  3        // acc X, Y, Z
#define AI_TARGET_FS     25       // Hz

// Quantization (từ quantization_params.json)
#define AI_INPUT_SCALE      0.00804556f
#define AI_INPUT_ZP         (3)
#define AI_OUTPUT_SCALE     0.00390625f
#define AI_OUTPUT_ZP        (-128)

// Normalization (từ mean_std.json)
#define NORM_X_MIN          (-16.000000f)
#define NORM_X_MAX          (15.992188f)
typedef struct {
    bool   is_fall;
    float  confidence;    // 0.0 – 1.0
    int8_t raw_out[2];    // [ADL, Fall]
} ai_result_t;

esp_err_t ai_model_init(void);

esp_err_t ai_model_run(const float window[][AI_NUM_CHANNELS],
                       ai_result_t *result);

#ifdef __cplusplus
}
#endif
