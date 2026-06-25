#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Cấu hình Firebase ─────────────────────────────────────────────
#define FIREBASE_PROJECT_ID   "sis-fall"
#define FIREBASE_API_KEY      "AIzaSyCLcLjmODx8oOUe26YhATqyb4Jd1fg4ZZs"
#define FIREBASE_HOST         "firestore.googleapis.com"

/**
 * @brief Khởi tạo module Firebase (thiết lập HTTP URL cho EG800K)
 *        Gọi một lần sau khi PDP context đã active.
 */
void sim_firebase_init(void);

/**
 * @brief Kích hoạt PDP context để có IP (cần cho HTTP)
 * @return true nếu thành công
 */
bool sim_firebase_start_pdp(void);

/**
 * @brief Cập nhật trạng thái thiết bị lên Firestore (upsert document).
 *        Path: /devices/{device_id}
 *        Dùng Firestore commit API (POST) — không cần PATCH.
 */
bool sim_firebase_update_status(const char *device_id,
                                 float lat,  float lon,
                                 float gps_accuracy,
                                 int   battery_pct,
                                 int   rssi,
                                 bool  fall_detected,
                                 float confidence,
                                 long  timestamp);

/**
 * @brief Ghi sự kiện té ngã mới vào collection fall_events (auto-ID).
 *        Path: /fall_events  (POST → Firestore tự tạo document ID)
 */
bool sim_firebase_push_fall_event(const char *device_id,
                                   float lat,  float lon,
                                   float confidence,
                                   int   battery_pct,
                                   long  timestamp);

#ifdef __cplusplus
}
#endif
