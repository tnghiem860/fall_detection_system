#pragma once
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Khởi tạo mạng 4G
bool sim_mqtt_init(void);
bool sim_mqtt_wait_network(void);

// Kích hoạt PDP context (TCP/IP Stack)
bool sim_mqtt_start_pdp(void);

// Kết nối MQTT broker
bool sim_mqtt_connect_broker(void);

// Kiểm tra trạng thái
bool sim_mqtt_is_connected(void);

// FIX #2: Trả về bool để caller biết thành công hay thất bại
// max_retries = 0 dùng giá trị mặc định (3 lần trước khi reinit GPRS)
bool sim_mqtt_ensure_connection(int max_retries);

// FIX #2: Topic được truyền thực sự — không bị ignore
bool sim_mqtt_publish(const char *topic, const char *payload);

// Các hàm publish chuyên biệt — mỗi hàm dùng đúng topic riêng
bool sim_mqtt_publish_fall_alert(const char *device_id, float confidence,
                                 float lat, float lon, bool has_location);
bool sim_mqtt_publish_location(const char *device_id, float lat, float lon,
                               const char *source);
bool sim_mqtt_publish_status(const char *device_id, bool is_fall, float confidence);

bool sim_mqtt_disconnect(void);

// ── Cấu hình Topic ───────────────────────────────────────────────────
// FIX #2: Mỗi loại message có topic riêng, Flutter app subscribe đúng channel
#define MQTT_TOPIC          "esp32/gps/flutter"
#define MQTT_LOCATION_TOPIC "esp32/gps/flutter"
#define MQTT_ALERT_TOPIC    "esp32/gps01/flutter"
#define MQTT_STATUS_TOPIC   "esp32/gps02/flutter"

#ifdef __cplusplus
}
#endif
