#include <zephyr/kernel.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/settings/settings.h>
#include <zephyr/sys/printk.h>

#include <cstring>
#include <cstdio>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

/* ------------------------------------------------------------------
 * Sampling
 *
 * The window, the rate and the axis count are fixed here rather than
 * read from the impulse, because a "bring your own model" deployment
 * does not describe them reliably. Edge Impulse wraps an uploaded
 * .tflite in a pass-through DSP block and then labels the flat input
 * buffer however it likes: the same 375-value window has been emitted
 * as 111x3, as 375x1 at 1 kHz with sensor "unknown", and as 125x3 at
 * 62.5 Hz, across successive deploys of the same model. Deriving the
 * sampling loop from that description meant the loop changed shape
 * every time Studio changed its mind.
 *
 * These four constants are the contract instead, and they must match
 * the training notebook:
 *   125 frames x 3 axes (accX, accY, accZ), interleaved
 *   16 ms per frame -> 62.5 Hz, a 2.00 s window
 *
 * Edge Impulse expects the buffer INTERLEAVED as
 *   x0,y0,z0, x1,y1,z1, ...
 * which is the order the model was trained on, because flattening a
 * (125, 3) array in the notebook varies the axis fastest.
 *
 * Windows overlap by 75%, so a new prediction covering the last 2 s
 * is produced roughly every 0.5 s.
 * ------------------------------------------------------------------ */
#define FRAME_COUNT      125
#define AXES_PER_FRAME   3

#define SAMPLE_PERIOD_MS 16

#define FEATURE_COUNT    (FRAME_COUNT * AXES_PER_FRAME)

#define FRAME_STRIDE (FRAME_COUNT / 4)
#define FRAME_KEEP   (FRAME_COUNT - FRAME_STRIDE)

/* The one thing still worth checking against the deployed model: that
 * the window we sample is exactly what the network expects, and that
 * the pass-through hands all of it over.
 *
 * The second half of this catches an upload configured with too few
 * axes, which is otherwise silent: the DSP block quietly selects a
 * subset, the firmware still compiles, and the model is fed a
 * fraction of the data it was trained on. An earlier deploy did
 * exactly that, passing 111 of 333 values.
 */
BUILD_ASSERT(
    EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE == FEATURE_COUNT,
    "The deployed model does not take a 125 frame x 3 axis window. "
    "Check the impulse input settings, or update the constants above "
    "to match the training notebook.");

BUILD_ASSERT(
    EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE == EI_CLASSIFIER_NN_INPUT_FRAME_SIZE,
    "The processing block is not passing the whole window to the "
    "network, so the upload selected only some of the axes.");

static const struct device *const accel =
    DEVICE_DT_GET(DT_ALIAS(accel0));

static float features[FEATURE_COUNT];

static int64_t next_sample_ms;

/* ------------------------------------------------------------------
 * Status LED
 *
 * Thingy:53 aliases: led0 = red, led1 = green, led2 = blue.
 * The blink rate doubles as a status code, so the board can be
 * diagnosed with no console attached:
 *
 *   500 ms - running and advertising
 *   100 ms - Bluetooth or the accelerometer failed to come up
 * ------------------------------------------------------------------ */
static const struct gpio_dt_spec status_led =
    GPIO_DT_SPEC_GET(DT_ALIAS(led0), gpios);

#define BLINK_OK_MS    500
#define BLINK_ERROR_MS 100

static volatile int blink_period_ms = BLINK_ERROR_MS;

#define HEARTBEAT_STACK_SIZE 512
#define HEARTBEAT_PRIORITY   7

static void heartbeat_thread(void *, void *, void *)
{
    if (!gpio_is_ready_dt(&status_led)) {
        printk("Status LED device not ready\n");
        return;
    }

    int err = gpio_pin_configure_dt(
        &status_led,
        GPIO_OUTPUT_INACTIVE);

    if (err) {
        printk("LED configure failed (%d)\n", err);
        return;
    }

    while (true) {
        gpio_pin_toggle_dt(&status_led);
        k_sleep(K_MSEC(blink_period_ms));
    }
}

K_THREAD_DEFINE(
    heartbeat_tid,
    HEARTBEAT_STACK_SIZE,
    heartbeat_thread,
    NULL, NULL, NULL,
    HEARTBEAT_PRIORITY,
    0,
    0);

/* ------------------------------------------------------------------
 * Advertising: Eddystone-URL beacon
 *
 * The device keeps its fixed name from CONFIG_BT_DEVICE_NAME. The
 * classification result travels in the Eddystone service data as the
 * URL body, so a scanner shows:
 *
 *   Complete Local Name: EdgeImpulse-01
 *   Eddystone URL:       http://www.wave
 *
 * The name lives in the SCAN RESPONSE, not the advertising packet:
 * flags + Eddystone UUID + service data already use 23 of the 31
 * available bytes, and "EdgeImpulse-01" needs another 16.
 * ------------------------------------------------------------------ */
#define EDDYSTONE_FRAME_URL       0x10
#define EDDYSTONE_SCHEME_HTTP_WWW 0x00

/*
 * Calibrated RSSI at 0 m. Informational only -- it tells a scanner how
 * to estimate distance and has no effect on discovery.
 */
#define EDDYSTONE_TX_POWER_0M ((uint8_t)((int8_t)-19))

#define EDDYSTONE_HEADER_LEN 5   /* UUID(2) + frame + tx power + scheme */
#define EDDYSTONE_URL_MAX    17  /* Eddystone-URL spec limit */

static uint8_t eddystone_svc_data[EDDYSTONE_HEADER_LEN + EDDYSTONE_URL_MAX] = {
    0xAA, 0xFE,                 /* Eddystone service UUID, little endian */
    EDDYSTONE_FRAME_URL,
    EDDYSTONE_TX_POWER_0M,
    EDDYSTONE_SCHEME_HTTP_WWW,  /* expands to "http://www." */
};

static const uint8_t eddystone_uuid[] = { 0xAA, 0xFE };

/*
 * Matches the reference beacon: BR/EDR not supported, without the
 * general-discoverable bit. Add BT_LE_AD_GENERAL here if you also want
 * the board to appear in phone OS Bluetooth menus, not just in
 * scanner apps.
 */
static const uint8_t ad_flags = BT_LE_AD_NO_BREDR;

/* ad[2].data_len is patched as the label changes length. */
static struct bt_data ad[] = {
    BT_DATA(BT_DATA_FLAGS,      &ad_flags,          sizeof(ad_flags)),
    BT_DATA(BT_DATA_UUID16_ALL, eddystone_uuid,     sizeof(eddystone_uuid)),
    BT_DATA(BT_DATA_SVC_DATA16, eddystone_svc_data, EDDYSTONE_HEADER_LEN),
};

static const struct bt_data sd[] = {
    BT_DATA(
        BT_DATA_NAME_COMPLETE,
        CONFIG_BT_DEVICE_NAME,
        sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

/*
 * Connectable, so the board can be opened in nRF Connect and OTA DFU
 * still works. The name comes from sd[] above, so do not set
 * BT_LE_ADV_OPT_USE_NAME as well -- that would add a second copy.
 */
static const struct bt_le_adv_param adv_param =
    BT_LE_ADV_PARAM_INIT(
        BT_LE_ADV_OPT_CONNECTABLE,
        BT_GAP_ADV_FAST_INT_MIN_2,
        BT_GAP_ADV_FAST_INT_MAX_2,
        NULL);

static bool advertising_active = false;

static void set_eddystone_url(const char *label)
{
    size_t len = strlen(label);

    if (len > EDDYSTONE_URL_MAX) {
        len = EDDYSTONE_URL_MAX;
    }

    memcpy(&eddystone_svc_data[EDDYSTONE_HEADER_LEN], label, len);

    ad[2].data_len = EDDYSTONE_HEADER_LEN + len;
}

static int start_advertising()
{
    int err = bt_le_adv_start(
        &adv_param,
        ad, ARRAY_SIZE(ad),
        sd, ARRAY_SIZE(sd));

    if (err) {
        printk("Advertising start failed (%d)\n", err);
        blink_period_ms = BLINK_ERROR_MS;
        return err;
    }

    advertising_active = true;
    blink_period_ms = BLINK_OK_MS;

    printk("Advertising as %s\n", CONFIG_BT_DEVICE_NAME);

    return 0;
}

/*
 * Rewrites the advertising payload in place. Unlike stop/start this
 * keeps the advertiser and its address stable, so the device does not
 * flicker in and out of a scanner list.
 */
static void publish_result(const char *label)
{
    set_eddystone_url(label);

    if (!advertising_active) {
        start_advertising();
        return;
    }

    int err = bt_le_adv_update_data(
        ad, ARRAY_SIZE(ad),
        sd, ARRAY_SIZE(sd));

    if (err) {
        printk("Advertising update failed (%d)\n", err);
    }
}

/*
 * Connectable advertising stops by itself once a central connects, so
 * it has to be restarted on disconnect -- otherwise the board goes
 * silent after the first time someone opens it in nRF Connect.
 */
static void on_connected(struct bt_conn *conn, uint8_t err)
{
    if (err) {
        printk("Connection failed (%u)\n", err);
        return;
    }

    advertising_active = false;

    printk("Connected\n");
}

static void on_disconnected(struct bt_conn *conn, uint8_t reason)
{
    printk("Disconnected (reason %u)\n", reason);

    advertising_active = false;

    start_advertising();
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected = on_connected,
    .disconnected = on_disconnected,
};

static void print_identity()
{
    bt_addr_le_t addrs[CONFIG_BT_ID_MAX];
    size_t count = ARRAY_SIZE(addrs);

    bt_id_get(addrs, &count);

    for (size_t i = 0; i < count; i++) {
        char buf[BT_ADDR_LE_STR_LEN];

        bt_addr_le_to_str(&addrs[i], buf, sizeof(buf));

        printk("BLE identity %u: %s\n", (unsigned)i, buf);
    }
}

/* ------------------------------------------------------------------
 * Accelerometer
 * ------------------------------------------------------------------ */
/*
 * Fills `frames` interleaved x,y,z frames starting at `dst`.
 *
 * This mirrors firmware-nordic-thingy53's ei_inertial_sensor.cpp --
 * same device (accel0 = ADXL362), same SENSOR_CHAN_ACCEL_XYZ read,
 * same sensor_value_to_double() conversion. Zephyr reports m/s^2 and
 * the reference firmware forwards it unscaled, so the training set is
 * in m/s^2 and no conversion belongs here.
 */
static int sample_into(float *dst, size_t frames)
{
    for (size_t f = 0; f < frames; f++) {

        k_sleep(K_TIMEOUT_ABS_MS(next_sample_ms));
        next_sample_ms += SAMPLE_PERIOD_MS;

        int err = sensor_sample_fetch(accel);

        if (err) {
            printk("Accel fetch failed (%d)\n", err);
            return err;
        }

        struct sensor_value raw[AXES_PER_FRAME];

        err = sensor_channel_get(accel, SENSOR_CHAN_ACCEL_XYZ, raw);

        if (err) {
            printk("Accel read failed (%d)\n", err);
            return err;
        }

        for (size_t a = 0; a < AXES_PER_FRAME; a++) {
            dst[f * AXES_PER_FRAME + a] =
                (float)sensor_value_to_double(&raw[a]);
        }
    }

    return 0;
}

static int get_signal_data(
    size_t offset,
    size_t length,
    float *out_ptr)
{
    memcpy(
        out_ptr,
        features + offset,
        length * sizeof(float));

    return 0;
}

int main(void)
{
    printk("Booted, starting Bluetooth\n");

    int err = bt_enable(NULL);

    if (err) {
        printk("Bluetooth init failed (%d)\n", err);
        printk("Is the network core flashed? "
               "Program merged_domains.hex, not merged.hex\n");
        return 0;
    }

    /*
     * CONFIG_BT_SETTINGS=y makes bt_enable() defer identity creation
     * until settings_load() runs. Without this call the stack never
     * becomes ready and every bt_le_adv_start() returns -EAGAIN.
     */
    if (IS_ENABLED(CONFIG_BT_SETTINGS)) {
        err = settings_load();

        if (err) {
            printk("settings_load failed (%d)\n", err);
        }
    }

    if (!bt_is_ready()) {
        printk("Bluetooth stack not ready after settings_load\n");
        return 0;
    }

    printk("Bluetooth ready\n");

    print_identity();

    /* Advertise straight away, before the first window is collected. */
    set_eddystone_url("UNKNOWN");
    start_advertising();

    if (!device_is_ready(accel)) {
        printk("Accelerometer %s not ready\n", accel->name);
        blink_period_ms = BLINK_ERROR_MS;
        return 0;
    }

    printk("Sampling %s: %d frames x %d axes every %d ms "
           "(%d.%02d Hz, %d ms window), new result every %d ms\n",
           accel->name,
           FRAME_COUNT,
           AXES_PER_FRAME,
           SAMPLE_PERIOD_MS,
           1000 / SAMPLE_PERIOD_MS,
           (100000 / SAMPLE_PERIOD_MS) % 100,
           FRAME_COUNT * SAMPLE_PERIOD_MS,
           FRAME_STRIDE * SAMPLE_PERIOD_MS);

    next_sample_ms = k_uptime_get() + SAMPLE_PERIOD_MS;

    if (sample_into(features, FRAME_COUNT)) {
        blink_period_ms = BLINK_ERROR_MS;
        return 0;
    }

    while (true)
    {
        signal_t signal;

        signal.total_length = FEATURE_COUNT;
        signal.get_data = get_signal_data;

        ei_impulse_result_t result;

        EI_IMPULSE_ERROR res =
            run_classifier(
                &signal,
                &result,
                false);

        if (res != EI_IMPULSE_OK) {
            printk("Inference failed (%d)\n", res);
        }
        else {
            float best_score = 0.0f;
            const char *best_label = "UNKNOWN";

            printk("Predictions:\n");

            for (size_t ix = 0;
                 ix < EI_CLASSIFIER_LABEL_COUNT;
                 ix++)
            {
                printk(
                    "%s : %.3f\n",
                    result.classification[ix].label,
                    result.classification[ix].value);

                if (result.classification[ix].value >
                    best_score)
                {
                    best_score =
                        result.classification[ix].value;

                    best_label =
                        result.classification[ix].label;
                }
            }

            /*
             * Below the impulse's own confidence threshold the argmax
             * is meaningless -- while the board sits still one class
             * always edges ahead. Report nothing rather than a label
             * the model is not confident about.
             */
            if (best_score < EI_CLASSIFIER_THRESHOLD) {
                best_label = "UNKNOWN";
            }

            printk(
                "Winner: %s (%.2f)\n",
                best_label,
                best_score);

            publish_result(best_label);
        }

        /*
         * Slide the window by FRAME_STRIDE frames: keep the newest
         * FRAME_KEEP frames, sample FRAME_STRIDE new ones. Counts are
         * in frames, so every offset is scaled by AXES_PER_FRAME.
         */
        memmove(
            features,
            features + (size_t)FRAME_STRIDE * AXES_PER_FRAME,
            (size_t)FRAME_KEEP * AXES_PER_FRAME * sizeof(float));

        if (sample_into(
                features + (size_t)FRAME_KEEP * AXES_PER_FRAME,
                FRAME_STRIDE)) {
            blink_period_ms = BLINK_ERROR_MS;
            return 0;
        }
    }

    return 0;
}
