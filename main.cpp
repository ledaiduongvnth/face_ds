#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <nvdsmeta_schema.h>
#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "FaceDetector.h"

#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define PGIE_NET_WIDTH 640
#define PGIE_NET_HEIGHT 480
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define NVDS_USER_FRAME_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))
#define USER_ARRAY_SIZE 14
#define USER_ARRAY_SIZE_CAP 30
#define MSCONV_CONFIG_FILE "/home/d/CLionProjects/demo_kafka/config/dstest4_msgconv_config.txt"

#define GST_CAPS_FEATURES_NVMM "memory:NVMM"
#define TRACKER_CONFIG_FILE "../dstest2_tracker_config.txt"
#define CONFIG_GROUP_TRACKER "tracker"
#define CONFIG_GROUP_TRACKER_WIDTH "tracker-width"
#define CONFIG_GROUP_TRACKER_HEIGHT "tracker-height"
#define CONFIG_GROUP_TRACKER_LL_CONFIG_FILE "ll-config-file"
#define CONFIG_GROUP_TRACKER_LL_LIB_FILE "ll-lib-file"
#define CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS "enable-batch-process"
#define CONFIG_GPU_ID "gpu-id"
#define MAX_TIME_STAMP_LEN 32

#define CHECK_ERROR(error) \
    if (error) { \
        g_printerr ("Error while parsing config file: %s\n", error->message); \
        goto done; \
    }
//add kafka

static gchar *cfg_file = NULL;
//static gchar *input_file = "/home/d/CLionProjects/demo_kafka/sample_720p.h264";
static gchar *input_file = "/home/d/CLionProjects/demo_kafka/sample_720p.h264";

static gchar *topic = "test123";
static gchar *conn_str = "172.16.10.30;9092;test123";
static gchar *proto_lib = "/home/d/CLionProjects/demo_kafka/libnvds_kafka_proto.so";
static gint schema_type = 0;
static gboolean display_off = FALSE;


GOptionEntry entries[] = {
        {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME, &cfg_file,
                                                              "Set the adaptor config file. Optional if connection string has relevant  details.", NULL},
        {"input-file", 'i', 0, G_OPTION_ARG_FILENAME, &input_file,
                                                              "Set the input H264 file", NULL},
        {"topic", 't', 0, G_OPTION_ARG_STRING, &topic,
                                                              "Name of message topic. Optional if it is part of connection string or config file.", NULL},
        {"conn-str", 0, 0, G_OPTION_ARG_STRING, &conn_str,
                                                              "Connection string of backend server. Optional if it is part of config file.", NULL},
        {"proto-lib", 'p', 0, G_OPTION_ARG_STRING, &proto_lib,
                                                              "Absolute path of adaptor library", NULL},
        {"schema", 's', 0, G_OPTION_ARG_INT, &schema_type,
                                                              "Type of message schema (0=Full, 1=minimal), default=0", NULL},
        {"no-display", 0, 0, G_OPTION_ARG_NONE, &display_off, "Disable display", NULL},
        {NULL}
};
static void generate_ts_rfc3339 (char *buf, int buf_size)
{
    time_t tloc;
    struct tm tm_log;
    struct timespec ts;
    char strmsec[6]; //.nnnZ\0

    clock_gettime(CLOCK_REALTIME,  &ts);
    memcpy(&tloc, (void *)(&ts.tv_sec), sizeof(time_t));
    gmtime_r(&tloc, &tm_log);
    strftime(buf, buf_size,"%Y-%m-%dT%H:%M:%S", &tm_log);
    int ms = ts.tv_nsec/1000000;
    g_snprintf(strmsec, sizeof(strmsec),".%.3dZ", ms);
    strncat(buf, strmsec, buf_size);
}

static gpointer meta_copy_func (gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;
    NvDsEventMsgMeta *dstMeta = NULL;

    dstMeta = static_cast<NvDsEventMsgMeta *>(g_memdup(srcMeta, sizeof(NvDsEventMsgMeta)));

    if (srcMeta->ts)
        dstMeta->ts = g_strdup (srcMeta->ts);

    if (srcMeta->sensorStr)
        dstMeta->sensorStr = g_strdup (srcMeta->sensorStr);

    if (srcMeta->objSignature.size > 0) {
        dstMeta->objSignature.signature = static_cast<gdouble *>(g_memdup(srcMeta->objSignature.signature,
                                                                          srcMeta->objSignature.size));
        dstMeta->objSignature.size = srcMeta->objSignature.size;
    }

    if(srcMeta->objectId) {
        dstMeta->objectId = g_strdup (srcMeta->objectId);
    }

    if (srcMeta->extMsgSize > 0) {

        NvDsPersonObject *srcObj = (NvDsPersonObject *) srcMeta->extMsg;
        NvDsPersonObject *obj = (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));

        obj->age = srcObj->age;

        if (srcObj->gender)
            obj->gender = g_strdup (srcObj->gender);
        if (srcObj->cap)
            obj->cap = g_strdup (srcObj->cap);
        if (srcObj->hair)
            obj->hair = g_strdup (srcObj->hair);
        if (srcObj->apparel)
            obj->apparel = g_strdup (srcObj->apparel);
        dstMeta->extMsg = obj;
        dstMeta->extMsgSize = sizeof (NvDsPersonObject);
    }

    return dstMeta;
}

static void meta_free_func (gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    NvDsEventMsgMeta *srcMeta = (NvDsEventMsgMeta *) user_meta->user_meta_data;

    g_free (srcMeta->ts);
    g_free (srcMeta->sensorStr);

    if (srcMeta->objSignature.size > 0) {
        g_free (srcMeta->objSignature.signature);
        srcMeta->objSignature.size = 0;
    }

    if(srcMeta->objectId) {
        g_free (srcMeta->objectId);
    }

    if (srcMeta->extMsgSize > 0) {

        NvDsPersonObject *obj = (NvDsPersonObject *) srcMeta->extMsg;

        if (obj->gender)
            g_free (obj->gender);
        if (obj->cap)
            g_free (obj->cap);
        if (obj->hair)
            g_free (obj->hair);
        if (obj->apparel)
            g_free (obj->apparel);
        g_free (srcMeta->extMsg);
        srcMeta->extMsgSize = 0;
    }
    g_free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
}

static void
generate_person_meta (gpointer data, gchar *face_vector)
{
    NvDsPersonObject *obj = (NvDsPersonObject *) data;
    obj->age = 100;
    obj->cap = g_strdup ("none");
    obj->hair = g_strdup ("datdat");
    obj->gender = g_strdup ("male");
    obj->apparel= g_strdup ("formal");
}

static void
generate_event_msg_meta (gpointer data, gint class_id, NvDsObjectMeta * obj_params)
{
    NvDsEventMsgMeta *meta = (NvDsEventMsgMeta *) data;
    meta->sensorId = 0;
    meta->placeId = 0;
    meta->moduleId = 0;
    meta->sensorStr = g_strdup ("sensor-0");

    meta->ts = (gchar *) g_malloc0 (MAX_TIME_STAMP_LEN + 1);
    meta->objectId = (gchar *) g_malloc0 (MAX_LABEL_SIZE);

    strncpy(meta->objectId, obj_params->obj_label, MAX_LABEL_SIZE);

    generate_ts_rfc3339(meta->ts, MAX_TIME_STAMP_LEN);

    /*
     * This demonstrates how to attach custom objects.
     * Any custom object as per requirement can be generated and attached
     * like NvDsVehicleObject / NvDsPersonObject. Then that object should
     * be handled in payload generator library (nvmsgconv.cpp) accordingly.
     */

    meta->type = NVDS_EVENT_ENTRY;
    meta->objType = NVDS_OBJECT_TYPE_PERSON;
    meta->objClassId = 11111111111111111111111.1;

    NvDsPersonObject *obj = (NvDsPersonObject *) g_malloc0 (sizeof (NvDsPersonObject));
    generate_person_meta (obj,meta->videoPath);

    meta->extMsg = obj;
    meta->extMsgSize = sizeof (NvDsPersonObject);
}

static gchar *
get_absolute_file_path (gchar *cfg_file_path, gchar *file_path)
{
    gchar abs_cfg_path[PATH_MAX + 1];
    gchar *abs_file_path;
    gchar *delim;

    if (file_path && file_path[0] == '/') {
        return file_path;
    }

    if (!realpath (cfg_file_path, abs_cfg_path)) {
        g_free (file_path);
        return NULL;
    }

    // Return absolute path of config file if file_path is NULL.
    if (!file_path) {
        abs_file_path = g_strdup (abs_cfg_path);
        return abs_file_path;
    }

    delim = g_strrstr (abs_cfg_path, "/");
    *(delim + 1) = '\0';

    abs_file_path = g_strconcat (abs_cfg_path, file_path, NULL);
    g_free (file_path);

    return abs_file_path;
}

static gboolean
set_tracker_properties (GstElement *nvtracker)
{
    gboolean ret = FALSE;
    GError *error = NULL;
    gchar **keys = NULL;
    gchar **key = NULL;
    GKeyFile *key_file = g_key_file_new ();

    if (!g_key_file_load_from_file (key_file, TRACKER_CONFIG_FILE, G_KEY_FILE_NONE,
                                    &error)) {
        g_printerr ("Failed to load config file: %s\n", error->message);
        return FALSE;
    }

    keys = g_key_file_get_keys (key_file, CONFIG_GROUP_TRACKER, NULL, &error);
    CHECK_ERROR (error);

    for (key = keys; *key; key++) {
        if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_WIDTH)) {
            gint width =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GROUP_TRACKER_WIDTH, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "tracker-width", width, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_HEIGHT)) {
            gint height =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GROUP_TRACKER_HEIGHT, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "tracker-height", height, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GPU_ID)) {
            guint gpu_id =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GPU_ID, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "gpu_id", gpu_id, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_CONFIG_FILE)) {
            char* ll_config_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                                                           g_key_file_get_string (key_file,
                                                                                  CONFIG_GROUP_TRACKER,
                                                                                  CONFIG_GROUP_TRACKER_LL_CONFIG_FILE, &error));
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "ll-config-file", ll_config_file, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_LL_LIB_FILE)) {
            char* ll_lib_file = get_absolute_file_path (TRACKER_CONFIG_FILE,
                                                        g_key_file_get_string (key_file,
                                                                               CONFIG_GROUP_TRACKER,
                                                                               CONFIG_GROUP_TRACKER_LL_LIB_FILE, &error));
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "ll-lib-file", ll_lib_file, NULL);
        } else if (!g_strcmp0 (*key, CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS)) {
            gboolean enable_batch_process =
                    g_key_file_get_integer (key_file, CONFIG_GROUP_TRACKER,
                                            CONFIG_GROUP_TRACKER_ENABLE_BATCH_PROCESS, &error);
            CHECK_ERROR (error);
            g_object_set (G_OBJECT (nvtracker), "enable_batch_process",
                          enable_batch_process, NULL);
        } else {
            g_printerr ("Unknown key '%s' for group [%s]", *key,
                        CONFIG_GROUP_TRACKER);
        }
    }

    ret = TRUE;
    done:
    if (error) {
        g_error_free (error);
    }
    if (keys) {
        g_strfreev (keys);
    }
    if (!ret) {
        g_printerr ("%s failed", __func__);
    }
    return ret;
}


void *set_metadata_ptr(int faceInfo[USER_ARRAY_SIZE])
{
    gint16 *user_metadata = (gint16*)g_malloc0(USER_ARRAY_SIZE_CAP);

    for(int i = 0; i < USER_ARRAY_SIZE; i++) {
        user_metadata[i] = faceInfo[i];
    }
    return (void *)user_metadata;
}

static gpointer copy_user_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    gint16 *src_user_metadata = (gint16 *)user_meta->user_meta_data;
    gint16 *dst_user_metadata = (gint16 *)g_malloc0(USER_ARRAY_SIZE_CAP);
    memcpy(dst_user_metadata, src_user_metadata, USER_ARRAY_SIZE_CAP);
    return (gpointer)dst_user_metadata;
}

static void release_user_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *) data;
    if(user_meta->user_meta_data) {
        g_free(user_meta->user_meta_data);
        user_meta->user_meta_data = NULL;
    }
}


extern "C"
bool NvDsInferParseRetinaNet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                              std::vector<NvDsInferParseObjectInfo> &objectList)
{
    std::vector<std::vector<float>> results;
    std::vector<bbox> boxes;
    Detector rf =  Detector();

    for (int i = 0; i < 3; i++) {
        std::vector<float> outputi = std::vector<float>((float *) outputLayersInfo[i].buffer, (float *) outputLayersInfo[i].buffer + outputLayersInfo[i].inferDims.numElements);
        results.emplace_back(outputi);
    }
    rf.Detect(boxes,results);
    for (auto &i : boxes){
        NvDsInferObjectDetectionInfo object;
        object.left = i.x1 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.top = i.y1 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.height = (i.y2 - i.y1) * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.width = (i.x2 - i.x1) * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[0] = i.point[0]._x* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[2] = i.point[1]._x* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[4] = i.point[2]._x* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[6] = i.point[3]._x* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[8] = i.point[4]._x* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[1] = i.point[0]._y* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[3] = i.point[1]._y* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[5] = i.point[2]._y* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[7] = i.point[3]._y* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[9] = i.point[4]._y* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        objectList.push_back(object);
    }
    return true;
}

static GstPadProbeReturn pgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static guint use_device_mem = 0;
    static NvDsInferNetworkInfo networkInfo {PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
    NvDsInferParseDetectionParams detectionParams;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(GST_BUFFER (info->data));
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        nvds_acquire_meta_lock (batch_meta);
        frame_meta->bInferDone = TRUE;
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
                continue;
            }
            NvDsInferTensorMeta *nvDSInferTensorMeta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
            for (unsigned int i = 0; i < nvDSInferTensorMeta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &nvDSInferTensorMeta->output_layers_info[i];
                info->buffer = nvDSInferTensorMeta->out_buf_ptrs_host[i];
                if (use_device_mem && nvDSInferTensorMeta->out_buf_ptrs_dev[i]) {
                    cudaMemcpy(nvDSInferTensorMeta->out_buf_ptrs_host[i], nvDSInferTensorMeta->out_buf_ptrs_dev[i], info->inferDims.numElements, cudaMemcpyDeviceToHost);
                }
            }
            std::vector<NvDsInferLayerInfo> outputLayersInfo(nvDSInferTensorMeta->output_layers_info,nvDSInferTensorMeta->output_layers_info + nvDSInferTensorMeta->num_output_layers);
            std::vector<NvDsInferObjectDetectionInfo> objectList;
            NvDsInferParseRetinaNet(outputLayersInfo, objectList);
            for (const auto &object:objectList) {
                NvDsUserMeta *user_meta_faceInfo = NULL;
                NvDsMetaType user_meta_type = NVDS_USER_FRAME_META_EXAMPLE;
                user_meta_faceInfo = nvds_acquire_user_meta_from_pool(batch_meta);
                int faceInfo[USER_ARRAY_SIZE];
                for(int i = 0; i < 10; i++) {
                    faceInfo[i] = (int)object.landmarks[i];
                }
                faceInfo[10] = (int)object.left;
                faceInfo[11] = (int)object.top;
                faceInfo[12] = (int)object.width;
                faceInfo[13] = (int)object.height;

                user_meta_faceInfo->user_meta_data = (void *)set_metadata_ptr(faceInfo);
                user_meta_faceInfo->base_meta.meta_type = user_meta_type;
                user_meta_faceInfo->base_meta.copy_func = (NvDsMetaCopyFunc)copy_user_meta;
                user_meta_faceInfo->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;
                nvds_add_user_meta_to_frame(frame_meta, user_meta_faceInfo);
                ////////////////////////////////////////////////////////////////
                NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
                obj_meta->unique_component_id = nvDSInferTensorMeta->unique_id;
                obj_meta->confidence = 1;

                obj_meta->object_id = UNTRACKED_OBJECT_ID;

                NvOSD_RectParams &rect_params = obj_meta->rect_params;
                NvOSD_TextParams &text_params = obj_meta->text_params;
                /* Assign bounding box coordinates. */
                rect_params.left = object.left;
                rect_params.top = object.top;
                rect_params.width = object.width;
                rect_params.height = object.height;

                /* Border of width 3. */
                rect_params.border_width = 3;
                rect_params.has_bg_color = 0;
                rect_params.border_color = (NvOSD_ColorParams) {1, 0, 0, 1};

                /* display_text requires heap allocated memory. */
                /* Display text above the left top corner of the object. */
                text_params.x_offset = rect_params.left;
                text_params.y_offset = rect_params.top - 10;
                /* Set black background for the text. */
                text_params.set_bg_clr = 1;
                text_params.text_bg_clr = (NvOSD_ColorParams) {
                        0, 0, 0, 1};
                /* Font face, size and color. */
                text_params.font_params.font_name = (gchar *) "Serif";
                text_params.font_params.font_size = 11;
                text_params.font_params.font_color = (NvOSD_ColorParams) {
                        1, 1, 1, 1};
                nvds_add_obj_meta_to_frame(frame_meta, obj_meta, NULL);
            }
        }
        nvds_release_meta_lock (batch_meta);
    }
    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
}

void print(std::vector <double> const &a) {
    for(int i=0; i < a.size(); i++)
        std::cout << a.at(i) << ' ';
    printf("\n");
}
static GstPadProbeReturn sgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static guint use_device_mem = 0;
    printf("--------------------------------------------\n");
    NvDsBatchMeta *batch_meta =gst_buffer_get_nvds_batch_meta(GST_BUFFER (info->data));
    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        /* Iterate object metadata in frame */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
            /* Iterate user metadata in object to search SGIE's tensor data */
            for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
                if (user_meta->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META){
                    NvDsInferTensorMeta *meta = (NvDsInferTensorMeta *) user_meta->user_meta_data;
                    for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                        info->buffer = meta->out_buf_ptrs_host[i];
                        if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
                            cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],info->inferDims.numElements, cudaMemcpyDeviceToHost);
                            std::vector<gdouble> outputi = std::vector<gdouble>((float *) info[i].buffer, (float *) info[i].buffer + info[i].inferDims.numElements);
//add kafka
                            NvDsEventMsgMeta *msg_meta = (NvDsEventMsgMeta *) g_malloc0 (sizeof (NvDsEventMsgMeta));
                            msg_meta->trackingId = 123456789;
                            //Convert vector to string
                            std::stringstream ss;
                            for(int i=0; i < outputi.size(); i++){
                                ss << "|" << outputi[i];
                            }
                            std::cout << ss.str() << ' ';
                            // gan char* = ss (vector)
                            msg_meta->videoPath = g_strdup (ss.str().c_str());
                            print(outputi);
                            generate_event_msg_meta (msg_meta, obj_meta->class_id, obj_meta);

                            NvDsUserMeta *user_event_meta = nvds_acquire_user_meta_from_pool (batch_meta);
                            if (user_event_meta) {
                                user_event_meta->user_meta_data = (void *) msg_meta;
                                user_event_meta->base_meta.meta_type = NVDS_EVENT_MSG_META;
                                user_event_meta->base_meta.copy_func = (NvDsMetaCopyFunc) meta_copy_func;
                                user_event_meta->base_meta.release_func = (NvDsMetaReleaseFunc) meta_free_func;
                                nvds_add_user_meta_to_frame(frame_meta, user_event_meta);
                            } else {
                                g_print ("Error in attaching event meta to buffer\n");
                            }
                        }

                    }
                }
            }
        }
    }

    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
    g_print ("In cb_newpad\n");
    GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
    const GstStructure *str = gst_caps_get_structure (caps, 0);
    const gchar *name = gst_structure_get_name (str);
    GstElement *source_bin = (GstElement *) data;
    GstCapsFeatures *features = gst_caps_get_features (caps, 0);
    /* Need to check if the pad created by the decodebin is for video and not
     * audio. */
    if (!strncmp (name, "video", 5)) {
        /* Link the decodebin pad only if decodebin has picked nvidia
         * decoder plugin nvdec_*. We do this by checking if the pad caps contain
         * NVMM memory features. */
        if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
            /* Get the source bin ghost pad */
            GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
            if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
                                           decoder_src_pad)) {
                g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
            }
            gst_object_unref (bin_ghost_pad);
        } else {
            g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
        }
    }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
                       gchar * name, gpointer user_data)
{
    g_print ("Decodebin child added: %s\n", name);
    if (g_strrstr (name, "decodebin") == name) {
        g_signal_connect (G_OBJECT (object), "child-added",
                          G_CALLBACK (decodebin_child_added), user_data);
    }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
    GstElement *bin = NULL, *uri_decode_bin = NULL;
    gchar bin_name[16] = { };

    g_snprintf (bin_name, 15, "source-bin-%02d", index);
    /* Create a source GstBin to abstract this bin's content from the rest of the
     * pipeline */
    bin = gst_bin_new (bin_name);

    /* Source element for reading from the uri.
     * We will use decodebin and let it figure out the container format of the
     * stream and the codec and plug the appropriate demux and decode plugins. */
    uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

    if (!bin || !uri_decode_bin) {
        g_printerr ("One element in source bin could not be created.\n");
        return NULL;
    }

    /* We set the input uri to the source element */
    g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

    /* Connect to the "pad-added" signal of the decodebin which generates a
     * callback once a new pad for raw data has beed created by the decodebin */
    g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
                      G_CALLBACK (cb_newpad), bin);
    g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
                      G_CALLBACK (decodebin_child_added), bin);

    gst_bin_add (GST_BIN (bin), uri_decode_bin);

    /* We need to create a ghost pad for the source bin which will act as a proxy
     * for the video decoder src pad. The ghost pad will not have a target right
     * now. Once the decode bin creates the video decoder and generates the
     * cb_newpad callback, we will set the ghost pad target to the video decoder
     * src pad. */
    if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
                                                                GST_PAD_SRC))) {
        g_printerr ("Failed to add ghost pad in source bin\n");
        return NULL;
    }

    return bin;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *) data;
    switch (GST_MESSAGE_TYPE (msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_ERROR: {
            gchar *debug;
            GError *error;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n",
                       GST_OBJECT_NAME (msg->src), error->message);
            if (debug)
                g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default:
            break;
    }
    return TRUE;
}

int main(int argc, char *argv[]){
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *queue =NULL, *streammux = NULL, *sink = NULL, *pgie =NULL, *nvvidconv = NULL, *nvosd = NULL,
    *tiler =NULL, *queue2, *queue3, *queue4,*queue6, *sgie1, *nvtracker;
    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    GstPad *queue_src_pad = NULL;
    GstPad *tiler_sink_pad = NULL;

    gchar *file1[] = {"file:///home/d/Downloads/videoplayback.mp4",
                      "file:///home/d/Downloads/videoplayback.mp4",
                      "file:///home/d/Downloads/videoplayback.mp4",
                      "file:///home/d/Downloads/videoplayback.mp4",
                      "file:///home/d/Downloads/videoplayback.mp4",
                      "file:///home/d/Downloads/videoplayback.mp4",
                      "file:///home/d/Downloads/videoplayback.mp4",
                      "file:///home/d/Downloads/videoplayback.mp4"};

    guint num_sources = 2;
    //add kafka
    GstPad *osd_sink_pad = NULL;
    GstPad *tee_render_pad = NULL;
    GstPad *tee_msg_pad = NULL;
    GstPad *sink_pad = NULL;
    GstPad *src_pad = NULL;
    GOptionContext *ctx = NULL;
    GOptionGroup *group = NULL;
    GError *error = NULL;
    GstElement *msgconv = NULL, *msgbroker = NULL, *tee = NULL;
    GstElement *queueConv = NULL, *queueBroker = NULL;
    ctx = g_option_context_new ("Nvidia DeepStream Test4");
    group = g_option_group_new ("test4", NULL, NULL, NULL, NULL);
    g_option_group_add_entries (group, entries);

    g_option_context_set_main_group (ctx, group);
    g_option_context_add_group (ctx, gst_init_get_option_group ());


    if (!g_option_context_parse (ctx, &argc, &argv, &error)) {
        g_option_context_free (ctx);
        g_printerr ("%s", error->message);
        return -1;
    }
    g_option_context_free (ctx);

    loop = g_main_loop_new(NULL, FALSE);
    pipeline = gst_pipeline_new("dstensor-pipeline");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    g_object_set(G_OBJECT (streammux),"enable-padding",TRUE, "width", MUXER_OUTPUT_WIDTH, "height",MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,"batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    g_object_set(G_OBJECT (pgie), "config-file-path", "../config/pgie.txt","output-tensor-meta", TRUE, NULL);
    nvtracker = gst_element_factory_make("nvtracker", "track-object");
    /* Set necessary properties of the tracker element. */
    if (!set_tracker_properties(nvtracker)) {
        g_printerr ("Failed to set tracker properties. Exiting.\n");
        return -1;
    }
    queue = gst_element_factory_make("queue", NULL);
    queue2 = gst_element_factory_make("queue", NULL);
    queue3 = gst_element_factory_make("queue", NULL);
    queue4 = gst_element_factory_make("queue", NULL);
    queue6 = gst_element_factory_make("queue", NULL);
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    g_object_set (G_OBJECT (nvvidconv), "nvbuf-memory-type", 3, NULL);

    sgie1 = gst_element_factory_make("nvinfer", "secondary1-nvinference-engine");
    g_object_set(G_OBJECT (sgie1), "config-file-path", "../config/sgie.txt","output-tensor-meta", TRUE, "process-mode", 2, NULL);
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
    tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
    g_object_set(G_OBJECT (tiler), "rows", 1, "columns",(guint) ceil(1.0 * num_sources / 1), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,NULL);
   //add kafka
    /* Create msg converter to generate payload from buffer metadata */
    msgconv = gst_element_factory_make ("nvmsgconv", "nvmsg-converter");
    /* Create msg broker to send payload to server */
    msgbroker = gst_element_factory_make ("nvmsgbroker", "nvmsg-broker");
    /* Create tee to render buffer and send message simultaneously*/
    tee = gst_element_factory_make ("tee", "nvsink-tee");
    /* Create queues */
    queueConv = gst_element_factory_make ("queue", "nvtee-que1");
    queueBroker = gst_element_factory_make ("queue", "nvtee-que2");
    g_object_set (G_OBJECT(msgconv), "config", MSCONV_CONFIG_FILE, NULL);
    g_object_set (G_OBJECT(msgconv), "payload-type", schema_type, NULL);

    g_object_set (G_OBJECT(msgbroker), "proto-lib", proto_lib,
                  "conn-str", conn_str, "sync", FALSE, NULL);

    if (topic) {
        g_object_set (G_OBJECT(msgbroker), "topic", topic, NULL);
    }

    if (cfg_file) {
        g_object_set (G_OBJECT(msgbroker), "config", cfg_file, NULL);
    }

    g_object_set (G_OBJECT (sink), "sync", TRUE, NULL);
    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);
    gst_bin_add_many(GST_BIN (pipeline),streammux, pgie, queue,nvtracker, queue6, sgie1, queue4,tiler, queue2, nvvidconv, queue3, nvosd,tee, queueBroker, queueConv, msgconv,
                     msgbroker,sink, NULL);

    for (int i = 0; i < num_sources; i++){
            GstPad *sinkpad, *srcpad;
            gchar pad_name[16] = {};
            GstElement *source_bin = create_source_bin(i, file1[i]);

            if (!source_bin) {
                g_printerr("Failed to create source bin. Exiting.\n");
                return -1;
            }

            gst_bin_add(GST_BIN (pipeline), source_bin);

            g_snprintf(pad_name, 15, "sink_%u", i);
            sinkpad = gst_element_get_request_pad(streammux, pad_name);
            if (!sinkpad) {
                g_printerr("Streammux request sink pad failed. Exiting.\n");
                return -1;
            }

            srcpad = gst_element_get_static_pad(source_bin, "src");
            if (!srcpad) {
                g_printerr("Failed to get src pad of source bin. Exiting.\n");
                return -1;
            }

            if (gst_pad_link(srcpad, sinkpad) != GST_PAD_LINK_OK) {
                g_printerr("Failed to link source bin to stream muxer. Exiting.\n");
                return -1;
            }

            gst_object_unref(srcpad);
            gst_object_unref(sinkpad);

    }
    ///////////////////////////////////////////////////For displaying into a grid///////////////////////////////////////
    guint pgie_batch_size;
    guint tiler_rows, tiler_columns;
    g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);

    g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
                  MUXER_OUTPUT_HEIGHT,
                  "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
    tiler_rows = (guint) sqrt (num_sources);
    tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
    /* we set the tiler properties here */
    g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
                  "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT, NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    gst_element_link_many(streammux, pgie, queue,nvtracker, queue6, nvvidconv, queue3, sgie1, queue4, tiler, queue2, nvosd,tee, NULL);
    if (!gst_element_link_many (queueConv, msgconv, msgbroker, NULL)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
    }
    if (!gst_element_link (queueBroker, sink)) {
        g_printerr ("Elements could not be linked. Exiting.\n");
        return -1;
    }
    //add kafka
    sink_pad = gst_element_get_static_pad (queueConv, "sink");
    tee_msg_pad = gst_element_get_request_pad (tee, "src_%u");
    tee_render_pad = gst_element_get_request_pad (tee, "src_%u");
    if (!tee_msg_pad || !tee_render_pad) {
        g_printerr ("Unable to get request pads\n");
        return -1;
    }

    if (gst_pad_link (tee_msg_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr ("Unable to link tee and message converter\n");
        gst_object_unref (sink_pad);
        return -1;
    }

    gst_object_unref (sink_pad);

    sink_pad = gst_element_get_static_pad (queueBroker, "sink");
    if (gst_pad_link (tee_render_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr ("Unable to link tee and render\n");
        gst_object_unref (sink_pad);
        return -1;
    }

    gst_object_unref (sink_pad);
    ///////////////////////////////////// post process the output from pgie ////////////////////////////////////////////
    queue_src_pad = gst_element_get_static_pad(queue, "src");
    gst_pad_add_probe(queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_pad_buffer_probe, NULL, NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////// get output from sgie /////////////////////////////////////////////////////////
    tiler_sink_pad = gst_element_get_static_pad(tiler, "sink");
    gst_pad_add_probe(tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_pad_buffer_probe, NULL, NULL);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT (pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}
