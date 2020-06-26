#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "postProcessRetina.h"

#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define PGIE_NET_WIDTH 640
#define PGIE_NET_HEIGHT 640
#define MUXER_BATCH_TIMEOUT_USEC 40000
#define NVDS_USER_FRAME_META_EXAMPLE (nvds_get_user_meta_type("NVIDIA.NVINFER.USER_META"))
#define USER_ARRAY_SIZE 14
#define USER_ARRAY_SIZE_CAP 30

#define GST_CAPS_FEATURES_NVMM "memory:NVMM"



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
                              NvDsInferNetworkInfo  const &networkInfo,
                              NvDsInferParseDetectionParams const &detectionParams,
                              std::vector<NvDsInferParseObjectInfo> &objectList)
{
    std::vector<std::vector<float>> results;
    std::vector<FaceDetectInfo> faceInfo;
    postProcessRetina rf =  postProcessRetina((string &) "model_path", "net3");

    for (int i = 0; i < 9; i++) {
        std::vector<float> outputi = std::vector<float>((float *) outputLayersInfo[i].buffer, (float *) outputLayersInfo[i].buffer + outputLayersInfo[i].inferDims.numElements);
        results.emplace_back(outputi);
    }

    rf.detect(results, 0.5, faceInfo, PGIE_NET_WIDTH);
    for (auto &i : faceInfo){
        NvDsInferObjectDetectionInfo object;
        object.left = i.rect.x1 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.top = i.rect.y1 * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.height = (i.rect.y2 - i.rect.y1) * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.width = (i.rect.x2 - i.rect.x1) * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[0] = i.pts.x[0]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[2] = i.pts.x[1]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[4] = i.pts.x[2]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[6] = i.pts.x[3]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[8] = i.pts.x[4]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[1] = i.pts.y[0]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[3] = i.pts.y[1]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[5] = i.pts.y[2]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[7] = i.pts.y[3]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
        object.landmarks[9] = i.pts.y[4]* MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
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
            NvDsInferParseRetinaNet(outputLayersInfo, networkInfo, detectionParams, objectList);


            for (const auto &object:objectList) {
                NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(batch_meta);
                obj_meta->unique_component_id = nvDSInferTensorMeta->unique_id;
                obj_meta->confidence = 0.0;

                obj_meta->object_id = UNTRACKED_OBJECT_ID;

                NvOSD_RectParams &rect_params = obj_meta->rect_params;
                NvOSD_TextParams &text_params = obj_meta->text_params;
                //////////////////////////////////////////////////////
                NvDsUserMeta *user_meta_faceInfo = NULL;
                NvDsMetaType user_meta_type = NVDS_USER_FRAME_META_EXAMPLE;
                user_meta_faceInfo = nvds_acquire_user_meta_from_pool(batch_meta);
                int faceInfo[USER_ARRAY_SIZE];
                for(int i = 0; i < 10; i++) {
                    faceInfo[i] = (int)object.landmarks[i];
                }
                faceInfo[10] = (int)object.left;
                faceInfo[11] = (int)object.top;
                faceInfo[12] = (int)object.left + (int)object.width;
                faceInfo[13] = (int)object.top + (int)object.height;

                user_meta_faceInfo->user_meta_data = (void *)set_metadata_ptr(faceInfo);
                user_meta_faceInfo->base_meta.meta_type = user_meta_type;
                user_meta_faceInfo->base_meta.copy_func = (NvDsMetaCopyFunc)copy_user_meta;
                user_meta_faceInfo->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;
                nvds_add_user_meta_to_frame(frame_meta, user_meta_faceInfo);
                //////////////////////////////////////////////////////
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
    }
    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
}

void print(std::vector <float> const &a) {
    for(int i=0; i < a.size(); i++)
        std::cout << a.at(i) << ' ';
    printf("\n");
}
static GstPadProbeReturn sgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static guint use_device_mem = 0;
    printf("--------------------------------------------\n");
    NvDsBatchMeta *batch_meta =gst_buffer_get_nvds_batch_meta(GST_BUFFER (info->data));
    NvDsUserMeta *user_meta = NULL;
    gint16 *user_meta_data = NULL;

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
                            std::vector<float> outputi = std::vector<float>((float *) info[i].buffer, (float *) info[i].buffer + info[i].inferDims.numElements);
                            print(outputi);
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
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =NULL, *decoder = NULL, *streammux = NULL,
    *sink = NULL, *pgie =NULL, *nvvidconv = NULL, *nvosd = NULL,  *tiler =NULL, *queue2, *queue3, *queue4, *queue5,
    *dsexample, *sgie1;
    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    GstPad *queue_src_pad = NULL;
    GstPad *tiler_sink_pad = NULL;

    guint i = 0;
//    gchar *file = "rtsp://admin:abcd1234@172.16.10.84/Streaming/Channels/101";
    gchar *file = "file:///home/d/Downloads/videoplayback.mp4";
    guint num_sources = 1;
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    pipeline = gst_pipeline_new("dstensor-pipeline");
    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    g_object_set(G_OBJECT (streammux),"enable-padding",TRUE, "width", MUXER_OUTPUT_WIDTH, "height",MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,"batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);
    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    g_object_set(G_OBJECT (pgie), "config-file-path", "../models/pgie.txt","output-tensor-meta", TRUE, "batch-size", num_sources, NULL);
    queue = gst_element_factory_make("queue", NULL);
    queue2 = gst_element_factory_make("queue", NULL);
    queue3 = gst_element_factory_make("queue", NULL);
    queue4 = gst_element_factory_make("queue", NULL);
    queue5 = gst_element_factory_make("queue", NULL);
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    g_object_set (G_OBJECT (nvvidconv), "nvbuf-memory-type", 3, NULL);

    dsexample = gst_element_factory_make ("dsexample", "example-plugin");
    g_object_set(G_OBJECT (dsexample), "full-frame", FALSE, "blur-objects", TRUE, NULL);
    sgie1 = gst_element_factory_make("nvinfer", "secondary1-nvinference-engine");
    g_object_set(G_OBJECT (sgie1), "config-file-path", "../models/sgie.txt","output-tensor-meta", TRUE, "process-mode", 2, NULL);
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
    tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
    g_object_set(G_OBJECT (tiler), "rows", 1, "columns",(guint) ceil(1.0 * num_sources / 1), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,NULL);
    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);
    gst_bin_add_many(GST_BIN (pipeline),streammux, pgie, queue, dsexample, queue5, sgie1, queue4,tiler, queue2, nvvidconv, queue3, nvosd,sink, NULL);

    for (i = 0; i < num_sources; i++){
        GstPad *sinkpad, *srcpad;
        gchar pad_name[16] = { };
        GstElement *source_bin = create_source_bin (i, file);

        if (!source_bin) {
            g_printerr ("Failed to create source bin. Exiting.\n");
            return -1;
        }

        gst_bin_add (GST_BIN (pipeline), source_bin);

        g_snprintf (pad_name, 15, "sink_%u", i);
        sinkpad = gst_element_get_request_pad (streammux, pad_name);
        if (!sinkpad) {
            g_printerr ("Streammux request sink pad failed. Exiting.\n");
            return -1;
        }

        srcpad = gst_element_get_static_pad (source_bin, "src");
        if (!srcpad) {
            g_printerr ("Failed to get src pad of source bin. Exiting.\n");
            return -1;
        }

        if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
            g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
            return -1;
        }

        gst_object_unref (srcpad);
        gst_object_unref (sinkpad);
    }

    gst_element_link_many(streammux, pgie, queue, nvvidconv, queue3, dsexample, queue5, sgie1, queue4, tiler, queue2, nvosd,sink, NULL);
    queue_src_pad = gst_element_get_static_pad(queue, "src");
    gst_pad_add_probe(queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_pad_buffer_probe, NULL, NULL);
    tiler_sink_pad = gst_element_get_static_pad(tiler, "sink");
    gst_pad_add_probe(tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_pad_buffer_probe, NULL, NULL);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    g_main_loop_run(loop);

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT (pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);

    return 0;
}
