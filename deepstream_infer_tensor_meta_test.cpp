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

#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 1280
#define PGIE_NET_WIDTH 640
#define PGIE_NET_HEIGHT 640
#define MUXER_BATCH_TIMEOUT_USEC 40000


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
    printf("size %zu\n", faceInfo.size());
    for (auto &i : faceInfo){
        printf("%f\n",i.score);
        NvDsInferObjectDetectionInfo object;
        object.left = i.rect.x1;
        object.top = i.rect.y1;
        object.height = i.rect.y2 - i.rect.y1;
        object.width = i.rect.x2 - i.rect.x1;
        objectList.push_back(object);
    }
    return true;
}


static GstPadProbeReturn pgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static guint use_device_mem = 0;
    static NvDsInferNetworkInfo networkInfo {PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
    NvDsInferParseDetectionParams detectionParams;
    NvDsBatchMeta *nvDsBatchMeta = gst_buffer_get_nvds_batch_meta(GST_BUFFER (info->data));
    for (NvDsMetaList *l_frame = nvDsBatchMeta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *nvDsFrameMeta = (NvDsFrameMeta *) l_frame->data;
        for (NvDsMetaList *l_user = nvDsFrameMeta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *nvDsUserMeta = (NvDsUserMeta *) l_user->data;
            if (nvDsUserMeta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META){
                continue;
            }
            NvDsInferTensorMeta *nvDSInferTensorMeta = (NvDsInferTensorMeta *) nvDsUserMeta->user_meta_data;
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

            for (const auto &rect:objectList) {
                NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool(nvDsBatchMeta);
                obj_meta->unique_component_id = nvDSInferTensorMeta->unique_id;
                obj_meta->confidence = 0.0;

                obj_meta->object_id = UNTRACKED_OBJECT_ID;

                NvOSD_RectParams &rect_params = obj_meta->rect_params;
                NvOSD_TextParams &text_params = obj_meta->text_params;

                /* Assign bounding box coordinates. */
                rect_params.left = rect.left * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                rect_params.top = rect.top * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
                rect_params.width = rect.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                rect_params.height =rect.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;

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
                nvds_add_obj_meta_to_frame(nvDsFrameMeta, obj_meta, NULL);
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
    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;
        /* Iterate object metadata in frame */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;
            /* Iterate user metadata in object to search SGIE's tensor data */
            for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL; l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
                if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                    continue;
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

    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
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
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =NULL, *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie =NULL, *nvvidconv = NULL, *nvosd = NULL,  *tiler =NULL, *queue2, *queue3, *queue4, *sgie1;
    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    GstPad *queue_src_pad = NULL;
    GstPad *tiler_sink_pad = NULL;

    guint i = 0;
    std::string file = "/home/d/Downloads/output.h264";
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
    sgie1 = gst_element_factory_make("nvinferonnx", "secondary1-nvinference-engine");
    g_object_set(G_OBJECT (sgie1), "config-file-path", "../models/sgie.txt","output-tensor-meta", TRUE, "process-mode", 2, NULL);
    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
    tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
    g_object_set(G_OBJECT (tiler), "rows", 1, "columns",(guint) ceil(1.0 * num_sources / 1), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,NULL);
    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);
    gst_bin_add_many(GST_BIN (pipeline),streammux, pgie, queue,sgie1,queue4, tiler, queue2, nvvidconv, queue3, nvosd, sink, NULL);
    for (i = 0; i < num_sources; i++){
        source = gst_element_factory_make("filesrc", NULL);
        h264parser = gst_element_factory_make("h264parse", NULL);
        decoder = gst_element_factory_make("nvv4l2decoder", NULL);
        gst_bin_add_many(GST_BIN (pipeline), source, h264parser, decoder, NULL);
        GstPad *sinkpad, *srcpad;
        gchar pad_name_sink[16];
        sprintf(pad_name_sink, "sink_%d", i);
        gchar pad_name_src[16] = "src";
        sinkpad = gst_element_get_request_pad(streammux, pad_name_sink);
        srcpad = gst_element_get_static_pad(decoder, pad_name_src);
        gst_pad_link(srcpad, sinkpad);
        gst_object_unref(sinkpad);
        gst_object_unref(srcpad);
        gst_element_link_many(source, h264parser, decoder, NULL);
        g_object_set(G_OBJECT (source), "location", file.c_str(), NULL);
    }
    gst_element_link_many(streammux, pgie, queue, sgie1, queue4, tiler, queue2, nvvidconv, queue3, nvosd, sink, NULL);
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
