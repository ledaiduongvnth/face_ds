#include <gst/gst.h>
#include <glib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"
#include <opencv2/objdetect/objdetect.hpp>
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#define INFER_PGIE_CONFIG_FILE  "../dstensor_pgie_config.txt"
#define INFER_SGIE1_CONFIG_FILE "../dstensor_sgie1_config.txt"
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2
#define PGIE_DETECTED_CLASS_NUM 4
#define MUXER_OUTPUT_WIDTH 1280
#define MUXER_OUTPUT_HEIGHT 720
#define PGIE_NET_WIDTH 640
#define PGIE_NET_HEIGHT 368
#define MUXER_BATCH_TIMEOUT_USEC 40000
gint frame_number = 0;
const gchar sgie1_classes_str[12][32] = {"black", "blue", "brown", "gold", "green","grey", "maroon", "orange", "red", "silver", "white", "yellow"};
const gchar sgie2_classes_str[20][32] = {"Acura", "Audi", "BMW", "Chevrolet", "Chrysler","Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia","Lexus", "Mazda", "Mercedes", "Nissan","Subaru", "Toyota", "Volkswagen"};
const gchar sgie3_classes_str[6][32] = {"coupe", "largevehicle", "sedan", "suv", "truck", "van"};
const gchar pgie_classes_str[PGIE_DETECTED_CLASS_NUM][32] = {"Vehicle", "TwoWheeler", "Person", "RoadSign"};
const guint sgie1_unique_id = 2;
const guint sgie2_unique_id = 3;
const guint sgie3_unique_id = 4;
extern "C"
bool NvDsInferParseCustomResnet(std::vector<NvDsInferLayerInfo>const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,NvDsInferParseDetectionParams const &detectionParams,std::vector<NvDsInferObjectDetectionInfo> &objectList);
static GstPadProbeReturn pgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static guint use_device_mem = 0;
    static NvDsInferNetworkInfo networkInfo {PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
    NvDsInferParseDetectionParams detectionParams;
    detectionParams.numClassesConfigured = 4;
    detectionParams.perClassPreclusterThreshold = {0.2, 0.2, 0.2, 0.2};
    static float groupThreshold = 1;
    static float groupEps = 0.2;

    NvDsBatchMeta *batch_meta =
            gst_buffer_get_nvds_batch_meta(GST_BUFFER (info->data));

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

        /* Iterate user metadata in frames to search PGIE's tensor metadata */
        for (NvDsMetaList *l_user = frame_meta->frame_user_meta_list; l_user != NULL; l_user = l_user->next) {
            NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
            if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                continue;

            /* convert to tensor metadata */
            NvDsInferTensorMeta *meta =
                    (NvDsInferTensorMeta *) user_meta->user_meta_data;
            for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                info->buffer = meta->out_buf_ptrs_host[i];
                if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
                    cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                               info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                }
            }
            /* Parse output tensor and fill detection results into objectList. */
            std::vector<NvDsInferLayerInfo>
                    outputLayersInfo(meta->output_layers_info,
                                     meta->output_layers_info + meta->num_output_layers);
            std::vector<NvDsInferObjectDetectionInfo> objectList;
            if (meta->network_info.width != networkInfo.width ||
                meta->network_info.height != networkInfo.height ||
                meta->network_info.channels != networkInfo.channels) {
                g_error ("failed to check pgie network info\n");
            }
            NvDsInferParseCustomResnet(outputLayersInfo, networkInfo, detectionParams, objectList);

            /* Seperate detection rectangles per class for grouping. */
            std::vector<std::vector<cv::Rect >> objectListClasses(PGIE_DETECTED_CLASS_NUM);
            for (auto &obj:objectList) {
                objectListClasses[obj.classId].emplace_back(obj.left, obj.top, obj.width, obj.height);
            }

            for (uint32_t c = 0; c < objectListClasses.size(); ++c) {
                auto &objlist = objectListClasses[c];
                if (objlist.empty())
                    continue;

                /* Merge and cluster similar detection results */
                cv::groupRectangles(objlist, groupThreshold, groupEps);

                /* Iterate final rectangules and attach result into frame's obj_meta_list. */
                for (const auto &rect:objlist) {
                    NvDsObjectMeta *obj_meta =
                            nvds_acquire_obj_meta_from_pool(batch_meta);
                    obj_meta->unique_component_id = meta->unique_id;
                    obj_meta->confidence = 0.0;

                    /* This is an untracked object. Set tracking_id to -1. */
                    obj_meta->object_id = UNTRACKED_OBJECT_ID;
                    obj_meta->class_id = c;

                    NvOSD_RectParams &rect_params = obj_meta->rect_params;
                    NvOSD_TextParams &text_params = obj_meta->text_params;

                    /* Assign bounding box coordinates. */
                    rect_params.left = rect.x * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                    rect_params.top = rect.y * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
                    rect_params.width = rect.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
                    rect_params.height =
                            rect.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;

                    /* Border of width 3. */
                    rect_params.border_width = 3;
                    rect_params.has_bg_color = 0;
                    rect_params.border_color = (NvOSD_ColorParams) {
                            1, 0, 0, 1};

                    /* display_text requires heap allocated memory. */
                    text_params.display_text = g_strdup(pgie_classes_str[c]);
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
    }
    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn sgie_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    static guint use_device_mem = 0;

    NvDsBatchMeta *batch_meta =
            gst_buffer_get_nvds_batch_meta(GST_BUFFER (info->data));

    /* Iterate each frame metadata in batch */
    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

        /* Iterate object metadata in frame */
        for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL;
             l_obj = l_obj->next) {
            NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

            /* Iterate user metadata in object to search SGIE's tensor data */
            for (NvDsMetaList *l_user = obj_meta->obj_user_meta_list; l_user != NULL;
                 l_user = l_user->next) {
                NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
                if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
                    continue;

                /* convert to tensor metadata */
                NvDsInferTensorMeta *meta =
                        (NvDsInferTensorMeta *) user_meta->user_meta_data;

                for (unsigned int i = 0; i < meta->num_output_layers; i++) {
                    NvDsInferLayerInfo *info = &meta->output_layers_info[i];
                    info->buffer = meta->out_buf_ptrs_host[i];
                    if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
                        cudaMemcpy(meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                                   info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
                    }
                }

                NvDsInferDimsCHW dims;

                getDimsCHWFromDims (dims, meta->output_layers_info[0].inferDims);
                unsigned int numClasses = dims.c;
                float *outputCoverageBuffer =
                        (float *) meta->output_layers_info[0].buffer;
                float maxProbability = 0;
                bool attrFound = false;
                NvDsInferAttribute attr;

                /* Iterate through all the probabilities that the object belongs to
                 * each class. Find the maximum probability and the corresponding class
                 * which meets the minimum threshold. */
                for (unsigned int c = 0; c < numClasses; c++) {
                    float probability = outputCoverageBuffer[c];
                    if (probability > 0.51 && probability > maxProbability) {
                        maxProbability = probability;
                        attrFound = true;
                        attr.attributeIndex = 0;
                        attr.attributeValue = c;
                        attr.attributeConfidence = probability;
                    }
                }

                /* Generate classifer metadata and attach to obj_meta */
                if (attrFound) {
                    NvDsClassifierMeta *classifier_meta =
                            nvds_acquire_classifier_meta_from_pool(batch_meta);

                    classifier_meta->unique_component_id = meta->unique_id;

                    NvDsLabelInfo *label_info =
                            nvds_acquire_label_info_meta_from_pool(batch_meta);
                    label_info->result_class_id = attr.attributeValue;
                    label_info->result_prob = attr.attributeConfidence;

                    /* Fill label name */
                    switch (meta->unique_id) {
                        case sgie1_unique_id:
                            strcpy(label_info->result_label,
                                   sgie1_classes_str[label_info->result_class_id]);
                            break;
                        case sgie2_unique_id:
                            strcpy(label_info->result_label,
                                   sgie2_classes_str[label_info->result_class_id]);
                            break;
                        case sgie3_unique_id:
                            strcpy(label_info->result_label,
                                   sgie3_classes_str[label_info->result_class_id]);
                            break;
                        default:
                            break;
                    }
                    gchar *temp = obj_meta->text_params.display_text;
                    obj_meta->text_params.display_text =
                            g_strconcat(temp, " ", label_info->result_label, nullptr);
                    g_free(temp);

                    nvds_add_label_info_meta_to_classifier(classifier_meta, label_info);
                    nvds_add_classifier_meta_to_object(obj_meta, classifier_meta);
                }
            }
        }
    }

    use_device_mem = 1 - use_device_mem;
    return GST_PAD_PROBE_OK;
}
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0;
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList *l_frame = NULL;
    NvDsMetaList *l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
         l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = (gchar *) g_malloc0(64);
        offset =
                snprintf(txt_params->display_text, 64, "Person = %d ",
                         person_count);
        offset =
                snprintf(txt_params->display_text + offset, 64,
                         "Vehicle = %d ", vehicle_count);

        /* Now set the offsets where the string should appear */
        txt_params->x_offset = 10;
        txt_params->y_offset = 12;

        /* Font , font-color and font-size */
        txt_params->font_params.font_name = (gchar *) "Serif";
        txt_params->font_params.font_size = 10;
        txt_params->font_params.font_color.red = 1.0;
        txt_params->font_params.font_color.green = 1.0;
        txt_params->font_params.font_color.blue = 1.0;
        txt_params->font_params.font_color.alpha = 1.0;

        /* Text background color */
        txt_params->set_bg_clr = 1;
        txt_params->text_bg_clr.red = 0.0;
        txt_params->text_bg_clr.green = 0.0;
        txt_params->text_bg_clr.blue = 0.0;
        txt_params->text_bg_clr.alpha = 1.0;

        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }


    g_print("Frame Number = %d Number of objects = %d "
            "Vehicle Count = %d Person Count = %d\n",
            frame_number, num_rects, vehicle_count, person_count);
    frame_number++;
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

int main(int argc, char *argv[]) {
    GMainLoop *loop = NULL;
    GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL, *queue =NULL, *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie =NULL, *nvvidconv = NULL, *nvosd = NULL, *sgie1 = NULL, *tiler =NULL, *queue2, *queue3, *queue4;
    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    GstPad *osd_sink_pad = NULL, *queue_src_pad = NULL, *tiler_sink_pad = NULL;
    guint i = 0;
    std::string file = "/home/d/Downloads/deepstream_sdk_v5.0.0_x86_64/opt/nvidia/deepstream/deepstream-5.0/samples/streams/sample_720p.huuu";
    guint num_sources = 1;
    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    pipeline = gst_pipeline_new("dstensor-pipeline");

    streammux = gst_element_factory_make("nvstreammux", "stream-muxer");
    g_object_set(G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",MUXER_OUTPUT_HEIGHT, "batch-size", num_sources,"batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

    pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
    g_object_set(G_OBJECT (pgie), "config-file-path", INFER_PGIE_CONFIG_FILE,"output-tensor-meta", TRUE, "batch-size", num_sources, NULL);

    queue = gst_element_factory_make("queue", NULL);
    queue2 = gst_element_factory_make("queue", NULL);
    queue3 = gst_element_factory_make("queue", NULL);
    queue4 = gst_element_factory_make("queue", NULL);

    sgie1 = gst_element_factory_make("nvinfer", "secondary1-nvinference-engine");
    g_object_set(G_OBJECT (sgie1), "config-file-path", INFER_SGIE1_CONFIG_FILE,"output-tensor-meta", TRUE, "process-mode", 2, NULL);


    nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
    sink = gst_element_factory_make("nveglglessink", "nvvideo-renderer");
    tiler = gst_element_factory_make("nvmultistreamtiler", "tiler");
    g_object_set(G_OBJECT (tiler), "rows", 1, "columns",(guint) ceil(1.0 * num_sources / 1), "width", 1920, "height", 1080,NULL);

    bus = gst_pipeline_get_bus(GST_PIPELINE (pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);
    gst_bin_add_many(GST_BIN (pipeline),streammux, pgie, queue, sgie1, queue2,tiler, queue3, nvvidconv, queue4, nvosd, sink, NULL);
    for (i = 0; i < num_sources; i++) {
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
    gst_element_link_many(streammux, pgie, queue, sgie1, queue2, tiler, queue3, nvvidconv, queue4, nvosd, sink, NULL);
    queue_src_pad = gst_element_get_static_pad(queue, "src");
    gst_pad_add_probe(queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER, pgie_pad_buffer_probe, NULL, NULL);
    tiler_sink_pad = gst_element_get_static_pad(tiler, "sink");
    gst_pad_add_probe(tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, sgie_pad_buffer_probe, NULL, NULL);
    osd_sink_pad = gst_element_get_static_pad(nvosd, "sink");
    gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, NULL, NULL);
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT (pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}
