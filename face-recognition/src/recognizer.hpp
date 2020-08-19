//
// Created by jiaopan on 8/19/20.
//

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "mxnet/c_predict_api.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "mxnet_mtcnn.hpp"
#include "face_align.hpp"

using namespace mxnet::cpp;
Context global_ctx(kCPU,0);
class Recognizer{
    public:
        int LoadMxnetModel(const std::string & fname, std::vector<char>& buf){
            std::ifstream fs(fname, std::ios::binary | std::ios::in);
            if (!fs.good()){
                std::cerr << fname << " does not exist" << std::endl;
                return -1;
            }
            fs.seekg(0, std::ios::end);
            int fsize = fs.tellg();
            fs.seekg(0, std::ios::beg);
            buf.resize(fsize);
            fs.read(buf.data(), fsize);
            fs.close();
            return 1;
        }
        int LoadExtractModule(const std::string& param_file, const std::string& json_file,int batch, int channel, int input_h, int input_w){
            std::vector<char> param_buffer;
            std::vector<char> json_buffer;
            if (LoadMxnetModel(param_file, param_buffer) == -1)
                return -1;
            if (LoadMxnetModel(json_file, json_buffer) == -1)
                return -1;
            int device_type = 1;
            int dev_id = 0;
            mx_uint  num_input_nodes = 1;
            const char * input_keys[1];
            const mx_uint input_shape_indptr[] = { 0, 4 };
            const mx_uint input_shape_data[] = {static_cast<mx_uint>(batch),static_cast<mx_uint>(channel),static_cast<mx_uint>(input_h),static_cast<mx_uint>(input_w)};
            input_keys[0] = "data";
            int ret = MXPredCreate(json_buffer.data(),param_buffer.data(),param_buffer.size(),device_type,dev_id,num_input_nodes,input_keys,input_shape_indptr,input_shape_data,&pred_feature);
            return ret == 0 ? 1 : -1;
        }
        int loadModel(std::string mtcnn_model, std::string params, std::string json) {
            int mtcnn_status = mtcnn.LoadModule(mtcnn_model);
            int extract_status = LoadExtractModule(params,json,1, 3, 112, 112);
            if (mtcnn_status == 1 && extract_status == 1)
                return 1;
            return -1;
        }
        std::vector<cv::Mat> createAlignFace(cv::Mat& img,int type) {
            std::vector<cv::Mat> aligned_faces;
            if (img.empty())
                return aligned_faces;
            cv::Mat src(5, 2, CV_32FC1, norm_face);
            std::vector<face_box> face_boxs;
            mtcnn.Detect(img, face_boxs);
            int index = 0;
            float max_box = 0;

            if (type == 1) {
                if (face_boxs.size() > 0) {
                    for (int i = 0; i < face_boxs.size(); i++) {
                        face_box face_box = face_boxs[i];
                        float box = (face_box.x1 - face_box.x0 + face_box.y1 - face_box.y0) / 2;
                        if (box > max_box) {
                            max_box = box;
                            index = i;
                        }
                    }
                    face_box face_box = face_boxs[index];
                    //cv::rectangle(img, cv::Point(face_box.x0, face_box.y0), cv::Point(face_box.x1, face_box.y1), cv::Scalar(0, 255, 0), 2);
                    //cv::imshow("img",img);
                    //cv::waitKey(2000);

                    float landmark[5][2] = {
                            { face_box.landmark.x[0] , face_box.landmark.y[0] },
                            { face_box.landmark.x[1] , face_box.landmark.y[1] },
                            { face_box.landmark.x[2] , face_box.landmark.y[2] },
                            { face_box.landmark.x[3] , face_box.landmark.y[3] },
                            { face_box.landmark.x[4] , face_box.landmark.y[4] }
                    };

                    cv::Mat dst(5, 2, CV_32FC1, landmark);
                    cv::Mat m = similarTransform(dst, src);
                    cv::Mat aligned(112, 112, CV_32FC3);
                    cv::Size size(112, 112);
                    cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
                    cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
                    aligned_faces.push_back(aligned);
                }
            }
            else{
                for (int i = 0; i < face_boxs.size(); i++) {
                    face_box face_box = face_boxs[i];
                    if (face_box.score * 100 < 70)
                        continue;
                    float landmark[5][2] = {
                            { face_box.landmark.x[0] , face_box.landmark.y[0] },
                            { face_box.landmark.x[1] , face_box.landmark.y[1] },
                            { face_box.landmark.x[2] , face_box.landmark.y[2] },
                            { face_box.landmark.x[3] , face_box.landmark.y[3] },
                            { face_box.landmark.x[4] , face_box.landmark.y[4] }
                    };
                    cv::Mat dst(5, 2, CV_32FC1, landmark);
                    cv::Mat m = similarTransform(dst, src);
                    cv::Mat aligned(112, 112, CV_32FC3);
                    cv::Size size(112, 112);
                    cv::Mat transfer = m(cv::Rect(0, 0, 3, 2));
                    cv::warpAffine(img, aligned, transfer, size, 1, 0, 0);
                    aligned_faces.push_back(aligned);
                }
            }
            return aligned_faces;
        }
        cv::Mat extractFeature(const cv::Mat& face){
            int width = face.cols;
            int height = face.rows;
            cv::Mat img_rgb(height, width, CV_32FC3);
            face.convertTo(img_rgb, CV_32FC3);
            cv::cvtColor(img_rgb, img_rgb, cv::COLOR_BGR2RGB);

            std::vector<float> input(3 * height * width);
            std::vector<cv::Mat> input_channels;
            set_input_buffer(input_channels, input.data(), height, width);
            cv::split(img_rgb, input_channels);

            MXPredSetInput(pred_feature, "data", input.data(), input.size());
            MXPredForward(pred_feature);
            mx_uint *shape = NULL;
            mx_uint shape_len = 0;
            MXPredGetOutputShape(pred_feature, 0, &shape, &shape_len);

            int feature_size = 1;
            for (unsigned int i = 0; i<shape_len; i++)
                feature_size *= shape[i];
            std::vector<float> feature(feature_size);

            MXPredGetOutput(pred_feature, 0, feature.data(), feature_size);

            cv::Mat output = cv::Mat(feature, true).reshape(1, 1);
            cv::normalize(output, output);

            return output;
        }
        double distance(const cv::Mat& base, const  cv::Mat& target) {
            cv::Mat broad;
            broad = target - base;
            cv::pow(broad, 2, broad);
            cv::reduce(broad, broad, 1, cv::REDUCE_SUM);

            double dis;
            cv::Point point;
            cv::minMaxLoc(broad, &dis, 0, &point, 0);
            return dis;
        }
        ~Recognizer() {
            if (pred_feature)
                MXPredFree(pred_feature);
        }
    private:
        PredictorHandle pred_feature;
        MxNetMtcnn mtcnn;
};