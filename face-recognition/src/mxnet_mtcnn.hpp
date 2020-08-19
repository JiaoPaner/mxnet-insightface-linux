//
// Created by jiaopan on 8/19/20.
//

#include <string>
#include <vector>

#include "mtcnn.hpp"
#include "mxnet/c_predict_api.h"
#include "comm_lib.hpp"
#include "buffer_file.hpp"

class MxNetMtcnn : public Mtcnn {

public:
    MxNetMtcnn() :rnet_batch_bound_(10000), onet_batch_bound_(10000), pnet_h(-1), pnet_w(-1) {};

    int LoadModule(const std::string& model_dir);
    int LoadModelFileToMem(const std::string& model_dir);
    void Detect(cv::Mat& img, std::vector<face_box>& face_list);
    void clearPredictVec();

    void set_batch_mode_bound(int r, int o){
        rnet_batch_bound_ = r;
        onet_batch_bound_ = o;
    }

    ~MxNetMtcnn();

protected:


    void CopyOnePatch(const cv::Mat& img, face_box&input_box, float * data_to, int width, int height);
    PredictorHandle LoadPNet(int h, int w);
    PredictorHandle LoadRNet(int batch);
    PredictorHandle LoadONet(int batch);
    PredictorHandle LoadMxNetModule(const std::string& param_file, const std::string& json_file,int batch, int channel, int input_h, int input_w);
    PredictorHandle LoadMxNetModuleByMem(const std::vector<char>& param_buffer, const std::vector<char>& json_buffer, int batch, int channel, int input_h, int input_w);

    void RunPNet(const cv::Mat& img, scale_window& win, std::vector<face_box>& box_list, PredictorHandle& pred_pNet);
    int RunPreLoadRNet(const cv::Mat& img, face_box& input_box, face_box& output_box);
    int RunPreLoadONet(const cv::Mat& img, face_box& input_box, face_box& output_box);

    void RunRNet(const cv::Mat& img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes);
    void RunONet(const cv::Mat& img, std::vector<face_box>& rnet_boxes, std::vector<face_box>& output_boxes);


private:

    std::string  model_dir_;
    PredictorHandle  RNet_;
    PredictorHandle  ONet_;
    int rnet_batch_bound_;
    int onet_batch_bound_;
    std::vector<PredictorHandle> PredictVec;
    int pnet_h;
    int pnet_w;
    std::map<std::string, std::vector<char>> param_buffer_map;
    std::map<std::string, std::vector<char>> json_buffer_map;
};
