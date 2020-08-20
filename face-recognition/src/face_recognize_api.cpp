//
// Created by jiaopan on 8/19/20.
//

#include "face_recognize_api.h"
#include <opencv2/opencv.hpp>
#include "recognizer.hpp"
#include "cJSON.h"
#include "utils.hpp"
#include "api_usages.hpp"
#include <time.h>

static Recognizer recognizer;
char* extractFaceFeatureByImage(cv::Mat image,int type) {
    cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
    char *resultJson;
    try {
        std::vector<cv::Mat>  aligned_faces = recognizer.createAlignFace(image,type);
        if (aligned_faces.size() == 0) {
            cJSON_AddNumberToObject(result, "status", -1);
            cJSON_AddStringToObject(result, "msg", "there is no face");
            cJSON_AddItemToObject(result, "embeddings", embeddings);
            resultJson = cJSON_PrintUnformatted(result);
            return resultJson;
        }
        for (int i = 0; i < aligned_faces.size(); i++) {
            cv::Mat features = recognizer.extractFeature(aligned_faces[i]);
            std::vector<double> vector = (std::vector<double>)features;

            std::stringstream ss;
            ss << std::setprecision(16);
            std::copy(vector.begin(), vector.end(), std::ostream_iterator<double>(ss, ","));
            std::string values = ss.str();
            values.pop_back();

            cJSON  *embedding;
            cJSON_AddItemToArray(embeddings, embedding = cJSON_CreateObject());
            cJSON_AddStringToObject(embedding, "embedding", values.c_str());
        }
        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "register success");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "register failed");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
}

char* extractFaceFeature(cv::Mat &face) {
    resize(face, face, cv::Size(112, 112));
    cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
    char *resultJson;
    try {
        if (face.empty()) {
            cJSON_AddNumberToObject(result, "status", -1);
            cJSON_AddStringToObject(result, "msg", "register failed,there is no face");
            cJSON_AddItemToObject(result, "embeddings", embeddings);
            resultJson = cJSON_PrintUnformatted(result);
            return resultJson;
        }

        cv::Mat features = recognizer.extractFeature(face);
        std::vector<double> vector = (std::vector<double>)features;

        std::stringstream ss;
        ss << std::setprecision(16);
        std::copy(vector.begin(), vector.end(), std::ostream_iterator<double>(ss, ","));
        std::string values = ss.str();
        values.pop_back();

        cJSON  *embedding;
        cJSON_AddItemToArray(embeddings, embedding = cJSON_CreateObject());
        cJSON_AddStringToObject(embedding, "embedding", values.c_str());

        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "register success");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "register failed");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
}
char* computeDistanceByMat(cv::Mat& base, cv::Mat& target,int detected) {
    cJSON  *result = cJSON_CreateObject();
    char *resultJson;
    double distance,sim = 0;
    Mat base_emb, target_emb;
    try {

        if (detected == 1) { //face alinged to do
            Mat baseAlign, targetAlign;
            resize(base, baseAlign, cv::Size(112, 112));
            resize(target, targetAlign, cv::Size(112, 112));
            base_emb = recognizer.extractFeature(baseAlign);
            target_emb = recognizer.extractFeature(targetAlign);
            distance = recognizer.distance(base_emb, target_emb);
        }
        else {
            std::vector<cv::Mat> base_vector = recognizer.createAlignFace(base,1);
            std::vector<cv::Mat> target_vector = recognizer.createAlignFace(target,1);

            if ((base_vector.empty() || target_vector.empty())) {
                cJSON_AddNumberToObject(result, "status", -1);
                cJSON_AddStringToObject(result, "msg", "compute failed,one of images has no face");
                cJSON_AddNumberToObject(result, "distance", -1);
                cJSON_AddNumberToObject(result, "sim", 0);
                resultJson = cJSON_PrintUnformatted(result);
                return resultJson;
            }
            base_emb = recognizer.extractFeature(base_vector[0]);
            target_emb = recognizer.extractFeature(target_vector[0]);
            distance = recognizer.distance(base_emb, target_emb);
        }
        cv::transpose(target_emb, target_emb);
        sim = base_emb.dot(target_emb);
        if (sim < 0)
            sim = 0;
        if (sim > 100)
            sim = 100;

        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "compute success");
        cJSON_AddNumberToObject(result, "distance",distance);
        cJSON_AddNumberToObject(result, "sim", sim * 100);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "compute failed");
        cJSON_AddNumberToObject(result, "distance", -1);
        cJSON_AddNumberToObject(result, "sim", 0);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
}

cv::Mat convertToMat(std::string str) {
    std::vector<double> v;
    std::stringstream ss(str);
    ss << std::setprecision(16);
    std::string token;
    while (std::getline(ss, token, ',')) {
        v.push_back(std::stod(token));
    }
    cv::Mat output = cv::Mat(v, true).reshape(1, 1);
    return output;
}

/*---------------------------------------api list -----------------------------------------------------------------*/
int loadModel(char* mtcnn_model,char* insightface_params,char * insightface_json) {
    std::cout << "loading model..." << std::endl;
    return recognizer.loadModel(mtcnn_model, insightface_params, insightface_json);
}

char * extractFaceFeatureByFile(char * src, int detected = 0, int type = 0){
    cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
    char *resultJson;
    cv::Mat image;
    try{
        image = imread(src);
    }
    catch (const std::exception&){
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "register failed,can not load file");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }


    if (detected == 1) {
        return extractFaceFeature(image); // todo face aligned
    }
    else{
        return extractFaceFeatureByImage(image,type);
    }
}

char * extractFaceFeatureByByte(unsigned char * src, int width, int height, int channels, int detected = 0,int type = 0){
    int format;
    switch (channels) {
        case 1:
            format = CV_8UC1;
            break;
        case 2:
            format = CV_8UC2;
            break;
        case 3:
            format = CV_8UC3;
            break;
        default:
            format = CV_8UC4;
            break;
    }
    cv::Mat image(height, width, format, src);
    if (detected == 1) {
        return extractFaceFeature(image);// todo face aligned
    }
    else {
        return extractFaceFeatureByImage(image,type);
    }
}

char*  extractFaceFeatureByBase64(char* base64_data, int detected = 0,int type = 0) {
    std::string data(base64_data);
    cJSON  *result = cJSON_CreateObject(), *embeddings = cJSON_CreateArray();
    char *resultJson;
    cv::Mat image;
    try {
        image = Utils::base64ToMat(data);
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "register failed,can not convert base64 to Mat");
        cJSON_AddItemToObject(result, "embeddings", embeddings);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    if (detected == 1) {
        return extractFaceFeature(image);// todo face aligned
    }
    else {
        return extractFaceFeatureByImage(image,type);
    }
}


char * computeDistance(char * base_emb, char * target_emb){
    cJSON  *result = cJSON_CreateObject();
    char *resultJson;
    try{
        std::string base(base_emb), target(target_emb);
        cv::Mat baseMat = convertToMat(base), targetMat = convertToMat(target);
        double distance = recognizer.distance(baseMat, targetMat);
        cv::transpose(targetMat, targetMat);
        double sim = baseMat.dot(targetMat);
        if (sim < 0)
            sim = 0;
        if (sim > 100)
            sim = 100;
        cJSON_AddNumberToObject(result, "status", 1);
        cJSON_AddStringToObject(result, "msg", "compute success");
        cJSON_AddNumberToObject(result, "distance", distance);
        cJSON_AddNumberToObject(result, "sim", sim * 100);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    catch (const std::exception&){
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "compute failed");
        cJSON_AddNumberToObject(result, "distance", -1);
        cJSON_AddNumberToObject(result, "sim", 0);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }

}
char * computeDistanceByFile(char * base_src, char * target_src, int detected = 0){
    cJSON  *result = cJSON_CreateObject();
    char *resultJson;
    cv::Mat base,target;
    try{
        base = imread(base_src);
        target = imread(target_src);
    }
    catch (const std::exception&){
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "can not load file");
        cJSON_AddNumberToObject(result, "distance", -1);
        cJSON_AddNumberToObject(result, "sim", 0);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    return computeDistanceByMat(base, target, detected);
}
char*  computeDistanceByBase64(char* base_data,char* target_data, int detected = 0) {
    std::string base_str(base_data);
    std::string target_str(target_data);
    cJSON  *result = cJSON_CreateObject();
    char *resultJson;
    cv::Mat base, target;
    try {
        base = Utils::base64ToMat(base_str);
        target = Utils::base64ToMat(target_str);
    }
    catch (const std::exception&) {
        cJSON_AddNumberToObject(result, "status", -1);
        cJSON_AddStringToObject(result, "msg", "can not convert base64");
        cJSON_AddNumberToObject(result, "distance", -1);
        cJSON_AddNumberToObject(result, "sim", 0);
        resultJson = cJSON_PrintUnformatted(result);
        return resultJson;
    }
    return computeDistanceByMat(base, target, detected);
}

void getUsages(){
    getPythonUsage();
}

/*----------------------------------------------------------------------------------------------*/

int main() {
    char* feature_model_params = "/home/jiaopan/projects/c++/mxnet-insightface-linux/face-recognition/model/feature_model/128/model-0000.params";
    char* feature_model_json = "/home/jiaopan/projects/c++/mxnet-insightface-linux/face-recognition/model/feature_model/128/model-symbol.json";
    int status = loadModel("/home/jiaopan/projects/c++/mxnet-insightface-linux/face-recognition/model/mtcnn", feature_model_params, feature_model_json);

    char* result = extractFaceFeatureByFile("/home/jiaopan/Downloads/dit.jpg",1,1);
    std::cout << result << std::endl;

    //char* result = computeDistance("0.029020833, -0.0068783676, -0.020256473, 0.08922711, 0.1520647, -0.063353509, -0.011182057, 0.23773466, -0.1844321, -0.0074027572, -0.14794661, 0.0051653897, 0.052046064, 0.056903902, 0.079200841, -0.0086160116, -0.10275403, 0.059177171, -0.075488754, -0.049072653, 0.027570521, 0.18733168, -0.010161424, -0.076425344, 0.02510317, -0.086876422, -0.099469766, 0.16163287, -0.084967688, -0.021871576, 0.14369343, 0.097055502, 0.0065840217, -0.018097101, -0.029480744, -0.066096894, 0.016942978, 0.15646647, 0.052293401, 0.12903209, -0.10855469, 0.044652212, 0.00074599194, -0.16131599, -0.072199926, -0.093255699, -0.068642981, 0.12120935, 0.006100629, -0.038449984, 0.073843718, -0.032517787, 0.031847525, -0.0082237236, -0.033020604, -0.026024288, 0.024662545, 0.097049288, -0.013186242, 0.0087926928, 0.085941195, -0.073864095, -0.034101862, -0.062069096, 0.059358981, 0.04966893, -0.036914833, 0.047939252, 0.054796625, -0.018790253, 0.060238205, 0.0076167355, -0.015216754, -0.061193034, -0.016889416, -0.03072207, -0.16774717, 0.068628848, -0.20049851, 0.020299155, 0.1187629, -0.0033529375, -0.030330595, -0.095323272, -0.0049259844, -0.076598287, -0.17594662, -0.073459841, -0.18311255, -0.031051619, 0.03720101, -0.23655726, -0.055039775, 0.012025496, -0.010668803, -0.054880727, -0.0031114377, 0.094407134, -0.03481745, 0.061542794, 0.13756463, 0.14968817, 0.0043450315, 0.042938448, -0.0020956821, 0.10656384, 0.012789681, -0.048712991, 0.0069972454, -0.027091177, -0.063452356, -0.052107193, 0.13972144, 0.0080326824, -0.21505292, -0.043706249, -0.16731535, -0.083271667, -0.055855714, 0.051882118, -0.040529616, 0.091872461, -0.19573945, 0.02434974, 0.037454743, -0.030554138, 0.01387815, 0.18017061","0.09884572, -0.068491496, -0.078664772, 0.032762606, 0.20072919, 0.024014864, 0.047349561, 0.20139465, -0.24945836, 0.016614795, -0.07572569, 0.11808685, 0.17146656, 0.089800715, -0.025433548, 0.040357802, -0.075069338, 0.044914488, -0.093217447, -0.005331621, 0.022735745, -0.058652312, 0.026685696, -0.022456085, 0.02619436, -0.010089835, -0.14127219, 0.14149182, 0.096490413, -0.01297182, 0.20552859, 0.1833231, 0.063791238, -0.054114789, -0.056761984, -0.088626556, -0.057970654, 0.11274455, -0.089882031, 0.036606628, -0.11511264, 0.09308967, 0.059945799, -0.10937923, 0.11097006, -0.019142125, -0.09145803, 0.1311432, 0.1252171, -0.038730226, -0.080530547, 0.081010491, 0.096036114, -0.096100479, 0.049759038, 0.031370837, -0.049491502, 0.13946846, -0.11055151, -0.0038676588, -0.056756437, -0.0038580534, -0.12307825, -0.10289554, -0.0033650277, 0.083691142, 0.028183304, 0.061029408, -0.0049481452, -0.043966934, 0.014331327, 0.03241935, -0.040123675, -0.14365859, -0.014310184, -0.074716471, -0.11835559, 0.082309127, -0.17734039, -0.045208976, 0.10009804, -0.041692022, -0.0055328049, -0.033204131, -0.026514528, -0.10978604, -0.095468342, -0.12418511, -0.15704995, 0.028290421, 0.099584438, -0.19754614, -0.070700161, 0.018390089, -0.062242635, 0.02848771, 0.019649331, 0.043790292, 0.060032263, -0.033581559, 0.10508171, 0.14813991, -0.042962868, -0.026277989, 0.026779169, 0.10178101, 0.037330393, -0.084159054, 0.031589653, -0.00096823327, -0.10374437, -0.052678231, 0.06412217, -0.034865569, -0.040277272, -0.06836205, 0.00085853663, -0.058881979, -0.019776454, 0.036946878, -0.045691911, 0.088631786, -0.11997437, 0.14228344, 0.12539271, -0.0025624367, 0.048964616, 0.05127272");
    //std::cout << result << std::endl;

    //char* result = computeDistanceByFile(base_src,target_src,1);
    //std::cout << result << std::endl;

    /*char* features = extractFaceFeatureByFile(base_src);
    std::cout << "f:" << features << std::endl;
    char* temp = extractFaceFeatureByFile(base_src,1);
    std::cout << "tmp:" << temp << std::endl;*/
    //char* features = extractFaceFeatureByFile(base_src);
    //std::cout << "file:" <<features << std::endl;
    /*
    Mat base = imread(base_src);
    std::string base64_data = Mat2Base64(base,"jpg");
    std::cout << "base64:" << base64_data << std::endl;
    Mat image = Base2Mat(base64_data);
    imshow("img",image);
    */

    /*
    Mat image = imread(base_src);
    char* features = extractFaceFeatureByFile(base_src);
    std::cout << "file:" <<features << std::endl;
    unsigned char* bytes;
    Utils::matToBytes(image, bytes);
    char* result = extractFaceFeatureByByte(bytes, image.cols, image.rows, 3);
    std::cout << "bytes:" << result << std::endl;
    std::fstream fs("test1.txt"); // 创建个文件流对象,并打开"file.txt"
    std::stringstream ss; // 创建字符串流对象
    ss << fs.rdbuf(); // 把文件流中的字符输入到字符串流中
    std::string str = ss.str(); // 获取流中的字符串
    fs.close();
    //char* Base64_features = extractFaceFeatureByBase64(str.c_str());
    //std::cout << "Base64_features:" << Base64_features << std::endl;
    */

    //clock_t start, ends;
    //start = clock();
    //char* result = extractFaceFeatureByFile(base_src,1);
    //char* result = computeDistanceByFile(base_src,target_src,1);
    //char * result = computeDistance("0.001904582255519927,-0.03358444571495056,0.07209812104701996,-0.07453728467226028,-0.09480775147676468,0.1412143707275391,-0.01371232327073812,0.1258006244897842,-0.0355433002114296,-0.1849799156188965,-0.1330768465995789,0.06725277751684189,0.002782873343676329,-0.1945522278547287,-0.1266706734895706,-0.0400518998503685,0.08389019221067429,-0.06386342644691467,-0.05708419904112816,-0.0190862026065588,-0.01694397442042828,-0.04052738845348358,0.1785626709461212,-0.05148022994399071,0.003840577322989702,0.004819559399038553,0.01944071613252163,0.02982298098504543,-0.1976806223392487,-0.08488152921199799,-0.0509159155189991,-0.08664298802614212,-0.0415748618543148,0.1495159268379211,0.0342765711247921,0.1110065504908562,0.0835961252450943,0.2231821715831757,0.06444680690765381,-0.004412301816046238,0.03264988213777542,-0.1913386881351471,-0.1344100683927536,-0.1806252151727676,0.02651478722691536,-0.1609379202127457,0.009578916244208813,0.116324171423912,-0.0928058996796608,-0.01579641923308372,0.01052638702094555,0.03573188930749893,0.0004111903836019337,-0.1166312173008919,-0.02100192010402679,-0.05954171717166901,0.0371582992374897,0.1102246418595314,0.00543747004121542,-0.08432062715291977,0.09994629770517349,0.04414885863661766,0.07149370759725571,-0.03217600658535957,0.01987116038799286,-0.01704858243465424,-0.03104584850370884,-0.05346007272601128,-0.1021623760461807,0.1012672111392021,-0.001778970356099308,-0.1595409661531448,0.00850287452340126,-0.006829991936683655,0.02489127591252327,-0.08353033661842346,-0.007338901981711388,0.05411912873387337,0.01147557888180017,0.01281584613025188,-0.1293287426233292,0.03651829063892365,0.08239502459764481,0.2198153883218765,-0.01740619912743568,0.1443982720375061,0.1397885829210281,0.1250724047422409,0.002506471471861005,-0.09321369230747223,-0.1344662606716156,0.07099231332540512,-0.1931862086057663,-0.1128925308585167,0.09496592730283737,-0.06092626973986626,0.05499249324202538,0.05576762557029724,-0.01085100974887609,0.07360327988862991,0.008220355957746506,0.08192269504070282,-0.01600385643541813,0.004211107268929482,0.06608087569475174,-0.01166537776589394,0.04429132863879204,0.07068134844303131,0.05547104403376579,-0.07967103272676468,-0.004074936732649803,0.02027429640293121,0.1132912188768387,-0.04848542436957359,-0.1109229996800423,0.001467815367504954,0.02389026060700417,-0.1408744901418686,0.1913734078407288,-0.01208099257200956,-0.1045026853680611,-0.01885988377034664,0.0231326874345541,-0.008011172525584698,-0.03078076243400574,-0.04710938781499863,-0.03079074621200562,-0.02908881939947605","-0.01734724082052708,-0.08714558929204941,0.06462283432483673,0.0497586764395237,0.05397667363286018,0.05008497089147568,-0.07444144040346146,0.04769236966967583,-0.05673764646053314,0.02155164629220963,-0.001271282788366079,-0.02401094511151314,-0.1042786240577698,-0.1305110901594162,0.03273851796984673,-0.01292992942035198,-0.01142883207648993,-0.07721524685621262,-0.05343680083751678,-0.06011213734745979,-0.04153147339820862,-0.008904881775379181,0.0581544004380703,0.02912480942904949,0.02510336227715015,0.03863447159528732,0.172502413392067,0.02068039029836655,-0.1480441838502884,-0.1015273779630661,0.0963934063911438,0.005043211858719587,-0.06674067676067352,0.05199049413204193,-0.03418152406811714,0.05651063099503517,-0.103370688855648,0.152672216296196,0.1128728315234184,-0.04879985749721527,-0.06635334342718124,-0.2057603150606155,-0.007018210366368294,-0.08640377968549728,0.06130568310618401,-0.09834355860948563,0.09432881325483322,0.200285941362381,-0.09588516503572464,-0.0800967738032341,-0.1028441414237022,0.03686016798019409,-0.03829369693994522,0.01790497452020645,-0.04969615116715431,-0.01677650399506092,0.09420422464609146,0.08492939174175262,0.1199324801564217,-0.1180474907159805,-0.02110130526125431,-0.06917722523212433,0.1529885828495026,-0.08341951668262482,0.1743736416101456,0.1169446706771851,0.01757773384451866,-0.04119010642170906,-0.09467846155166626,0.1110495775938034,0.04717665165662766,-0.1486655026674271,0.08248256891965866,0.04345584288239479,0.06488332152366638,-0.1416355967521667,0.05270294472575188,0.05787716060876846,0.009049562737345695,-0.1212630495429039,-0.1150816082954407,0.0233288649469614,0.1272041499614716,0.1341315656900406,0.02640937082469463,0.1838724464178085,0.2324260771274567,0.0847427025437355,0.09913955628871918,-0.09201724827289581,-0.1403491050004959,0.11358542740345,-0.08415281772613525,-0.1467654556035995,-0.005118575412780046,-0.04538737609982491,0.01942905411124229,-0.01610242761671543,0.004208523314446211,-0.08143758028745651,-0.06267066299915314,0.1083783134818077,0.07753641903400421,-0.007348802872002125,0.1602224260568619,-0.05077390372753143,0.1791536808013916,0.0290319137275219,-0.04726168513298035,-0.005182774737477303,0.01393530610948801,0.05595970898866653,0.03178289532661438,-0.07569053024053574,-0.06223411485552788,-0.103047601878643,0.1080227941274643,-0.002767237834632397,0.1277550160884857,-0.01449598744511604,-0.05861172080039978,-0.1677603870630264,0.03137180954217911,-0.0109736816957593,0.01398298982530832,-0.0440329909324646,0.0007399056921713054,-0.07762440294027328");
    //std::cout << result << std::endl;
    /*cv::Mat img = imread(base_src);
    cv::Mat range = img(cv::Range(0, img.rows), cv::Range(0,img.cols));
    resize(range, range, cv::Size(112, 112));
    cv::imwrite("range.jpg", range);*/
    //ends = clock();
    //std::cout << "result time:" << ends - start << "ms" << std::endl;

}