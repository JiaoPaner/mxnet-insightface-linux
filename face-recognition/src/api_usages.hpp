//
// Created by jiaopan on 8/20/20.
//

void getPythonUsage(){
    std::cout << "use api." << std::endl;
    std::cout << "result.status = 1 ---> success" << std::endl;
    std::cout << "result.status = -1 ---> error" << std::endl;
    std::cout << "================================================\n"
            <<"python usage:\n"
            <<"import ctypes\n"
            <<"so = ctypes.cdll.LoadLibrary(\"libface_recognition.so\")\n"
            <<"model_params = \"model/feature_model/128/model-0000.params\"\n"
            <<"model_json = \"model/feature_model/128/model-symbol.json\"\n"
            <<"mtcnn_model = \"model/mtcnn\"\n"
            <<"status = so.loadModel(mtcnn_model.encode(), model_params.encode(), model_json.encode()) \n"
            <<"file = \"/home/jiaopan/Downloads/dit.jpg\"\n"
            <<"result = so.extractFaceFeatureByFile(file.encode(),1,1)\n"
            <<"result = ctypes.string_at(result, -1).decode(\"utf-8\")\n"
            <<"print(result)\n"
            <<"{\"status\":1,\"msg\":\"register success\",\"embeddings\":[{\"embedding\":\"0.1246355548501015,0.01359560526907444,...,0.04933002963662148\"}]})\n"
            << std::endl;
    std::cout << "================================================\n"
            <<"Java usage:\n"
            <<"import com.sun.jna.Library;\n"
            <<"import com.sun.jna.Native;\n"
            <<"public interface FaceLibrary extends Library {\n"
            <<"   "<<"FaceLibrary instance = (FaceLibrary) Native.loadLibrary(\"dir/libface_recognition.so\",FaceLibrary.class);\n"
            <<"   "<<"void getUsages();\n"
            <<"   "<<"int loadModel(String mtcnn_model,String insightface_params,String insightface_json);\n"
            <<"   "<<"String extractFaceFeatureByFile(String src, int detected, int type);\n"
            <<"}\n"
            <<"public static void main(String[] args) {\n"
            <<"   "<<"String model_params = \"model/feature_model/128/model-0000.params\";\n"
            <<"   "<<"String model_json = \"model/feature_model/128/model-symbol.json\";\n"
            <<"   "<<"String mtcnn_model = \"model/mtcnn\";\n"
            <<"   "<<"FaceLibrary.instance.getUsages();\n"
            <<"   "<<"FaceLibrary.instance.loadModel(mtcnn_model,model_params,model_json);\n"
            <<"   "<<"String file = \"/home/jiaopan/Downloads/dit.jpg\";\n"
            <<"   "<<"String result = FaceLibrary.instance.extractFaceFeatureByFile(file,0,1);\n"
            <<"   "<<"System.out.println(result);\n"
            <<"}\n"
            <<"{\"status\":1,\"msg\":\"register success\",\"embeddings\":[{\"embedding\":\"0.1246355548501015,0.01359560526907444,...,0.04933002963662148\"}]})\n"
            << std::endl;
    std::cout << "================================================" << std::endl;
}
