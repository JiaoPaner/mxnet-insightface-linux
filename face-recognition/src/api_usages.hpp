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
    std::cout << "================================================" << std::endl;
}
