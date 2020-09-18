//
// Created by jiaopan on 8/19/20.
//

extern "C" {

int loadModel(char* mtcnn_model, char* insightface_params, char * insightface_json);

char*  detectFaceByFile(char* src,int type);
/*
    base64_data:"/9j/4AAQSkZJRgABAQE..."
*/
char*  detectFaceByBase64(char* base64_data,int type);

char*  extractFaceFeatureByFile(char* src,int detected,int type);

char*  extractFaceFeatureByByte(unsigned char* src, int width, int height, int channels, int detected,int type);

/*
    base64_data:"/9j/4AAQSkZJRgABAQE..."
*/
char*  extractFaceFeatureByBase64(char* base64_data,int detected,int type);

/*
    distance < 1:same person or not
    base/target:face features
*/
char*  computeDistance(char* base,char* target);

/*
    base/target:image path
*/
char*  computeDistanceByFile(char* base_src, char* target_src, int detected);

/*
    base/target:"/9j/4AAQSkZJRgABAQE..."
*/
char*  computeDistanceByBase64(char* base_data,char* target_data, int detected);

void getUsages();

}


