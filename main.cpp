#include <opencv2/opencv.hpp>
#include <iostream>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include<omp.h>

using namespace cv;
using namespace std;


Mat get_kernel(int size) {
    /*
     * 定义kernel
     */
    Mat kernel(size, size, CV_32F);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (j == 0)
                kernel.at<float>(i, j) = 1;
            else
                kernel.at<float>(i, j) = 0;
        }
    }
    return kernel;
}

Mat conv2(const Mat &img, Mat kernel) {
    /*
     * 卷积操作
     */
    int img_weight = img.cols;
    int img_height = img.rows;
    int kernel_size = kernel.rows; // 卷积核一般为方形
    int pad_h = kernel_size / 2; // 对原图进行扩充，目的是保持输出与原图大小一致
    int pad_w = kernel_size / 2;
    Mat result(img_height, img_weight, CV_32F);

    // 对图像进行边缘填充
    Mat img_pad;
    copyMakeBorder(img, img_pad, pad_h, pad_h, pad_w, pad_w, BORDER_REPLICATE);

    for (int i = pad_h; i < img_height + pad_h; i++) {
        for (int j = pad_w; j < img_weight + pad_w; j++) {
            float sum = 0;
            for (int m = 0; m < kernel_size; m++) {
                for (int n = 0; n < kernel_size; n++) {
                    sum += img_pad.at<float>(i - pad_h + m, j - pad_w + n) * kernel.at<float>(m, n);
                }
            }
            result.at<float>(i - pad_h, j - pad_w) = sum;
        }
    }
    return result;
}

int main() {
    while (true) {
        cout << "选择程序运行模式：" << endl;
        cout << "0:串行  1:并行  2:终止" << endl;
        VideoCapture cap("video.avi");
        if (!cap.isOpened()) {
            cout << "Error opening video file" << endl;
            return -1;
        }
        double fps = cap.get(CAP_PROP_FPS);
        int frameNum = cap.get(CAP_PROP_FRAME_COUNT); //获取总帧数
        cout << "总帧数:" << frameNum << endl;
        double videoWidth = cap.get(CAP_PROP_FRAME_WIDTH); //获取帧宽度
        double videoHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //获取帧高度

        Mat kernel = get_kernel(2); //定义卷积核

        Mat *frames = new Mat[frameNum]; //动态开辟内存，保存每帧内容
        Mat *outputFrames = new Mat[frameNum]; //动态开辟内存，保存每帧内容
//        cout << &outputFrames[frameNum]-outputFrames << endl;
        Mat frame;

        for (int i = 0; i < frameNum; i++) {
            // openMP与openCV存在冲突，读取视频帧的时候不支持并行
            cap.read(frames[i]);
//            cout << "当前帧大小:" << i << "," << frames[i].cols << "," << frames[i].rows << endl;
            if (frames[i].empty()) {
                cout << "读取到错误帧:" << i << "," << frames[i].cols << "," << frames[i].rows << endl;
                break;
            }
        }

        int choice;
        cin >> choice;
        auto start = chrono::high_resolution_clock::now();

        string save_file;

//        VideoWriter  video_writer(save_file,CAP_PROP_FOURCC,fps,Size(videoHeight,videoWidth));
        VideoWriter video_writer;

        switch (choice) {
            case 0:
                save_file = "serial.avi";
                video_writer.open(save_file, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                                  Size(videoWidth, videoHeight));

                for (int i = 0; i < frameNum; i++) {

                    Mat img_conv = conv2(frames[i], kernel);
                    outputFrames[i] = img_conv;
                    if (i % 10 == 0) {
                        cout << "当前处理第" << i << "帧" << endl;
                        std::ostringstream oss;

                        oss << "D:\\CLionProjects\\ParallelComputing\\output1\\" << i << ".jpg";
//                        cout<<oss.str()<<endl;
                        imwrite(oss.str(), img_conv);
                    }
                }

                break;
            case 1:
                save_file = "parallel.avi";
#pragma omp parallel for
                for (int i = 0; i < frameNum; i++) {
                    Mat img_conv = conv2(frames[i], kernel);
                    outputFrames[i] = img_conv;
                    if (i % 10 == 0) {
                        cout << "当前处理第" << i << "帧" << endl;
                        std::ostringstream oss;
                        oss << "D:\\CLionProjects\\ParallelComputing\\output2\\" << i << ".jpg";
                        imwrite(oss.str(), img_conv);
                    }
                }
                break;
            default:
                std::cout << "2" << std::endl;
                break;
        }

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        if (choice != 2)
            cout << "Time taken by program: " << duration.count() << " milliseconds" << endl;
        // 释放空间
        cap.release();
//        video_writer.release();
        delete[] frames;
        delete[] outputFrames;

    }

    return 0;
}
