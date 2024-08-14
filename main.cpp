#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
using namespace cv;
using namespace pcl;

const double fx = 9.842439e+02;
const double fy = 9.808141e+02;
const double cx = 6.900000e+02;
const double cy = 2.331966e+02;
const double baseline = 5.370000e-01;

// 创建点云
pcl::PointCloud<pcl::PointXYZRGB>::Ptr createPointCloud(const Mat& left_img, const Mat& right_img) {
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
    Mat disparity_sgbm, disparity;
    sgbm->compute(left_img, right_img, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    PointCloud<PointXYZRGB>::Ptr pointcloud(new PointCloud<PointXYZRGB>);
    for (int v = 0; v < left_img.rows; v++) {
        for (int u = 0; u < right_img.cols; u++) {

            // 视差过滤
            if (disparity.at<float>(v, u) <= 5 || disparity.at<float>(v, u) >= 96) continue;

            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * baseline / (disparity.at<float>(v, u));
            // 位置过滤
                if (depth > 20 || -y*depth>3) continue;

            PointXYZRGB point;
            point.x = x * depth;
            point.y = y * depth;
            point.z = depth;

            point.b = left_img.at<Vec3b>(v, u)[0];
            point.g = left_img.at<Vec3b>(v, u)[1];
            point.r = left_img.at<Vec3b>(v, u)[2];
            pointcloud->push_back(point);
        }
    }
    return pointcloud;
}

// 过滤点云
pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    // 体素网格滤波器
    VoxelGrid<PointXYZRGB> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(0.1f, 0.1f, 0.1f);
    PointCloud<PointXYZRGB>::Ptr filteredCloud(new PointCloud<PointXYZRGB>);
    voxel_grid.filter(*filteredCloud);

    // 创建统计滤波器
    StatisticalOutlierRemoval<PointXYZRGB> sor;
    sor.setInputCloud(filteredCloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*filteredCloud);

    return filteredCloud;
}

// 加载位姿数据
std::vector<Eigen::Matrix4f> loadGtPoses(const std::string &gt_pose_file) {
    std::ifstream gt_file(gt_pose_file);
    std::vector<Eigen::Matrix4f> gt_poses;
    std::string line;

    while (std::getline(gt_file, line)) {
        std::istringstream iss(line);
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                iss >> pose(i, j);
            }
        }
        gt_poses.push_back(pose);
    }

    return gt_poses;
}

// 加载图像文件
std::vector<std::string> loadImages(const std::string &image_folder) {
    std::vector<std::string> image_files;
    for (const auto &entry : filesystem::directory_iterator(image_folder)) {
        image_files.push_back(entry.path().string());
    }
    sort(image_files.begin(), image_files.end());
    return image_files;
}

// 根据位姿合并点云
pcl::PointCloud<pcl::PointXYZRGB>::Ptr mergePointClouds(
        const std::vector<std::string> &left_image_files,
        const std::vector<std::string> &right_image_files,
        const std::vector<Eigen::Matrix4f> &gt_poses,
        visualization::PCLVisualizer &viewer){

    PointCloud<PointXYZRGB>::Ptr merged_cloud(new PointCloud<PointXYZRGB>);

    for (size_t i = 0; i < left_image_files.size(); ++i) {
        Mat left_img = imread(left_image_files[i], IMREAD_COLOR);
        Mat right_img = imread(right_image_files[i], IMREAD_COLOR);

        if (left_img.empty() || right_img.empty()) {
            std::cerr << "Error loading images: " << left_image_files[i] << " or " << right_image_files[i] << std::endl;
            continue;
        }

        PointCloud<PointXYZRGB>::Ptr cloud = createPointCloud(left_img, right_img);
        PointCloud<PointXYZRGB>::Ptr filtered_cloud = filterPointCloud(cloud);

        Eigen::Matrix4f pose = gt_poses[i];
        transformPointCloud(*filtered_cloud, *filtered_cloud, pose);
        *merged_cloud += *filtered_cloud;
        std::cerr << "Merged image: " << left_image_files[i] << " and " << right_image_files[i] << std::endl;

        viewer.removeAllPointClouds();
        viewer.addPointCloud(merged_cloud, "merged_cloud");
        viewer.spinOnce(10);

        //test
        if (i == 500) {
            break;
        }
    }

    return merged_cloud;
}

// 处理点云数据集
void processDataset(const string &left_image_folder, const string &right_image_folder, const string &gt_pose_file) {
    // 加载图像文件
    vector<string> left_image_files = loadImages(left_image_folder);
    vector<string> right_image_files = loadImages(right_image_folder);

    // 加载位姿数据
    std::vector<Eigen::Matrix4f> gt_poses = loadGtPoses(gt_pose_file);

    // 初始化可视化
    visualization::PCLVisualizer viewer("Point Cloud Viewer");
    viewer.setBackgroundColor(128, 128, 128);

    // 合并点云
    PointCloud<PointXYZRGB>::Ptr merged_cloud = mergePointClouds(left_image_files, right_image_files, gt_poses, viewer);

    // 相机坐标系转PCL坐标系
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform(0, 0) = 1.0;  // x -> x
    transform(1, 1) = -1.0; // y -> -y
    transform(2, 2) = -1.0; // z -> -z

    // 应用变换
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::transformPointCloud(*merged_cloud, *transformed_pointcloud, transform);

    io::savePCDFile("result_cloud.pcd", *transformed_pointcloud);
    std::cout << "Merged point cloud saved to result_cloud.pcd" << std::endl;
    viewer.spin();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <left_image_folder> <right_image_folder> <gt_pose_file>" << std::endl;
        return 1;
    }

    std::string left_image_folder = argv[1];
    std::string right_image_folder = argv[2];
    std::string gt_pose_file = argv[3];

    processDataset(left_image_folder, right_image_folder, gt_pose_file);

    return 0;
}
