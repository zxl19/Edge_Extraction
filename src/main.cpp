#include <iostream>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common_headers.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/fpfh.h>

#include <Config.hpp>
#include <edgeExtraction.hpp>

int main(int argc, char *argv[])
{
    // *Parse Config
    Config config;
    std::string input_pcd = config.input_pcd;
    std::string output_pcd = config.output_pcd;

    // *Read point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_edge(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(input_pcd, *cloud);
    std::cout << "Number of points in the input cloud is:" << cloud->points.size() << std::endl;

    // *Create the filtering object
    // pcl::VoxelGrid<pcl::PointXYZI> sor;
    // sor.setInputCloud(cloud);
    // sor.setLeafSize(0.1f, 0.1f, 0.1f);
    // sor.filter(*cloud);

    edgeExtraction(cloud, cloud_edge);

    pcl::io::savePCDFileASCII(output_pcd, *cloud);
    std::cerr << "Saved " << cloud->size() << " data points to test_pcd.pcd." << std::endl;

    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud_edge);
    while (!viewer.wasStopped())
    {
    }

    return 0;
}