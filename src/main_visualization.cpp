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


int main(int argc, char *argv[])
{
    // *Parse Config
    Config config;
    std::string input_pcd = config.input_pcd;
    std::string output_pcd = config.output_pcd;

    // *Read point cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::io::loadPCDFile(input_pcd, *cloud);
    std::cout << "Number of points in the input cloud is:" << cloud->points.size() << std::endl;

    // *Create the filtering object
    // pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
    // sor.setInputCloud(cloud);
    // sor.setLeafSize(0.1f, 0.1f, 0.1f);
    // sor.filter(*cloud);

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr Normals(new pcl::PointCloud<pcl::PointXYZRGBA>);
    Normals->resize(cloud->size());

    // *K nearest neighbor search
    int KNumbersNeighbor = 10; // numbers of neighbors 7 , 120
    std::vector<int> NeighborsKNSearch(KNumbersNeighbor);
    std::vector<float> NeighborsKNSquaredDistance(KNumbersNeighbor);

    int *NumbersNeighbor = new int[cloud->points.size()];
    pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
    kdtree.setInputCloud(cloud);
    pcl::PointXYZRGBA searchPoint;

    double *SmallestEigen = new double[cloud->points.size()];
    double *MiddleEigen = new double[cloud->points.size()];
    double *LargestEigen = new double[cloud->points.size()];

    double *DLS = new double[cloud->points.size()];
    double *DLM = new double[cloud->points.size()];
    double *DMS = new double[cloud->points.size()];
    double *Sigma = new double[cloud->points.size()];

    //  ************ All the Points of the cloud *******************
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        searchPoint.x = cloud->points[i].x;
        searchPoint.y = cloud->points[i].y;
        searchPoint.z = cloud->points[i].z;

        if (kdtree.nearestKSearch(searchPoint, KNumbersNeighbor, NeighborsKNSearch, NeighborsKNSquaredDistance) > 0)
        {
            NumbersNeighbor[i] = NeighborsKNSearch.size();
        }
        else
        {
            NumbersNeighbor[i] = 0;
        }

        float Xmean;
        float Ymean;
        float Zmean;
        float sum = 0.00;
        // *Computing Covariance Matrix
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += cloud->points[NeighborsKNSearch[ii]].x;
        }
        Xmean = sum / NumbersNeighbor[i];
        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += cloud->points[NeighborsKNSearch[ii]].y;
        }
        Ymean = sum / NumbersNeighbor[i];
        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += cloud->points[NeighborsKNSearch[ii]].z;
        }
        Zmean = sum / NumbersNeighbor[i];

        float CovXX;
        float CovXY;
        float CovXZ;
        float CovYX;
        float CovYY;
        float CovYZ;
        float CovZX;
        float CovZY;
        float CovZZ;

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].x - Xmean));
        }
        CovXX = sum / (NumbersNeighbor[i] - 1);

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].y - Ymean));
        }
        CovXY = sum / (NumbersNeighbor[i] - 1);

        CovYX = CovXY;

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
        }
        CovXZ = sum / (NumbersNeighbor[i] - 1);

        CovZX = CovXZ;

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((cloud->points[NeighborsKNSearch[ii]].y - Ymean) * (cloud->points[NeighborsKNSearch[ii]].y - Ymean));
        }
        CovYY = sum / (NumbersNeighbor[i] - 1);

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((cloud->points[NeighborsKNSearch[ii]].y - Ymean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
        }
        CovYZ = sum / (NumbersNeighbor[i] - 1);

        CovZY = CovYZ;

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((cloud->points[NeighborsKNSearch[ii]].z - Zmean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
        }
        CovZZ = sum / (NumbersNeighbor[i] - 1);

        // *Computing Eigenvalue and EigenVector
        Eigen::Matrix3f Cov;
        Cov << CovXX, CovXY, CovXZ, CovYX, CovYY, CovYZ, CovZX, CovZY, CovZZ;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(Cov);
        if (eigensolver.info() != Eigen::Success)
            abort();

        double EigenValue1 = eigensolver.eigenvalues()[0];
        double EigenValue2 = eigensolver.eigenvalues()[1];
        double EigenValue3 = eigensolver.eigenvalues()[2];

        double Smallest = 0.00;
        double Middle = 0.00;
        double Largest = 0.00;
        if (EigenValue1 < EigenValue2)
        {
            Smallest = EigenValue1;
        }
        else
        {
            Smallest = EigenValue2;
        }
        if (EigenValue3 < Smallest)
        {
            Smallest = EigenValue3;
        }

        if (EigenValue1 <= EigenValue2 && EigenValue1 <= EigenValue3)
        {
            Smallest = EigenValue1;
            if (EigenValue2 <= EigenValue3)
            {
                Middle = EigenValue2;
                Largest = EigenValue3;
            }
            else
            {
                Middle = EigenValue3;
                Largest = EigenValue2;
            }
        }

        if (EigenValue1 >= EigenValue2 && EigenValue1 >= EigenValue3)
        {
            Largest = EigenValue1;
            if (EigenValue2 <= EigenValue3)
            {
                Smallest = EigenValue2;
                Middle = EigenValue3;
            }
            else
            {
                Smallest = EigenValue3;
                Middle = EigenValue2;
            }
        }

        if ((EigenValue1 >= EigenValue2 && EigenValue1 <= EigenValue3) || (EigenValue1 <= EigenValue2 && EigenValue1 >= EigenValue3))
        {
            Middle = EigenValue1;
            if (EigenValue2 >= EigenValue3)
            {
                Largest = EigenValue2;
                Smallest = EigenValue3;
            }
            else
            {
                Largest = EigenValue3;
                Smallest = EigenValue2;
            }
        }

        SmallestEigen[i] = Smallest;
        MiddleEigen[i] = Middle;
        LargestEigen[i] = Largest;

        DLS[i] = std::abs(SmallestEigen[i] / LargestEigen[i]); // std::abs ( LargestEigen[i] -  SmallestEigen[i] ) ;
        DLM[i] = std::abs(MiddleEigen[i] / LargestEigen[i]);   // std::abs (  LargestEigen[i] - MiddleEigen[i] ) ;
        DMS[i] = std::abs(SmallestEigen[i] / MiddleEigen[i]);  // std::abs ( MiddleEigen[i] -  SmallestEigen[i] ) ;
        Sigma[i] = (SmallestEigen[i]) / (SmallestEigen[i] + MiddleEigen[i] + LargestEigen[i]);
    } // For each point of the cloud

    std::cout << "Computing Sigma is Done! " << std::endl;
    // *Color Map For the difference of the eigen values

    double MaxD = 0.00;
    double MinD = cloud->points.size();

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        if (Sigma[i] < MinD)
            MinD = Sigma[i];
        if (Sigma[i] > MaxD)
            MaxD = Sigma[i];
    }

    std::cout << "Minimum is :" << MinD << std::endl;
    std::cout << "Maximum  is :" << MaxD << std::endl;

    //   *****************************************
    // *Red and white (khaki)

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        cloud->points[i].r = 240;
        cloud->points[i].g = 230;
        cloud->points[i].b = 140;
    }

    float step = ((MaxD - MinD) / 100);
    std::vector<Eigen::Vector3f> Edge;
    int EdgeNum = 0;
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        if (Sigma[i] > (MinD + (10 * step)))
        {
            // *Original: 6 * step
            cloud->points[i].r = 255;
            cloud->points[i].g = 0;
            cloud->points[i].b = 0;
            Eigen::Vector3f temp(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
            Edge.push_back(temp);
            EdgeNum++;
        }
    }

    Eigen::Vector3f mu(0.0, 0.0, 0.0);
    float mu_distance = 0.0;
    float mean_distance = 0.0;
    const int N = Edge.size();
    for(size_t i = 0; i < N; i++) {
        mu = mu + Edge[i];
        mean_distance = mean_distance + sqrt(Edge[i].dot(Edge[i]));
    }
    mu = mu / N;
    mu_distance = sqrt(mu.dot(mu));
    mean_distance = mean_distance / N;
    std::cout << "Edge Points Center: " << mu(0) << ' ' << mu(1) << ' ' << mu(2) << std::endl;
    std::cout << "Distance of Edge Points Center: " << mu_distance << std::endl;
    std::cout << "Mean Distance of Edge Points: " << mean_distance << std::endl;

    //   *****************************************

    std::cout << "Number of Edge points  is :" << EdgeNum << std::endl;

    pcl::io::savePCDFileASCII(output_pcd, *cloud);
    std::cerr << "Saved " << cloud->size() << " data points to test_pcd.pcd." << std::endl;

    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {
    }

    return 0;
}