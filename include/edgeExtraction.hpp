#include <pcl/point_types.h>

void edgeExtraction(const pcl::PointCloud<pcl::PointXYZI>::Ptr &pointcloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr &pointcloud_edge)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr Normals(new pcl::PointCloud<pcl::PointXYZI>);
    Normals->resize(pointcloud_in->size());

    // *K nearest neighbor search
    int KNumbersNeighbor = 10; // numbers of neighbors 7 , 120
    std::vector<int> NeighborsKNSearch(KNumbersNeighbor);
    std::vector<float> NeighborsKNSquaredDistance(KNumbersNeighbor);

    int *NumbersNeighbor = new int[pointcloud_in->points.size()];
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(pointcloud_in);
    pcl::PointXYZI searchPoint;

    double *SmallestEigen = new double[pointcloud_in->points.size()];
    double *MiddleEigen = new double[pointcloud_in->points.size()];
    double *LargestEigen = new double[pointcloud_in->points.size()];

    double *DLS = new double[pointcloud_in->points.size()];
    double *DLM = new double[pointcloud_in->points.size()];
    double *DMS = new double[pointcloud_in->points.size()];
    double *Sigma = new double[pointcloud_in->points.size()];

    //  ************ All the Points of the cloud *******************
    for (size_t i = 0; i < pointcloud_in->points.size(); ++i)
    {
        searchPoint.x = pointcloud_in->points[i].x;
        searchPoint.y = pointcloud_in->points[i].y;
        searchPoint.z = pointcloud_in->points[i].z;

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
            sum += pointcloud_in->points[NeighborsKNSearch[ii]].x;
        }
        Xmean = sum / NumbersNeighbor[i];
        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += pointcloud_in->points[NeighborsKNSearch[ii]].y;
        }
        Ymean = sum / NumbersNeighbor[i];
        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += pointcloud_in->points[NeighborsKNSearch[ii]].z;
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
            sum += ((pointcloud_in->points[NeighborsKNSearch[ii]].x - Xmean) * (pointcloud_in->points[NeighborsKNSearch[ii]].x - Xmean));
        }
        CovXX = sum / (NumbersNeighbor[i] - 1);

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((pointcloud_in->points[NeighborsKNSearch[ii]].x - Xmean) * (pointcloud_in->points[NeighborsKNSearch[ii]].y - Ymean));
        }
        CovXY = sum / (NumbersNeighbor[i] - 1);

        CovYX = CovXY;

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((pointcloud_in->points[NeighborsKNSearch[ii]].x - Xmean) * (pointcloud_in->points[NeighborsKNSearch[ii]].z - Zmean));
        }
        CovXZ = sum / (NumbersNeighbor[i] - 1);

        CovZX = CovXZ;

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((pointcloud_in->points[NeighborsKNSearch[ii]].y - Ymean) * (pointcloud_in->points[NeighborsKNSearch[ii]].y - Ymean));
        }
        CovYY = sum / (NumbersNeighbor[i] - 1);

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((pointcloud_in->points[NeighborsKNSearch[ii]].y - Ymean) * (pointcloud_in->points[NeighborsKNSearch[ii]].z - Zmean));
        }
        CovYZ = sum / (NumbersNeighbor[i] - 1);

        CovZY = CovYZ;

        sum = 0.00;
        for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii)
        {
            sum += ((pointcloud_in->points[NeighborsKNSearch[ii]].z - Zmean) * (pointcloud_in->points[NeighborsKNSearch[ii]].z - Zmean));
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

    // std::cout << "Computing Sigma is Done! " << std::endl;
    // *Color Map For the difference of the eigen values

    double MaxD = 0.00;
    double MinD = pointcloud_in->points.size();

    for (size_t i = 0; i < pointcloud_in->points.size(); ++i)
    {
        if (Sigma[i] < MinD)
            MinD = Sigma[i];
        if (Sigma[i] > MaxD)
            MaxD = Sigma[i];
    }

    // std::cout << "Minimum is :" << MinD << std::endl;
    // std::cout << "Maximum  is :" << MaxD << std::endl;

    //   *****************************************

    float step = ((MaxD - MinD) / 100);
    std::vector<Eigen::Vector3f> Edge;
    int EdgeNum = 0;
    for (size_t i = 0; i < pointcloud_in->points.size(); ++i)
    {
        if (Sigma[i] > (MinD + (10 * step)))
        {
            // *Original: 6 * step
            pointcloud_edge->push_back(pointcloud_in->points[i]);
            Eigen::Vector3f temp(pointcloud_in->points[i].x, pointcloud_in->points[i].y, pointcloud_in->points[i].z);
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
    // std::cout << "Edge Points Center: " << mu(0) << ' ' << mu(1) << ' ' << mu(2) << std::endl;
    // std::cout << "Distance of Edge Points Center: " << mu_distance << std::endl;
    // std::cout << "Mean Distance of Edge Points: " << mean_distance << std::endl;
    // std::cout << "Number of Edge points  is :" << EdgeNum << std::endl;
}