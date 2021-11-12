// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "utility.h"


class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    // 地面点点云
    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    // 初始化为FLT_MAX
    cv::Mat rangeMat; // range matrix for range image
    // 初始化为0，-1表示不用于分割的点，999999表示点数不足的簇点
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    // 初始化为1
    int labelCount;

    float startOrientation;
    float endOrientation;

    // 该msg类型是作者自定义的
    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    // 初始化为“上右左下”四邻域
    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    // 所有被push进queue的元素的下标，这些点就是当前簇的所有元素          评论：其实用局部变量也行
    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for BFS (breadth-first search) process of segmentation
    uint16_t *queueIndY;

public:
    ImageProjection():
        nh("~"){

        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;  // 请记住，作者使用强度分量为 -1 来标注该点是否为“无效点”

        allocateMemory();
        resetParameters();
    }

    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        // 如果左上角为二维数组的(0,0)下标元素的话
        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);  // 上方邻居
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);  // 右侧邻居
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);  // 左侧邻居
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);  // 下方邻居

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }

    void resetParameters(){
        laserCloudIn->clear();
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));  // groundMat中所有像素值均初始化为0
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    ~ImageProjection(){}

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn);
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
    }
    
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        //    ROS Message 类型的点云，转换为 PCL 类型的点云，存入成员变量
        copyPointCloud(laserCloudMsg);
        // 2. Start and end angle of a scan
        //    计算该帧点云的起始、终止方位角，存入成员变量
        findStartEndAngle();
        // 3. Range image projection
        //    利用柱面投影法，将点云转换为深度图，存入成员变量
        projectPointCloud();
        // 4. Mark ground points
        //    在深度图上，将点云划分为地面点和非地面点
        groundRemoval();
        // 5. Point cloud segmentation
        //    点云分割
        cloudSegmentation();
        // 6. Publish all clouds
        //    发布点云话题消息
        publishCloud();
        // 7. Reset parameters for next iteration
        resetParameters();
    }

    void findStartEndAngle(){
        // start and end orientation of this cloud
        // atan2(y,x) 返回相对x轴正方向（逆时针为正）的航向角，范围为 (-π,π]
        // 作者加了 负号(-)，变为 顺时针为正
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;

        // segMsg.endOrientation - segMsg.startOrientation 的范围为 (-2π,2π)，因此这个if是不可能进入的
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        }
        // 理想情况下一帧点云为一圈扫描，segMsg.endOrientation - segMsg.startOrientation 接近0，因此正常会进入该 else if
        else if (segMsg.endOrientation - segMsg.startOrientation < M_PI){
            // 会使得 segMsg.endOrientation ≈ segMsg.startOrientation + 2π
            segMsg.endOrientation += 2 * M_PI;
        }

        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
    }

    void projectPointCloud(){
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();

        for (size_t i = 0; i < cloudSize; i++){

            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            // find the row and column index in the iamge for this point

            // 俯仰角（角度制），水平为0°，仰角为正，俯角为负
            verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
            // ① 转换为雷达最下方一根线为0°，向上角度增加
            // ② 根据俯仰角，计算当前点为第几根线的点（最下方的线编号为0，向上编号增加）   评论：俯仰角到线号的映射方式不太严谨
            // 线号，在深度图中为行号，因此作者起了变量名：rowInd(ex)
            rowIdn = (verticalAngle + ang_bottom) / ang_res_y;

            // 去除异常线号
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // atan2(x,y) 返回相对y轴正方向（顺时针为正）的航向角，范围为 (-π,π]
            // 水平方位角（角度制）
            // Velodyne的坐标系为：x朝右，y朝前，z朝上
            // 因此，前方方位角为0°，顺时针为正，右侧为90°，左侧为-90°
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            // ① 通过 -90°，转换为 相对x轴正方向（顺时针为正）的航向角，范围为 (-3/2 π, 1/2 π]      评论：刚才直接写 -atan2(y,x)不就完事了
            //    相当于，右侧方位角为0°，顺时针为正
            // ② 通过 ÷angle_res_x，转换为扇区号
            // ③ 通过 +Horizon_SCAN/2，正左方(-x轴)变为第0号扇区，顺时针扇区号增加
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;

            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            // 去除异常扇区号
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            // 该点与雷达的距离，即激光射程
            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            if (range < 0.1)  // 舍弃射程1dm以下的点，估计是为了去除 (0,0,0) 这种点，因为有些雷达会将未反射的无效点填充为(0,0,0)
                continue;
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // (深度图的)行号 + （ 列号 / 1万 ），将该信息存入intensity分量的意义是什么？
            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn  + rowIdn * Horizon_SCAN;

            // fullCloud相当于原始点云按照空间顺序整理过了，intensity存的是行列号之和信息
            fullCloud->points[index] = thisPoint;

            // fullInfoCloud的intensity存的是深度值（其实可以直接用xyz坐标算出来，算是冗余信息吧）
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
        }
    }


    /*
    作用：
    ① 为成员变量 groundMat 赋值，-1 表示无效点，1 表示地面点
    ② 为成员变量 labelMat 赋值，-1 表示不用于分割点点，包含地面点和无效点
    ③ 为成员变量 groundCloud 赋值，其为 fullCloud 中的全部地面点
    */
    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        // 一列一列地遍历
        for (size_t j = 0; j < Horizon_SCAN; j++){       // j表示列号
            for (size_t i = 0; i < groundScanInd; i++){  // i表示行号

                lowerInd = j + ( i )*Horizon_SCAN;  // 当前遍历像素的一维索引
                upperInd = j + (i+1)*Horizon_SCAN;  // 当前遍历像素上方像素的一维索引

                // 作者使用intensity分量为-1，标注“无效点”
                // 如果 当前遍历像素 或 当前遍历像素上方的像素，其中一个为无效点
                if (fullCloud->points[lowerInd].intensity == -1 || fullCloud->points[upperInd].intensity == -1){
                    // no info to check, invalid points
                    // 在 groundMat 中将其标注为 -1，表示“无效点”
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }
                    
                // 计算这两个点，在x、y、z分量上的差，也可以理解为，相对当前点，其上方点在空间中的坐标
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                // 计算仰角
                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                // 如果 仰角<10°，可以认为当前点和其上方点都是地面点
                // 疑问：
                // ① sensorMountAngle的含义不清楚
                // ② 只要是水平面，都会被认定为地面点
                if (abs(angle - sensorMountAngle) <= 10){
                    // 在 groundMat 中标注为1，表示“地面点”
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }

        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
        // 一行一行地遍历
        for (size_t i = 0; i < N_SCAN; i++){            // i表示行号
            for (size_t j = 0; j < Horizon_SCAN; j++){  // j表示列号
                // 将 ①地面点 ②无效点 在labelMat中标注为 -1，表示接下来不用这些点进行“点云分割”操作
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }

        // 如果有节点订阅了 /ground_cloud 话题（该话题的消息类型为 PointCloud2）
        if (pubGroundCloud.getNumSubscribers() != 0){
            // 一行一行地遍历
            // 将地面点加入到 groundCloud 成员变量点云中，最终 groundCloud 为 fullCloud 的子集
            for (size_t i = 0; i <= groundScanInd; i++){    // i表示行号
                for (size_t j = 0; j < Horizon_SCAN; j++){  // j表示列号
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    /*
    作用：
    ① 对每个点调用 labelComponents() 进行聚类，相当于更新 labelMat 成员变量
    ② 为成员变量 segmentedCloud 赋值，存放接下来用于提取特征的点
    ③ 为成员变量 segMsg 进行赋值，其有多个属性，这些属性都是对 segmentedCloud 的补充说明
    ④ 为成员变量 segmentedCloudPure 进行赋值，将簇号存入intensity分量

    */
    void cloudSegmentation(){
        // segmentation process
        // 一列一列地遍历
        for (size_t i = 0; i < N_SCAN; i++){            // i表示列号
            for (size_t j = 0; j < Horizon_SCAN; j++){  // j表示行号
                // labelMat 中的值为0，表示该点用于分割
                if (labelMat.at<int>(i,j) == 0){
                    // 以当前点为中心，进行聚类
                    labelComponents(i, j);
                }
            }
        }

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        // 一行一行地遍历
        for (size_t i = 0; i < N_SCAN; i++) {            // i表示行号

            // 这个变量含义很不好理解，猜测是，当前行（ring）在 segmentedCloud 中的起始下标，+5可能是待会算曲率时，前几个点左侧没有点，不好算曲率
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; j++) {  // j表示列号
                // 除了无效点之外
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                    // outliers that will not be used for optimization (always continue)
                    // 999999 表示聚类后点数不足的簇的点
                    if (labelMat.at<int>(i,j) == 999999){
                        // 在仰角>0的激光线中，水平每隔5个点取一个离群点，放入成员变量 outlierCloud 中
                        if (i > groundScanInd && j % 5 == 0){
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                            continue;
                        }else{
                            continue;
                        }
                    }

                    // majority of ground points are skipped
                    // 大部分地面点都被跳过了，只每5个选用1个                        评论：可以理解为对地面点进行降采样吗
                    if (groundMat.at<int8_t>(i,j) == 1){
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                            continue;
                    }

                    // 首先，成员变量 segmentedCloud 存放的是接下来用于特征提取的点
                    // 然后，segMsg 中的 segmentedCloudGroundFlag 属性用来标注每个点是否为地面点
                    //      segMsg 中的 segmentedCloudColInd 属性用来标注每个点在深度图中的列号
                    //      segMsg 中的 segmentedCloudRange 属性用来标注每个点的深度值

                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    sizeOfSegCloud++;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization
        // 如果有节点订阅了 /segmented_cloud_pure 话题
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            // 一行一行地遍历
            for (size_t i = 0; i < N_SCAN; i++){            // i表示行号
                for (size_t j = 0; j < Horizon_SCAN; j++){  // j表示列号
                    // 如果当前点被分配了有效的簇号
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        // 将当前点加入到 segmentedCloudPure 中，intensity分量置为簇号
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);
                    }
                }
            }
        }
    }

    /*
    作用：
    ① 以参数(row,col)为中心使用BFS进行聚类，如果聚类完成后，如果为有效簇（簇中元素数量足够），则将这些点在 labelMat 中的值置为簇号，如果为无效簇，则置为 999999
       即，为成员变量 labelMat 赋值

    假如当前 (row,col) 已经聚类过了，调用该方法，会进入while循环，但是由于邻域也已聚类过了，因此会退出while循环，当前簇的点数为1，被认为是无效簇。
    个人认为，不如在方法最前面加个if判断一下，当前(row,col)在labelMat中的值是否为0，不为0直接返回。
    */
    void labelComponents(int row, int col){
        // use std::queue、std::vector、std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};  // 用来记录当前簇跨越了哪些线号

        // queueIndX、queueIndY 使用数组模拟队列，队列中元素为深度图的行列下标
        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;  // 指向队头元素
        int queueEndInd = 1;  // 指向队尾元素的下一个元素（可以认为是新元素入队时的位置）

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;  // 和allPushedIndX、allPushedIndY配套使用，表示当前簇的点的数量
        
        // 广度优先搜索，使用队列实现
        while(queueSize > 0){
            // Pop point
            // 队头元素出队
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            queueSize--;
            queueStartInd++;

            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;  // labelCount 表示簇编号，从1开始

            // Loop through all the neighboring grids of popped grid
            // 遍历队头元素的四邻域
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); iter++){
                // new index
                // 邻域元素在深度图中的下标
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                // index should be within the boundary
                // 检查行号下标是否越界
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;

                // at range image margin (left or right side)
                // 将深度图弯曲成一个圆柱表面，因此不会存在列号越界的问题
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;

                // prevent infinite loop (caused by put already examined point back)
                // 如果该邻居点已经分配了簇号，或者为不参与聚类的点(-1)，则跳过
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                // 比较队头点和邻居点的深度值大小，将较大的深度值赋值给 d1，将较小的深度值赋值给 d2
                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                              rangeMat.at<float>(thisIndX, thisIndY));

                // 这里的alpha表示激光雷达在该方向上的角分辨率（角度制）                评论：建议在纸上画一个示意图理解
                if ((*iter).first == 0)
                    alpha = segmentAlphaX;
                else
                    alpha = segmentAlphaY;

                // angle 能够反应两个点之间的远近程度，angle∈(0,90°)，angle越小说明两个点越远，angle越大说明两个点越近
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){  // 分割阈值为60°
                    // 当前邻居入队
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    // 标记当前邻居为相同的簇号
                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;  // 记录当前簇跨越了 thisIndX 号线

                    // allPushedIndX、allPushedIndY记录了当前簇的所有点的行列下标
                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }

        // check if this segment is valid
        bool feasibleSegment = false;
        // 如果簇中点数>=30，则认为是有效簇
        if (allPushedIndSize >= 30){
            feasibleSegment = true;
        }
        // 否则，如果簇中点数>=5
        // 统计当前簇跨越了几根线，如果跨越的线数>=3，也认为是有效簇
        else if (allPushedIndSize >= segmentValidPointNum){
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; i++){
                if (lineCountFlag[i] == true){
                    lineCount++;
                }
            }
            if (lineCount >= segmentValidLineNum){
                feasibleSegment = true;
            }
        }

        // segment is valid, mark these points
        if (feasibleSegment == true){
            labelCount++;  // 簇号++，用于下个簇的聚类
        }
        // 如果是无效簇，则将簇中所有点在 labelMat 中标记为 999999                           评论：作者不会用枚举类型吗
        else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; i++){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
            }
        }
    }

    
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        // 将成员变量 segMsg，发布到 /segmented_cloud_info 话题
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);

        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;

        // 将成员变量 outlierCloud，发布到 /outlier_cloud 话题
        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);

        // segmented cloud with ground
        // 将成员变量 segmentedCloud, 发布到 /segmented_cloud 话题
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);

        // projected full cloud
        // 如果有节点订阅了 /full_cloud_projected 话题
        // 则将成员变量 fullCloud 发布到该话题
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }
        // original dense ground cloud
        // 如果有节点订阅了 /ground_cloud 话题
        // 则将成员变量 groundCloud 发布到该话题
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground
        // 如果有节点订阅了 /segmented_cloud_pure 话题
        // 则将成员变量 segmentedCloudPure 发布到该话题
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }

        // projected full cloud info
        // 如果有节点订阅了 /full_cloud_info 话题
        // 则将成员变量 fullInfoCloud 发布到该话题
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};




int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP;

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
