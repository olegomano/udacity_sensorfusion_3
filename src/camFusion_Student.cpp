
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

#include "camFusion.hpp"
#include "dataStructures.h"

double distance(const cv::Point& p1, const cv::Point& p2){
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return dx*dx + dy*dy;
}

double distance(const LidarPoint& p1, const LidarPoint& p2){
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;

    return dx*dx + dy*dy + dz*dz;
    
}


double distance(const cv::Rect& r1, const cv::Rect& r2){
    float r1Cx = (r1.tl().x + r1.br().x) / 2;
    float r1Cy = (r1.tl().y + r1.br().y) / 2;
    float r2Cx = (r2.tl().x + r2.br().x) / 2;
    float r2Cy = (r2.tl().y + r2.br().y) / 2;
    
    return (r1Cx - r2Cx)*(r1Cx - r2Cx) + (r1Cy - r2Cy)*(r1Cy - r2Cy);

}


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        std::vector<std::vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (std::vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, xwmax = -1e8,ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            xwmax = xwmax>xw ? xwmax : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
            //std::cout << cv::Point(x,y) << std::endl;
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, xh=%2.2f m", xwmin, xwmax -  xwmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    std::string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches){
    double meanDistance = 0;
    
    std::vector<std::pair<double,cv::DMatch>> matches;

    for(int i =0; i < kptMatches.size(); i++){
        cv::KeyPoint pointPrev = kptsPrev[kptMatches[i].queryIdx];
        cv::KeyPoint pointCurr = kptsCurr[kptMatches[i].trainIdx];

        if(boundingBox.roi.contains(pointPrev.pt) && boundingBox.roi.contains(pointCurr.pt)){
            double d = distance(pointPrev.pt,pointCurr.pt);
            meanDistance+=d;
            matches.push_back(std::make_pair(d,kptMatches[i]));
        }
    }
    meanDistance /= matches.size();

    for(int i = 0; i < matches.size(); i++){
        if(matches[i].first < meanDistance){
            boundingBox.kptMatches.push_back(matches[i].second);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg){
    
    std::map<int,int> prevCurrMap;
    for(int i  = 0; i < kptMatches.size(); i++){
        prevCurrMap[kptMatches[i].queryIdx] = kptMatches[i].trainIdx;
    }

    std::vector<double> ratios;
    for(auto iter = prevCurrMap.begin(); iter != prevCurrMap.end(); ++iter){
        cv::KeyPoint prevPoint = kptsPrev[iter->first];
        cv::KeyPoint currPoint = kptsCurr[iter->second];

        for(auto diter = prevCurrMap.begin(); diter != prevCurrMap.end(); ++diter){
            if(diter->first != iter->first){
                double dprev = distance(prevPoint.pt, kptsPrev[diter->first].pt);
                double dcurr = distance(currPoint.pt, kptsCurr[diter->second].pt);
               // std::cout << diter->first << "->" << diter->second << " " << dprev << " " << dcurr << std::endl;
                if(dprev != 0 && dcurr != 0){
                    double ratio = sqrt(dcurr / dprev);
                    if(ratio != 1.0){
                        ratios.push_back( ratio);
                    }
                }
            }
        }
    }
    std::sort(ratios.begin(),ratios.end());
    double medianRatio = ratios[ratios.size() / 2]; 
    TTC = -1.0 / ( (1 - medianRatio)*frameRate );
    std::cout << "TTC Camera " << TTC << " seconds, median " << medianRatio <<" " << frameRate << std::endl;
}

LidarPoint getIndicativePoint(std::vector<LidarPoint>& points){
    std::sort(points.begin(),points.end(),[]( const LidarPoint& lhs, const LidarPoint& rhs ){
        return lhs.x < rhs.x;
    });
    double meanX =0;
    for(int i =0; i < points.size(); i++){
        meanX+=points[i].x;
    }
    meanX/=points.size();

    double stdDev = 0;
    for(int i = 0; i < points.size(); i++){
        stdDev+=((points[i].x - meanX)*(points[i].x - meanX));
    }
    stdDev /= (points.size() - 1);
    stdDev = sqrt(stdDev);

    for(int i = 0; i < points.size(); i++){
        if( abs(points[i].x - meanX) < stdDev/1.3){
            return points[i];
        }
    }
    return points[0];
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC){
    
    LidarPoint prev = getIndicativePoint(lidarPointsPrev);
    LidarPoint curr = getIndicativePoint(lidarPointsCurr);

    double velocity = (prev.x - curr.x) * frameRate;
    TTC = prev.x / velocity; 
    //std::cout << "TTC Lidar " << TTC << " seconds " << " Velocity " << velocity << " " << prev.x << " " << curr.x << "  " << frameRate << std::endl;
    
}


int findOwner(cv::KeyPoint& kp, std::vector<BoundingBox>& boxes ){
    for(int i = 0; i < boxes.size(); i++){
        if(boxes[i].roi.contains(kp.pt)){
            return i;
        };
    }
    return -1;
}




void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame){
    //fill in bbBestMatches

    std::map<int,std::map<int,int>> bbConnect; //by index

    for(int i =0; i < matches.size(); i++){
        cv::KeyPoint pointPrev = prevFrame.keypoints[matches[i].queryIdx];
        cv::KeyPoint pointCurr = currFrame.keypoints[matches[i].trainIdx];

        int ownerBoxPrev = findOwner(pointPrev,prevFrame.boundingBoxes);
        int ownerBoxCurr = findOwner(pointCurr,currFrame.boundingBoxes);
       
        if(ownerBoxPrev >= 0 && ownerBoxCurr >= 0){
            if(bbConnect.find(ownerBoxPrev) == bbConnect.end()){
                bbConnect[ownerBoxPrev] = std::map<int,int>();
            }
            if(bbConnect[ownerBoxPrev].find(ownerBoxCurr) == bbConnect[ownerBoxPrev].end()){
                bbConnect[ownerBoxPrev][ownerBoxCurr] = 0;
            }
            bbConnect[ownerBoxPrev][ownerBoxCurr]++;
        }
    }
    for(auto iter = bbConnect.begin(); iter!=bbConnect.end(); ++iter){
        int boxPrevIndex = iter->first;
        std::map<int,int> connections = iter->second;
        int maxMatches      = 0;
        int maxMatchesIndex = -1;
        for(auto iterConnect = connections.begin(); iterConnect != connections.end(); ++iterConnect){
            if(iterConnect->second > maxMatches){
                maxMatches = iterConnect->second;
                maxMatchesIndex = iterConnect->first;
            }
            //std::cout << boxPrevIndex << "->" << iterConnect->first << ": " << iterConnect->second << std::endl; 
        }
        if(maxMatchesIndex != -1){ //if too little connections we ignore it
            //since we are tracking a car, my assumption is the car can only move so much in one frame, so the bounding boxes need to be close to eachother
            cv::Rect prevFrameBox = prevFrame.boundingBoxes[boxPrevIndex].roi;
            cv::Rect currFrameBox = currFrame.boundingBoxes[maxMatchesIndex].roi;
            float d = distance(prevFrameBox,currFrameBox);
         //   std::cout << boxPrevIndex << "<->" << maxMatchesIndex << " " << d << std::endl;
            bbBestMatches[prevFrame.boundingBoxes[boxPrevIndex].boxID] = currFrame.boundingBoxes[maxMatchesIndex].boxID;

       //     std::cout << currFrameBox << " " << currFrame.boundingBoxes[maxMatchesIndex].lidarPoints.size() << " matched to " << prevFrameBox << " " << prevFrame.boundingBoxes[boxPrevIndex].lidarPoints.size() << std::endl;
        }
    }

    for(auto iter = bbBestMatches.begin(); iter != bbBestMatches.end(); ++iter){
    //    std::cout << currFrame.boundingBoxes[iter->first].roi <<  " " << currFrame.boundingBoxes[iter->first].lidarPoints.size() <<  " matched to " << " " <<prevFrame.boundingBoxes[iter->second].roi << " " << prevFrame.boundingBoxes[iter->second].lidarPoints.size() << std::endl ;
    }
    // ...
}
