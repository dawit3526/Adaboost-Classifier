/****************************************************
*	Adaptive Boost Classifer class                  *
*	Copyright, 2016, Visual Intelligence Studio     *
*	Carnegie Mellon University                      *
*                                                   *
*****************************************************/
//	Consists of implementation of Adaptive Boost
//  algorithm. It can be used in two class
//  classification. Built-in functions allow us to
//  perform training, calculate confusion matrix etc.
//  The data for training is read from a specified 
//  .csv file

#ifndef ADABOOST_CLASSIFIER_H
#define ADABOOST_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "CustomClassifier.h"
#include "misc.h"

using namespace std;
using namespace cv;

class AdaBoostClassifier : public CustomClassifier{
private:
	CvBoost boost;						
	CvBoostParams cvBoostParams;

	int numOfFeaturesPerFrame;			//number of features we use in classification
	vector<int> indicesOfFeaturesToUse;

	void initClassifier();

public:
	AdaBoostClassifier();
	AdaBoostClassifier(const string & dataCSVFileName, const string & featuresTagsFilename, const vector<int> & indicesOfFeaturesToUse);
	~AdaBoostClassifier(){}

	void setParams(DistensionClassifierParams distClassPars){ ; }
	void runTraining();													//when we want to recalibrate from the scratch	
	void calcAndShowTrainingResults();
	int readClassifierFromFile(const string & filename);				//when we just want to read previous adaboost calibration result //TODO not working correctly
	float predictSample(FeaturesCluster & featuresCluster, vector<float> & confidenceVector);
};

#endif
