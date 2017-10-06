#include "AdaBoostClassifier.h"

void AdaBoostClassifier::initClassifier(){
	//set parameters of classifier
	cvBoostParams = CvBoostParams(CvBoost::REAL, numOfFeaturesPerFrame, 0, 1, false, 0);
	cout << "Classifier intiatied with N features per frame. N = " << numOfFeaturesPerFrame << endl;
}

AdaBoostClassifier::AdaBoostClassifier(){
	this->dataCSVFileName = string("tuning_output.txt");
	numOfFeaturesPerFrame = 5;								//Leave this number low so that it does not crash at any time (better use other constructor)!
	indicesOfFeaturesToUse = vector<int>{0, 1, 2, 3, 4};
	initFeatureDataFromFile();
	readTagsFromFile("distensionFeaturesTags.txt");
	initClassifier();
}

AdaBoostClassifier::AdaBoostClassifier(const string & dataCSVFileName, const string & featuresTagsFilename, const vector<int> & indicesOfFeaturesToUse){
	this->dataCSVFileName = dataCSVFileName;
	numOfFeaturesPerFrame = (int)indicesOfFeaturesToUse.size();  //(int)indicesOfFeaturesToUse.size();
	this->indicesOfFeaturesToUse = indicesOfFeaturesToUse;
	initFeatureDataFromFile();
	readTagsFromFile(featuresTagsFilename);
	initClassifier();
}

void AdaBoostClassifier::runTraining(){
	cout << "AdaBoost Training started" << endl;
	classifierStatus = CS_NOTREADY;

	//1. The training process (for info: http://opencv.itseez.com/2.4/modules/ml/doc/boosting.html?highlight=adaboost#cvboost-train) and http://robertour.com/2012/01/24/adaboost-on-opencv-2-3/
	//Train the classifier
	boost.train(&cvml, cvBoostParams, false);

	//2. Test and train errors:
	// 1. Declare a couple of vectors to save the predictions of each sample
	std::vector<float> train_responses, test_responses;
	// 2. Calculate the training error
	float fl1 = boost.calc_error(&cvml, CV_TRAIN_ERROR, &train_responses);
	// 3. Calculate the test error
	float fl2 = boost.calc_error(&cvml, CV_TEST_ERROR, &test_responses);

	//for clarification what we use for actual classification
	predictedResponsesAll = train_responses;

	//print results which are specific to this classifier
	cout << "Number of all frames: " << realResponsesAll.size() << endl;
	cout << "Number of train frames: " << train_responses.size() << endl;
	cout << "Number of test frames: " << test_responses.size() << endl;
	cout << "Error train:	" << fl1 << string("%") << endl;
	cout << "Error test:	" << fl2 << string("%") << endl;

	//6. save the classifier into a file
	string classifierFilename = string("./out_trained_distension_boost.xml");
	std::remove(classifierFilename.c_str());
	boost.save(classifierFilename.c_str(), "boost");
	cout << "Classifier saved to file: " << classifierFilename << endl;

	//specify that we finished training and classifier is ready to go
	classifierStatus = CS_TRAINED;
}

void AdaBoostClassifier::calcAndShowTrainingResults(){

	//Right now only show confusion matrix:
	vector<vector<double> > confusionMatrix;
	int confMatStatus = calcConfusionMatrix(realResponsesAll, predictedResponsesAll, numOfClasses, confusionMatrix);

	if (confMatStatus == 0){
		printConfusionMatrix(confusionMatrix, numOfClasses);
	}
	else{
		cout << "Calculation of confusion matrix was impossible." << endl;
	}
}

int AdaBoostClassifier::readClassifierFromFile(const string & filename){
	classifierStatus = CS_NOTREADY;

	cout << "Trying to read Adaboost classifier settings from file: " << filename << endl;
	
	if (!ifFileExists(filename)){
		cout << string("The specified file does not exist: ") + filename << endl;
		char c;
		cin >> c;
		throw invalid_argument(string("The specified file does not exist: ") + filename);
		return -1;
	}

	boost.load(filename.c_str());
	cout << "Reading classifier finished successfully." << endl;

	classifierStatus = CS_READFROMFILE;
	return 0;
}

float AdaBoostClassifier::predictSample(FeaturesCluster & featuresCluster, vector<float> & confidenceVector){
	//prepare output
	confidenceVector.clear();
	confidenceVector.push_back(-1.0);
	confidenceVector.push_back(-1.0);	//no confidence data available

	//get features 
	vector<double> featureVectorDbl;
	featuresCluster.getFeaturesVals(featureVectorDbl);
	//convert to float
	std::vector<float> sampleFeatVec(featureVectorDbl.begin(), featureVectorDbl.end());

	//check if classifier is ready
	if (classifierStatus == CS_NOTREADY){
		cout << "The classifier is not prepared. Train the classifier before or read the classifier data from file prior to predict()" << endl;
		throw invalid_argument(string("The classifier is not prepared. Train the classifier before or read the classifier data from file prior to predict()"));
	}

	//check if size of data is correct!
	if (sampleFeatVec.size() != numOfFeaturesPerFrame){
		cout << "Ada Predict: The sample vector has wrong size! It is different that the ones used for training." << endl;
		throw invalid_argument(string("Ada Predict: The sample vector has wrong size! It is different that the ones used for training."));
	}

	//Be careful!!!
	//It is a strange thing. Opencv consideres the number of features in a wrong way, namely it considers it to be of size equal to: (vector size we used to train + 1), and we have to artificially improve our vector by putting 0 as first feature
	vector<float> vecPlusOne{ 0 };
	vecPlusOne.insert(vecPlusOne.end(), sampleFeatVec.begin(), sampleFeatVec.end());

	//convert to float mat
	cv::Mat sampleMat;
	Mat tempMat(vecPlusOne);
	tempMat.convertTo(sampleMat, CV_32FC1);

	float label = boost.predict(sampleMat);
	return label;
}