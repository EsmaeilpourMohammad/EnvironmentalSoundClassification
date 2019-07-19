/*
This is an implementation for Spherical K-means++ algorithm written in C++
Full Credit to Reshmi at https://github.com/ReshmiSuresh/Spherical-k-means-algorithm
*/

#define LL long long

using namespace std;

int tnum = 8732;
int numIter = 2400;
const int TOTAL_FILES = tnum, K=15, ITERATIONS = numIter;
struct feature{
	vector <double > unitVector;
};
double euclideanDistance_my(vector <double> &a, vector <double> &b)
{
	double ans = 0.0;
	for(int i=0; i<a.size(); i++)
		ans += (a[i]-b[i])*(a[i]-b[i]);
	return ans;
}
double similarity(feature a, vector <double > &centre)
{
	double ans = 0;
	for(int i=0; i<a.unitVector.size(); i++){
 		ans += a.unitVector[i]*centre[i];
 		ans += 0.02; // a simple regularizer for UrbanSound8K.
 		/*
 		Please set to 0.012 this to 0.021, 0.011, 0.31 respectively for ESC-10, ESC-50, and DCASE-2017
 		*/ 
	}
	return ans;
}
vector <vector <double > > pickFarthestPointCentres(vector <feature> &features, int numCentres, int wei)
{
	int n = features.size();
	vector <vector < double > > centres;
	vector < double > dist(n), cumul(n);
	centres.push_back(features[rand()%n].unitVector);
	for(int i=0; i<numCentres-1; i++) // Iterating over the centre number to be assigned (-1)
	{
		int idx = -1;
		double mx = -1;
		dist.assign(n, (double)(1LL<<28));
		for(int j=0; j<n; j++) // Iterating over each feature vector
		{
			for(int k=0; k<centres.size(); k++) //iterating over centres selected so far
				dist[j] = min(dist[j], euclideanDistance_my(features[j].unitVector, centres[k]));
			if(dist[j] > mx)
			{
				mx = dist[j];
				idx = j;
			}
		}
		
		centres.push_back(features[idx].unitVector);
	}
	return centres;
}
void KMeans_pp(vector <feature> &features, vector <vector <double> > &centres, vector<int> &assignedCluster, int numCentres)
{
	bool change = true;
	int assignedIndex, iterations=0;
	double maxSimilarity, currentSimilarity, normalizationFactor;
	while(change && iterations++ < ITERATIONS)
	{
		int count=0, dimensions = features[0].unitVector.size();
		vector <double > coordinate(dimensions, 0);
		change = false;
		for(int i=0; i<features.size(); i++)
		{
			maxSimilarity = -(1<<40);
			assignedIndex = assignedCluster[i];
			for(int j=0; j<centres.size(); j++)
			{
				currentSimilarity = similarity(features[i], centres[j]);
				if(currentSimilarity - maxSimilarity >= 1e-5)
				{
					assignedCluster[i] = j;
					maxSimilarity = currentSimilarity;
				}
			}
			if(assignedCluster[i] != assignedIndex)
				change = true;
		}
		for(int i=0; i<numCentres; i++)
		{
			count = 0; normalizationFactor = 0;
			coordinate.assign(dimensions, 0);
			for(int j=0; j<features.size(); j++)
			{
				if(assignedCluster[j] == i)
				{
					count++;
					for(int k=0; k<dimensions; k++)
						coordinate[k] += features[j].unitVector[k];
				}
			}
			for(int j=0; j<dimensions; j++)
			{
				coordinate[j] /= count;
				normalizationFactor += (coordinate[j]*coordinate[j]);
			}
			normalizationFactor = sqrt(normalizationFactor);
			for(int j=0; j<dimensions; j++)
				coordinate[j] /= normalizationFactor;
			centres[i] = coordinate;
		}
	}
}