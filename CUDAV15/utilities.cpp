#include "utilities.h"

void read_directory(const std::string& name, vector<string>& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			if (strcmp(data.cFileName, ".") != 0 && strcmp(data.cFileName, "..") != 0)
				v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

string StringPadding(string original, size_t charCount)
{
	original.resize(charCount, ' ');
	return original;
}

pair<int*, int*> loadData(const string dirName, const string fileName, unsigned int &n, unsigned int &W, unsigned int &expectedResult) {
	ifstream dataFile(dirName + "\\" + fileName), resultFile(dirName + "_optimum\\" + fileName);

	if (dataFile.is_open() && resultFile.is_open()) {
		dataFile >> n >> W;
		resultFile >> expectedResult;
	}
	else {
		std::cout << "B³¹d otwarcia pliku " << fileName << endl;
	}

	int *values = new int[n];
	int *weights = new int[n];

	for (int i = 0; i < n; i++) {
		dataFile >> values[i] >> weights[i];
	}

	dataFile.close();
	resultFile.close();
	return pair<int*, int*>(values, weights);
}

void generateRandomDataFile(const string fileName, unsigned int n, unsigned int W) {
	int min = 50;
	int max = 5000;
	ofstream outputFile(fileName);

	outputFile << n << " " << W << endl;
	for (int i = 0; i < n; ++i) {
		outputFile << min + (rand() % static_cast<int>(max - min + 1))
			<< " " << min + (rand() % static_cast<int>(max - min + 1))
			<< endl;
	}

	outputFile.close();
}