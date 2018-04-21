#pragma once
#include <vector>
#include <string>
#include <windows.h>
#include <fstream>
#include <iostream>
#include <utility>
#include <chrono>

using namespace std;

void read_directory(const std::string& name, vector<string>& v);

string StringPadding(string original, size_t charCount);

pair<int*, int*> loadData(const string dirName, const string fileName, unsigned int &n, unsigned int &W, unsigned int &expectedResult);

void generateRandomDataFile(const string fileName, unsigned int n, unsigned int W);