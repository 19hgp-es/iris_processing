#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

double atof(string str){
	return atof(str.c_str());
}

void split_double(string str, vector<double> &v){
	string temp, token; double d;
	stringstream ss;
	for(int i = 0 ; i < str.size() ; i++){
		if(str[i] == '[' or str[i] == ']')
			continue;
		temp += str[i];
	}
	
	ss.str(temp);
	while(ss >> token){
		d = atof(token);
		if(d != 0)
			v.push_back(d);
	}
}

string get_name(string str){
	string name;
	stringstream ss(str);
	ss >> name;
	if(isalpha(name[0]))
		return name;
	else
		return "NONE";
}

void print(vector<double> v, string name){
	for(int i = 0 ; i < v.size() ; i++){
		cout << name << " "<< i << " : " << v[i] << endl;
	}
	cout << endl;
}


int main(){
	string name, buf;
	vector<double> weight1_0, bias1_0, weight1_1, bias1_1, 
					weight2_0, bias2_0, weight2_1, bias2_1,
					weight3_0, bias3_0, weight3_1, bias3_1,
					weight4_0, bias4_0, weight4_1, bias4_1,
					weight5_0, bias5_0, weight5_1, bias5_1,
					weight_gap_0, bias_gap_0,
					weight_gap_1, bias_gap_1;
	vector<double> *pointer;
	fstream fin("a.txt");
	while(!fin.eof()){
		getline(fin, buf);
		name = get_name(buf);
		if(name != "NONE"){
			if(name == "layer1.0.weight")
				pointer = &weight1_0;
			if(name == "layer1.0.bias")
				pointer = &bias1_0;
			if(name == "layer1.1.weight")
				pointer = &weight1_1;
			if(name == "layer1.1.bias")
				pointer = &bias1_1;
			
			if(name == "layer2.0.weight")
				pointer = &weight2_0;
			if(name == "layer2.0.bias")
				pointer = &bias2_0;
			if(name == "layer2.1.weight")
				pointer = &weight2_1;
			if(name == "layer2.1.bias")
				pointer = &bias2_1;
			
			if(name == "layer3.0.weight")
				pointer = &weight3_0;
			if(name == "layer3.0.bias")
				pointer = &bias3_0;
			if(name == "layer3.1.weight")
				pointer = &weight3_1;
			if(name == "layer3.1.bias")
				pointer = &bias3_1;
			
			if(name == "layer4.0.weight")
				pointer = &weight4_0;
			if(name == "layer4.0.bias")
				pointer = &bias4_0;
			if(name == "layer4.1.weight")
				pointer = &weight4_1;
			if(name == "layer4.1.bias")
				pointer = &bias4_1;

			if(name == "layer5.0.weight")
				pointer = &weight5_0;
			if(name == "layer5.0.bias")
				pointer = &bias5_0;
			if(name == "layer5.1.weight")
				pointer = &weight5_1;
			if(name == "layer5.1.bias")
				pointer = &bias5_1;

			if(name == "gap.0.weight")
				pointer = &weight_gap_0;
			if(name == "gap.0.bias")
				pointer = &bias_gap_0;
			if(name == "gap.1.weight")
				pointer = &weight_gap_1;
			if(name == "gap.1.bias")
				pointer = &bias_gap_1;
		}
		split_double(buf, *pointer);
	}

	print(weight1_0, "weight1_0");
	print(bias1_0, "bias1_0");
	print(weight1_1, "weight1_1");
	print(bias1_1, "bias1_1");

	print(weight2_0, "weight2_0");
	print(bias2_0, "bias2_0");
	print(weight2_1, "weight2_1");
	print(bias2_1, "bias2_1");
	
	print(weight3_0, "weight3_0");
	print(bias3_0, "bias3_0");
	print(weight3_1, "weight3_1");
	print(bias3_1, "bias3_1");
	
	print(weight4_1, "weight4_1");
	print(bias4_1, "bias4_1");
	print(weight4_1, "weight4_1");
	print(bias4_1, "bias4_1");
	
	print(weight5_1, "weight5_1");
	print(bias5_1, "bias5_1");
	print(weight5_1, "weight5_1");
	print(bias5_1, "bias5_1");

	print(weight_gap_0, "gap.0.weight");
	print(bias_gap_0, "gap.0.bias");
	print(weight_gap_1, "gap.1.weight");
	print(bias_gap_1, "gap.1.bias");
	
}	