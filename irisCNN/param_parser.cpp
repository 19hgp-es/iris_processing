#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>
#include <map>

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
	string name, buf, map_id;
	vector<string> map_id_vector;
	map<string, vector<double> > M;
	fstream fin("param_old.txt");
	while(!fin.eof()){
		getline(fin, buf);
		name = get_name(buf);

		if(name != "NONE" and M.find(name) == M.end()){
			vector<double> v;
			M[name] = v;
			map_id = name;
			map_id_vector.push_back(map_id);
		}
		split_double(buf, M[map_id]);
	}

	for(int i = 0 ; i < map_id_vector.size() ; i++){
		// cout << map_id_vector[i] << endl;
		print(M[map_id_vector[i]], map_id_vector[i]);
	}
}	
