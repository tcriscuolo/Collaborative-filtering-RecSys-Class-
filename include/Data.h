#ifndef DATA_H_INCLUDED
#define DATA_H_INCLUDED
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <utility>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <set>
#include <random>
#include <unordered_map>
using namespace std;

class Tuple {
	public:
		// Initialize a train tuple
		void initialize(const std::string &u,const std::string &i, 
										const double &r, const int &t) {
			this->user = stoi(u.substr(1, u.length() ));
			this->item = stoi(i.substr(1, i.length() ));
			this->rating = r;
			this->timestamp = t;
			this->is_train = true;
		}

		// Initialize a test tuple
		void initialize(std::string u, std::string i) {
			this->user = stoi(u.substr(1, u.length()));
			this->item = stoi(i.substr(1, i.length()));
			this->is_train = false;
		}
		bool is_train;
		int user;
		int item;
		double rating;
		int timestamp;
};


// Contains information about training and test data
class Data {
	public:
	Data();

	int get_num_of_user();
	int get_num_of_item();

	std::vector<Tuple>& get_samples(); 

	string get_user_hash(const int &);
	string get_item_hash(const int &);

	int get_num_of_sample();
	Tuple& get_sample(const int&);

	vector<int>& get_users();
	vector<int>& get_items(); 

	void read_train(std::string);
	void read_test(std::string);
							
				
	private:
		int num_of_user;
		int num_of_item;
		int is_train;	
		// Store a matrix as vector because it is safer
		// value of each cell store a tuple (rating, timestamp)
		//std::vector<std::vector<pair<double, double> > > M;
		std::vector<Tuple> data;
		std::vector<int> users_id;
		std::vector<int> items_id;
		
		unordered_map<int, int> u_map;
		unordered_map<int, int> i_map;
		unordered_map<int, string> user_hash;
		unordered_map<int, string> item_hash;
};

#endif
