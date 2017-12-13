#include "../include/Matrix.h"

Matrix::Matrix() {
}
// Initialize matrix with user/item rating with timestamp
Matrix::Matrix(Data &D) {
	this->num_of_user = D.get_num_of_user();
	this->num_of_item = D.get_num_of_item();
	this->num_of_sample = D.get_num_of_sample();
	this->users_id = D.get_users();
	this->items_id = D.get_items();

	//initialize_matrix();
	string spc = " ";
	for(int idx = 0; idx< this->num_of_sample; idx++) {
		Tuple s = D.get_sample(idx);
		int user = s.user;
		int item = s.item;

		U[user][item] = make_pair(s.rating, s.timestamp);
		I[item][user] = make_pair(s.rating, s.timestamp);	
	}

	this->compute_global_mean();
	this->compute_global_sdv();
	// store all user mean
	for(auto u: users_id) {	
		this->compute_user_mean(u);
		this->compute_user_sdv(u);
		//cout << u << " " << this->user_mean[u] << " " << this->user_sdv[u] << endl;
	}
	for(auto i: items_id) {
		this->compute_item_mean(i);
		this->compute_item_sdv(i);

	}
	//print();
}

int Matrix::get_num_user() {
	return ((int)this->U.size());
} 

int Matrix::get_num_item() {
	return ((int)this->I.size());
}

/////////////////////////////////////////////
int Matrix::get_num_items_rated(const int &u) {
	return ((int) this->U[u].size());
}

int Matrix::get_num_user_rated(const int &i) {
	return ((int) this->I[i].size());
}

double Matrix::get_rating(const int &u, const int &i) {
	return (U[u][i].first);
}

//////////////////////////////////////////////
void Matrix::set_rating(const int &u, const int &i, const double &v) {
	this->U[u][i].first = v;
	this->I[i][u].first = v;
}

void Matrix::set_tstamp(const int &u, const int &i, const int &t) {
	this->U[u][i].second = t;
	this->I[i][u].second = t;
}

//////////////////////////////////////////////
double Matrix::get_user_mean(const int &u) {
	return (this->user_mean[u]);
}
double Matrix::get_user_sdv(const int &u) {
	return (this->user_sdv[u]);
}

double Matrix::get_global_mean() {
	return (this->global_mean);
}

double Matrix::get_global_sdv() {
	return (this->global_sdv);
}

double Matrix::get_item_mean(const int &i) {
	return (this->item_mean[i]);
}

double Matrix::get_item_sdv(const int &i) {
	return (this->item_sdv[i]);
}
///////////////////////////////////////////////
int Matrix::get_tstamp(const int &u, const int &i) {
	return (U[u][i].second);
}

//////////////////////////////////////////////// 
double Matrix::get_sparsity() {
	int n = this->num_of_user;
	int m = this->num_of_item;
	int k = this->num_of_sample;
	return (1 - double(k)/double(n*m));
}

// Print all user/item rating and timestamp
void Matrix::print() {
	for(auto& u: U) {
		for(auto& i: u.second) {
			cout << u.first << " "<< i.first << " " << 
							i.second.first << " " << i.second.second << endl;
		}
	}
}

void Matrix::print_users() {
	for(auto u: users_id) {
		cout << u << " " << user_mean[u] << " " << item_sdv[u] <<  endl;
	}
	int total = this->get_num_user();
	cout << "Total: " <<  total << " users" << endl;
}
// return all items rated by the given user
item_set& Matrix::get_items_rated_by(const int &u) {
	return U[u];
}

// return all user who rated the given item
user_set& Matrix::get_who_rated(const int &i) {
	return (I[i]);
}

void Matrix::compute_user_mean(const int &u) {
	int n = U[u].size();
	double sum = 0;
	
	// for all item rated by user u it add its value
	for(auto &i: U[u]) {
		sum += i.second.first;	
	}
	// divide by the total number of items rated
	double mean = sum/(1.0*n);
	this->user_mean[u] = mean;
}

void Matrix::compute_item_mean(const int &i) {
	int n = (int)I[i].size();
	double sum = 0;

	for(auto &j: I[i]) {
		sum += j.second.first;
	}

	double mean = sum/(1.0*n);
	this->item_mean[i] = mean;
}

void Matrix::compute_global_mean() {
	int k = this->num_of_sample;
	
	double sum = 0;
	for(auto& u: U) {
			for(auto& i: u.second) {
				sum += i.second.first; 
			}
	}
	double mean = (1.0/(1.0*k))*sum;
	this->global_mean = mean;
	//cout << this->global_mean << endl;
}

void Matrix::compute_global_sdv() {
	int k = this->num_of_sample;
	
	double sum = 0;
	double g_mean = this->global_mean;
	
	for(auto& u: U) {
			for(auto& i: u.second) {
				double dif = i.second.first - g_mean;
				sum += pow(dif, 2);
			}
	}

	double sdv = sqrt((1.0)/(1.0*k) * sum);
	this->global_sdv = sdv;
	//cout << this->global_sdv;
}

// compute user scoring standard deviation
// and store it on variable user_sdv
void Matrix::compute_user_sdv(const int &u) {

	double sum = 0;
	double u_mean = this->user_mean[u];
	item_set R = this->get_items_rated_by(u);  
	int n = R.size();
	
	// Check if user has rated only one item
	if(n <= 1) {
		this->user_sdv[u] = 0;
		return;
	}

	for(auto& i: R) {
		double dif = (i.second.first - u_mean);
		sum += pow(dif, 2);
	}

	double mean = sum*(1.0/(1.0*n));
	double sdv = sqrt(mean);
	this->user_sdv[u] = sdv;
}

void Matrix::compute_item_sdv(const int &i) {
	
	double sum = 0;
	double i_mean = this->item_mean[i];
	item_set I = this->get_who_rated(i);
	int n = (int)I.size();

	if( n<= 1) {
		this->item_sdv[i] = 0;
		return;
	}

	for(auto& v: I) {
		double dif = (v.second.first - i_mean);
		sum += pow(dif, 2);
	}
	double mean = sum*(1.0/(1.0*n));
	double sdv = sqrt(mean);
	this->item_sdv[i] = sdv;
}

// remove all user user who has less than
// a threshold number of ratings or zero standard deviation
void Matrix::filter_user_cold_start(const int &thres) {
	set<int> to_remove;
	double eps = 1e-18;
	for(auto u : this->users_id) {
		if(this->get_num_items_rated(u) <= thres || this->get_user_sdv(u) < eps)  {
			to_remove.insert(u);
		}
	}
	
	// remove all direct information of user u
	for(auto u : to_remove) {
	item_set u_items = this->get_items_rated_by(u);

		// remove all ratings given by user u
		for(auto i : u_items) {
			I[i.first].erase(u);
			if((int)I[i.first].size() == 0) {
				I.erase(i.first);

				items_id.erase(find(items_id.begin(), 
														items_id.end(), i.first));
			}
		}
								
		if(user_mean.find(u) != user_mean.end()) 
			user_mean.erase(u);
		if(user_sdv.find(u) != user_sdv.end())
			user_sdv.erase(u);
		if(U.find(u) != U.end())
			U.erase(u);
		if(find(users_id.begin(), users_id.end(), u) != users_id.end()) 
			users_id.erase(find(users_id.begin(), 
													users_id.end(), u));

	}


	// Update the following variables
	// item mean
	// item sdv
	this->compute_global_mean();
	this->compute_global_sdv();

	for(auto i: items_id) {
		this->compute_item_mean(i);
		this->compute_item_sdv(i);	
	}	
}


bool Matrix::find_user(const int &u) {
	if(this->U.find(u) != this->U.end()) return true;

	return false;
}

bool Matrix::find_item(const int &i) {
	if(this->I.find(i) != this->I.end()) return true;

	return false;
}

vector<int>& Matrix::get_users() {
	return (this->users_id);
}

vector<int>& Matrix::get_items() {
	return (this->items_id);
}


