#ifndef MATRIX_H_INCLUDED
#define MATRIX_H_INCLUDED
#include <algorithm>
#include "Data.h"

// define two macros for user-item
// and item-user
#define EPSILON 1e-20
#define item_set unordered_map<int, pair<double, int> >
#define user_set unordered_map<int, pair<double, int> >

class Matrix{
	public:
	Matrix();
	Matrix(Data &D);
	
	// Getters
	int get_num_user();
	int get_num_item();

	double get_rating(const int &, const int &);

	int get_num_items_rated(const int &);
	int get_num_user_rated(const int &); 	

	double get_users_mean();
	double get_users_sdv();
	double get_items_mean();
	double get_items_sdv();

	double get_user_mean(const int &);
	double get_user_sdv(const int &);
	double get_item_mean(const int &);
	double get_item_sdv(const int &);
	double get_global_mean();
	double get_global_sdv();
	
	int get_tstamp(const int &, const int &);

	double get_user_norm(const int &);
	double get_item_norm(const int &);

	vector<int>& get_users();
	vector<int>& get_items(); 

	item_set& get_items_rated_by(const int &);
	user_set& get_who_rated(const int &);
	double get_sparsity();
	void print();
	void print_users();

	bool find_user(const int &);
	bool find_item(const int &);

	// Setters
	void set_rating(const int &, const int &, const double &);
	void set_tstamp(const int &, const int &, const int &);

	void filter_user_cold_start(const int &);
	private:
		void initialize_matrix();

		int num_of_user;
		int num_of_item;
		int num_of_sample;
		double global_mean; 
		double global_sdv;

		unordered_map<int, double> user_mean;
		unordered_map<int, double> user_sdv;
		unordered_map<int, double> item_mean;
		unordered_map<int, double> item_sdv;

		vector<int> users_id;
		vector<int> items_id;

		void compute_global_mean();
		void compute_global_sdv();
		void compute_user_mean(const int &);
		void compute_user_sdv(const int &);
		void compute_item_mean(const int &);
		void compute_item_sdv(const int &);
		void compute_user_norm(const int &);
		void compute_item_norm(const int &);

		unordered_map<int, item_set > U; 		// matrix in which user is a row and item is a column
		unordered_map<int, user_set > I; 	// matrix in which item is a row and user is a column 
		//vector<vector<pair<float, int> > > M; 
};

#endif
