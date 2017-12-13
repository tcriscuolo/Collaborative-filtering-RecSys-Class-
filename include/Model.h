#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED
#include "Thread.h"
#include "Data.h"
#include "Matrix.h"

using namespace std;

// Auxiliary structures to store user/item similarity
typedef struct UserSim {
	int user;
	double similarity;
	UserSim(int a, double b) {
		this->user = a;
		this->similarity = b;
	}
	bool operator<(const UserSim &b) const {
		return this->similarity > b.similarity;
	}
} UserSim; 

typedef struct ItemSim {
	int item;
	double similarity;
	ItemSim(int a, double b) {
		this->item = a;
		this->similarity = b;
	} 
	bool operator<(const ItemSim &b) const {
		return (this->similarity > b.similarity);
	}
}ItemSim;

// Baseline implemented from the following survey
// http://files.grouplens.org/papers/FnT%20CF%20Recsys%20Survey.pdf
class Baseline {
	public:
		Baseline(Data &D);
	
		void train(); 
		double predict(const int &, const int &);
		static void item_bias(Baseline *, const vector<int> &, const int &, const int &);
		static void user_bias(Baseline *, const vector<int> &, const int &, const int &);
		
		double rate;		
		double cu;
		double ci;
		
		unordered_map<int, double> u_base;
		unordered_map<int, double> i_base;
		Matrix M;
		Data D;
};


// Builds a user based model using Pearson's similarity
class UserBasedCF {
	public:
		UserBasedCF(Data &);
	  ~UserBasedCF();	
		void train();

		void pre_process(Data &);
		double predict(const int &u, const int &i);

		void build_pearson_neighbourhood(const int &u);
		static void build_pearson_neighbourhood(UserBasedCF *, vector<int> &, 
																						const int&, const int&);
		static void pre_build_pearson_neighbourhood(UserBasedCF *, vector<int> &, 
																							const int&, const int&);
		void compute_candidates();
		int max_search;
		int max_neighbour;
		int cold_start;
		int max_cand;
		int min_rating;
		int max_rating;
		double sim_thres;		
		
		vector<int> candidates;
		unordered_map<int, vector<UserSim> > neigh;
		Baseline *B;
		Data raw_data;
		Matrix *M;
};

class ItemBasedCF {
	public: 
	ItemBasedCF(Data &D);
	~ItemBasedCF();

	void pre_process(Data&);
	int max_neighbour;
	int max_search;
	int cold_start;

	void remove_bias();
	void compute_candidates();
	void pre_compute_norm();
	void build_item_nei(int& );

	double predict(const int&,const int&);
	static void build_cos_neigh(ItemBasedCF *, const vector<int> &, const vector<int> &,
															const int &, const int &);
	
	static void compute_items_norm(ItemBasedCF *, const vector<int> &, 
																const int &, const int&);

	unordered_map<int, vector<ItemSim> > sim; 
	unordered_map<int, long double> item_norm;
	vector<int> candidates;
	
	double min_sim;
	int min_user;
	int max_user;
	
	Baseline *B;
	Baseline *H;
	Data raw_data;
	Matrix *M;
};

// Implementation of Singular Value Decomposition (svd) 
// with stochastic gradient descendent
class Svd {
	public:
		Svd(Data &D, const int& = 3, const int& = 200, const  double& = 1e-3, const double& = 8e-2);
		~Svd();
		void add_item_noise();
		void add_user_noise();		
  	//static void helper(Svd *, const vector<int> &, const int &, const int&);
		void regurlarize();
		void initialize_items();
		void initialize_users();
		void initialize_features(); 
		void find_optmal_ssgd();
		void find_optmal_sgd();
		double predict();
		void print_users();
		void print_items();

		double predict(const int &, const int &);
		long double dot_product(const vector<long double> &, const vector<long double> &);
		Data D; 
		Baseline *B;
		Matrix *M;
		int k;						 															// number of latent factors
		int m; 						   														// number of user
		int n;							  													// number of item
		int max_iteration;
		long double l_rate;
		long double lambda;
		unordered_map<int, vector<long double> > p;   	// user;
		unordered_map<int, vector<long double> > q;     // item;
		
		static void regularize(Svd *,const vector<int> &, const int &, const int&);

};

class MetaModel {
		public:
		MetaModel(Data &D, Data &H);
						
		static void train_svd(MetaModel *, Data&, const vector<int>& ,
													const vector<int>& , const vector<int>& ,
													const int& , const int& );

		static void helper(const int &);	
		double predict(const int &u, const int &i);

		Svd *svd;
		UserBasedCF *user_based;
		ItemBasedCF *item_based;
		Baseline *baseline;
};

// Predict all sample on the given test 
class Validate {
	public:
	Validate();
	Validate(Data&);
	Validate(Baseline&);	

	void predict(Baseline &);
	void predict(UserBasedCF &);
	void predict(ItemBasedCF &);
	void predict(Svd &);
	void predict(MetaModel &);
	private:
	double to_valid_range(const double &);
	Data D;
};
 
#endif
