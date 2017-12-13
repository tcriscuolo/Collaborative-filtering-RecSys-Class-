#include "../include/Model.h"

Svd::Svd(Data &D, const int& latent_factor, const int& max_it, const double& l_rate, const double &lambda) {
	this->B = new Baseline(D);
	this->D = D;
	this->M = new Matrix(D);
	this->k = latent_factor;								// number of factors
	this->l_rate = l_rate;									// learning rate
	this->lambda = lambda;									// regularization value
	this->max_iteration = max_it;						// number of iteration
	this->m = this->M->get_num_user();
	this->n = this->M->get_num_item();
		
	this->regurlarize();
	this->initialize_features();
	this->find_optmal_ssgd();
	//this->print_users();
}

Svd::~Svd() {
	delete this->B;
	delete this->M;
}

void Svd::initialize_features() {	
	// Initialize p_ij
	for(int user : M->get_users()) {
		if(M->find_user(user)) {
			for(int j = 0; j < this->k; j++) {
				p[user].push_back(0.1);
			}
		}
	}

	// Initialize q_ij
	for(int item : M->get_items()) {
		if(M->find_item(item)) {
			for(int j = 0; j < this->k; j++) {
				q[item].push_back(0.1);
			}
		}
	}
}

long double Svd::dot_product(const vector<long double> &u,const vector<long double> &v) {
	long double res = 0.0;
	int k = (int)u.size();
	for(int i = 0; i < this->k; i++) {
		res += (u[i] * v[i]);
	}
	return res;
}

void Svd::add_user_noise() {
	// for each pi and ui it add some noise to it
	double seed = 1001;
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<long double> dis(-1, 1);
	for(int user: this->M->get_users()) {
		for(int f = 0; f < this->k; f++) {
			long double noise = 2 * this->l_rate * dis(generator);
			this->p[user][f] += noise; 
		}
	}
}

void Svd::add_item_noise() {

	double seed = 1001;
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<long double> dis(-1, 1);
	for(int item: this->M->get_items()) {
		for(int f =0; f < this->k; f++) {
			long double noise = 2* this->l_rate * dis(generator);
			this->q[item][f] += noise;
		}
	}
}

// Stochastic gradient descendent with sampling
void Svd::find_optmal_ssgd() {

	unsigned int it = 0;	
	int num_samp = this->D.get_num_of_sample();
	unsigned int max_it = (this->max_iteration * num_samp);
	vector<Tuple>& samples = this->D.get_samples();
  int k = (int)samples.size();
	srand(1001);

	do {
		//int N = 0;
		//double sum_dui = 0.0;
		unsigned int idx = (rand() % k);
		Tuple t = samples[idx];
		int user = t.user;
		int item = t.item;


		long double rui = (long double)this->M->get_rating(user, item);
		long double residual = rui - this->dot_product(p[user], q[item]);
			//N++;

		for(int i = 0; i < this->k; i++) {
			q[item][i] += (l_rate * ((residual * p[user][i]) - (lambda * q[item][i])));
			p[user][i] += (l_rate * ((residual * q[item][i]) - (lambda * p[user][i]))); 
				
		}
		/*
		Computing error is too expensive
		sum_dui += pow((rui - B->predict(user,item) - dot_product(p[user], q[item])), 2);
		sum_dui += lambda*(B->u_base[user] + B->i_base[item] +
										dot_product(p[user], p[user]) +
										dot_product(q[item], q[item]));	
				
		 */
		
		//long double error = sqrt(sum_dui/(1.0*N));
		//if(error > last_error) break;
		//cout << error << endl;
		//last_error = error;
		it++;
	} while(it < max_it); 
	

}

void Svd::find_optmal_sgd() {

	//long	double last_error = 1e9;
	//long double sum_dui;
	//long double delta_error;
	int it = 0;	
	do {
		//int N = 0;
		//double sum_dui = 0.0;
		for(Tuple& t: D.get_samples()) {
			int user = t.user;
			int item = t.item;

			long double rui = (long double)this->M->get_rating(user, item);
			long double residual = rui - this->dot_product(p[user], q[item]);
			//N++;

			for(int i = 0; i < this->k; i++) {
				q[item][i] += (l_rate * ((residual * p[user][i]) - (lambda * q[item][i])));
				p[user][i] += (l_rate * ((residual * q[item][i]) - (lambda * p[user][i]))); 
				
			}
			/*
			Computing error is too expensive
			sum_dui += pow((rui - B->predict(user,item) - dot_product(p[user], q[item])), 2);
			sum_dui += lambda*(B->u_base[user] + B->i_base[item] +
											dot_product(p[user], p[user]) +
											dot_product(q[item], q[item]));	
				
			 */
		}
		
		//long double error = sqrt(sum_dui/(1.0*N));
		//if(error > last_error) break;
		//cout << error << endl;
		//last_error = error;
		it++;
	} while(it < this->max_iteration); 
	
}


double Svd::predict(const int &u, const int &i) {
	if((!M->find_user(u) || !M->find_item(i)))
					return B->predict(u, i);

	long double base = B->predict(u, i);

	long double latent = this->dot_product(p[u], q[i]);

	double res =  base + latent;
	//res = min(res, 10.0);
	//res = max(res, 0.0);
	return (double)res;	
}

void Svd::print_users() {
	for(int user: M->get_users()) {
		cout << "User: " << user << " ";
		for(int i = 0; i < k; i++) {
			cout << p[user][i] << " ";
		}
		cout << endl;
	}
}

void Svd::print_items() {
	for(int item : M->get_items()) {
		cout << "Item: " << item << " ";
		for(int i = 0; i < k; i++) {
			cout << q[item][i] << " ";
		}
		cout << endl;
	}
}

void Svd::regularize(Svd *instance, const vector<int> &users,
											const int &start,	const int &end) {				
	// I am a thread I work hard
	for(int k = start; k < end; k++) {
		int user = users[k];
		item_set& Iu = instance->M->get_items_rated_by(user);
		
		for(auto& i: Iu) {
			int item = i.first;
			double bias = instance->B->predict(user, item);
			double actual = instance->M->get_rating(user, item);
			double reg = actual - bias;
			instance->M->set_rating(user, item, reg);
		}
	}
}
//void Svd::helper(Svd *instance, const vector<int> &users, const int &start, const int &end) {
	//instance->reg(ref(users), ref(start), ref(end));
//}

void Svd::regurlarize() {
	//string tab = "\t";
	vector<int>& users = this->M->get_users();
	vector<Bound> bnd = bounds(NUM_THREAD, (int)users.size());
	vector<thread> threads; 
	// Parallel regularization 
	for(int i = 0; i < NUM_THREAD; i++) {
		threads.push_back(std::thread(regularize, this, ref(users), ref(bnd[i].start),
														ref(bnd[i].end)));
	}

	for(auto& t: threads) {
		t.join();
	}

	/*  
	for(int user : this->M->get_users()) {
		item_set Iu = M->get_items_rated_by(user);
		for(auto& i: Iu) {
			int item = i.first;
			double bias = this->B->predict(user, item);
				double actual = this->M->get_rating(user, item);
				double reg = actual - bias;
				this->M->set_rating(user, item, reg);
				//cout << user << tab << item << tab << actual << tab << reg << endl;
		}
	}

	*/
}

