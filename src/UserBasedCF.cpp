#include "../include/Model.h"

mutex user_based_barrier;

UserBasedCF::UserBasedCF(Data &D) {
	this->max_neighbour = 20;
	this->sim_thres =  2e-2;
	this->cold_start = 5;
 
	// A user has to have a minimum and a maximum number
	// of rating to be considered for a neighbourhood 
	this->min_rating = 3;
	this->max_rating = 700;
	
	this->raw_data = D;
	this->B = new Baseline(D);
	this->M = new Matrix(D);

	// filter users candidates
	this->compute_candidates();	
}


UserBasedCF::~UserBasedCF() {
	delete this->B;
	delete this->M;
}
/* Deprecated 
vector<UserSim>* UserBasedCF::build_cos_neighbourhood(const int &u, const int &i) {
	vector<UserSim> *res = new vector<UserSim>;
	
	item_set Iu = this->M->get_items_rated_by(u);
	user_set candidates = this->M->get_who_rated(i);

	int searched = 0;	
	for(auto& neigh: candidates) {
		int n = neigh.first;
		
		double sum = 0.0;
		double sum_left = 0.0;
		double sum_right = 0.0;
		
		item_set In = M->get_items_rated_by(n);
		
		if(n != u && In.find(i) != In.end()) 	{

			for(auto& tmp: Iu) {
				int item = tmp.first;

				if(In.find(item) != In.end()) { 		
					double left = M->get_rating(u, item) - M->get_user_mean(u);
					double right = M->get_rating(n, item) - M->get_user_mean(n);
					sum += (left * right);
					sum_left += pow(left, 2);
					sum_right += pow(right, 2);
				}
			}
			if(sum_left < EPSILON || sum_right < EPSILON) continue;
		
			double sim = sum/(sqrt(sum_left) * sqrt(sum_right));
			UserSim cand = UserSim(n, sim);
			res->push_back(cand);
			searched++;
		}
	//	if(searched > this->max_search) break;
	}
	
	sort(res->begin(), res->end());
	return res;
}
 */ 
void UserBasedCF::build_pearson_neighbourhood(UserBasedCF *instance, vector<int> &u_idx,
																const int &start, const int &end) {
	for(int j = start; j < end; j++) {
		int u = u_idx[j];
		if(instance->M->get_num_items_rated(u) < instance->cold_start) continue;
		vector<UserSim> q;

		double u_mean = instance->M->get_user_mean(u);
		item_set& Iu = instance->M->get_items_rated_by(u);

		for(int k = 0; k < instance->candidates.size(); k++) {
			int v = instance->candidates[k];
			double sim = 0;
			double v_mean = instance->M->get_user_mean(v);
		
			long double sum_num = 0.0;
			long double sum_u = 0.0;
			long double sum_v = 0.0;
			int intersection = 0;
			item_set Iv = instance->M->get_items_rated_by(v);

			for(auto& tmp: Iu) {
				int item = tmp.first;
			
				if(Iv.find(item) != Iv.end()) {
					intersection++;
					long double left = instance->M->get_rating(u, item) - u_mean;
					long double right = instance->M->get_rating(v, item) - v_mean;
					sum_num += (left * right);
					sum_u += pow(left, 2);
					sum_v += pow(right, 2);
				}
			}
			if(isnan(sqrt(sum_u) * sqrt(sum_v)) || intersection <= 3) continue;
			long double den = sqrt(sum_u) * sqrt(sum_v);
			sim = sum_num/den;
			UserSim tmp = UserSim(v, sim);
			if(sim >= instance->sim_thres) q.push_back(tmp);
		}
		sort(q.begin(), q.end());
		user_based_barrier.lock();
		instance->neigh[u] = q;
		user_based_barrier.unlock();
	}
}

void UserBasedCF::compute_candidates() {
	for(auto& u: this->M->get_users()) {
			int count = this->M->get_num_items_rated(u);
			if(count >= this->min_rating && count <= this->max_rating) 
				candidates.push_back(u);
		}	
}




void pearson_similarity(UserBasedCF *instance, const int& u, 
												const int& start, const int& end) {
	vector<int> candidates = instance->candidates;
	double u_mean = instance->M->get_user_mean(u);
	item_set& Iu = instance->M->get_items_rated_by(u);
	vector<UserSim> q;

	//double u_mean = M->get_user_mean(u); 
	for(int k = start; k < end; k++) {
		int v = instance->candidates[k];
		long double sim = 0;
		long double v_mean = instance->M->get_user_mean(v);
		
		long double	sum_num = 0.0;
		long double sum_u = 0.0;
		long double sum_v = 0.0;
		int intersection = 0;	
		item_set& Iv = instance->M->get_items_rated_by(v);
		
		for(auto& tmp: Iu) { 
			int item = tmp.first;
			if(Iv.find(item) != Iv.end()) {
				intersection++;
				long double left = instance->M->get_rating(u, item) - u_mean;
				long double right = instance->M->get_rating(v, item) - v_mean;
				sum_num += (left * right);
				sum_u += pow(left, 2);
				sum_v += pow(right, 2);
			}
		}

		if(isnan(sqrt(sum_u)*sqrt(sum_v)) || intersection < 2) continue;

		long double den = sqrt(sum_u) * sqrt(sum_v);
		sim = sum_num/den;

		if(isnan(sim)) continue;
		
		UserSim tmp = UserSim(v, sim);
		q.push_back(tmp);
	}
	user_based_barrier.lock();
	for(auto& c: q) {
		instance->neigh[u].push_back(c);
	}
	user_based_barrier.unlock();
}


void UserBasedCF::build_pearson_neighbourhood(const int &u) {
	vector<UserSim> q;
	
	vector<thread> t1;
	vector<Bound> bnd = bounds(NUM_THREAD, (int)candidates.size());

	for(int t = 0; t < NUM_THREAD; t++) {
		t1.push_back(thread(pearson_similarity, this, ref(u),
														ref(bnd[t].start), ref(bnd[t].end)));
	}

	for(auto& t: t1) {
		t.join();
	}
  /*
	double u_mean = M->get_user_mean(u);
	item_set& Iu = M->get_items_rated_by(u);

	//double u_mean = M->get_user_mean(u); 
	for(auto neigh: candidates) {
		int v = neigh;
		double sim = 0;
		double v_mean = M->get_user_mean(v);
		
		double	sum_num = 0.0;
		double sum_u = 0.0;
		double sum_v = 0.0;
		int intersection = 0;	
		item_set Iv = M->get_items_rated_by(v);
		
		for(auto& tmp: Iu) { 
			int item = tmp.first;
			if(Iv.find(item) != Iv.end()) {
				intersection++;
				double left = M->get_rating(u, item) - u_mean;
				double right = M->get_rating(v, item) - v_mean;
				sum_num += (left * right);
				sum_u += pow(left, 2);
				sum_v += pow(right, 2);
			}
		}

		if(isnan(sqrt(sum_u)*sqrt(sum_v)) || intersection < 2) continue;

		double den = sqrt(sum_u) * sqrt(sum_v);
		sim = sum_num/den;
		UserSim tmp = UserSim(v, sim);
		q.push_back(tmp);
	}
	sort(q.begin(), q.end());

	neigh[u] = q;
	*/
}

void UserBasedCF::pre_build_pearson_neighbourhood(UserBasedCF *instance, vector<int>& users,
																									const int& start, const int &end) {
	for(int k = start; k < end; k++) {
		int u= users[k];
		
		vector<UserSim> q;
		//check if user exists on traning data

		long double u_mean = instance->M->get_user_mean(u);
		item_set& Iu = instance->M->get_items_rated_by(u);


		//double u_mean = M->get_user_mean(u); 
		for(auto neigh: instance->candidates) {
			int v = neigh;
			if(v == u) continue;

			long double sim = 0;
			long double v_mean = instance->M->get_user_mean(v);
		
			long double	sum_num = 0.0;
			long double sum_u = 0.0;
			long double sum_v = 0.0;
			int intersection = 0;	
			item_set& Iv = instance->M->get_items_rated_by(v);
		
			for(auto& tmp: Iu) { 
				int item = tmp.first;
				if(Iv.find(item) != Iv.end()) {
					intersection++;
					long double left = instance->M->get_rating(u, item) - u_mean;
					long double right = instance->M->get_rating(v, item) - v_mean;
					sum_num += (left * right);
					sum_u += pow(left, 2);
					sum_v += pow(right, 2);
				}
			}

			if(isnan(sqrt(sum_u)*sqrt(sum_v)) || intersection <= 2) continue;

			long double den = sqrt(sum_u) * sqrt(sum_v);
			sim = sum_num/den;
			// just consider user who has more than
			// a threashold 
			if(isnan(sim) || sim < instance->sim_thres) continue;
			
			UserSim tmp = UserSim(v, sim);
			q.push_back(tmp);
		}
		sort(q.begin(), q.end());	

		user_based_barrier.lock();
		instance->neigh[u] = q;
		user_based_barrier.unlock();
	}
}

// Pre-process user neighbourhood form
// the given data (expected to be a test data)
void UserBasedCF::pre_process(Data& D) {
	vector<int> users; 
	
	for(auto& u: D.get_users()) {
		if(M->find_user(u) && M->get_num_items_rated(u) >= cold_start)
			users.push_back(u);
	}
	vector<Bound> bnd = bounds(NUM_THREAD, (int)users.size());
	vector<thread> tr;

	for(int t = 0; t < NUM_THREAD; t++ ) {
		tr.push_back(thread(pre_build_pearson_neighbourhood, this, ref(users),
														ref(bnd[t].start), ref(bnd[t].end)));

	}

	for(auto& t: tr) {
		t.join();
	}
}


double UserBasedCF::predict(const int &u, const int &i) {
	if(!M->find_user(u) || !M->find_item(i))  {
		return B->predict(u, i);
	}

	// Compute neighbourhood on demand
	if(neigh.find(u) == neigh.end()) {
		build_pearson_neighbourhood(u);
		sort(neigh[u].begin(), neigh[u].end());
	}

	vector<UserSim>& neig = this->neigh[u]; 
	
	//build_pearson_neighbourhood(u, i, ref(neig));
	user_set& Ui = this->M->get_who_rated(i);
	long double res = 0.0;
	long double sum = 0.0;
	long double sum_sim = 0.0;
	int intersection = 0;
	
	// Aggregate similarities to make a prediction
	for(int k = 0;  k < (int)neig.size(); k++) {
		UserSim tmp = neig[k];
		int n = tmp.user;

		if((n == u) || (Ui.find(n) == Ui.end())) continue;
		intersection++;
		long double sim = tmp.similarity;
			
		sum += (sim*((M->get_rating(n, i) - M->get_user_mean(n))/M->get_user_sdv(n)));
		sum_sim += abs(sim);	
	} 
	// check corner case where it does not have a similar neighbour
		
	res = M->get_user_mean(u) + M->get_user_sdv(u)*(sum/sum_sim);	
	
	if(isnan(res) || intersection <= 2) {
		return B->predict(u, i);
	}

	//res = min((double)res, 10.0);
	
	//res = max((double)res,	 0.0);
	return res;
}
