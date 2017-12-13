#include "../include/Model.h"

// Baseline using 
mutex b_barrier;

Baseline::Baseline(Data &D) {
	this->D = D;
	this->M = Matrix(D);
	double g_mean = M.get_global_mean();
	this->cu = 5.0;
	this->ci = 2.0;
	// 25 10
	// melhor 10 / 5
	vector<int> use = M.get_users();
	vector<int> itm = M.get_items();


	vector<int> users(use.begin(), use.end());
	vector<int> items(itm.begin(), itm.end());

	random_shuffle(users.begin(), users.end());
	random_shuffle(items.begin(), items.end());

	vector<Bound> user_bnd = bounds(NUM_THREAD, (int)users.size());
	vector<Bound> item_bnd = bounds(NUM_THREAD, (int)items.size());

	vector<thread> t1;

	for(int t = 0; t < NUM_THREAD; t++) {
		t1.push_back(thread(user_bias, this,  ref(users), ref(user_bnd[t].start), ref(user_bnd[t].end)));	
	}

	for(auto& t : t1) {
		t.join();
	}
	t1.clear();
	vector<thread> t2;
	for(int t = 0; t < NUM_THREAD; t++) {
		t2.push_back(thread(item_bias, this, ref(items), ref(item_bnd[t].start), ref(item_bnd[t].end)));
	}

	for(int t = 0; t < NUM_THREAD; t++) {
		t2[t].join();
	}
	t2.clear();
	/*
	for(auto& u: M.get_users()) {
		int user = u;
		double bu = 0;
		int card = M.get_num_items_rated(user);
		for(auto& i: M.get_items_rated_by(user)) {
				double rating = i.second.first;
				bu += (rating - g_mean);
		}
		this->u_base[u] = (bu/(card + cu));
	}

	for(auto& i: M.get_items()) {
		int item = i;
		double bi = 0;
		int card = M.get_num_user_rated(item);
		for(auto& u: M.get_who_rated(item)) {

			int user = u.first;
			double bu = this->u_base[user];
			bi += (M.get_rating(user, item) - g_mean - bu);
		}

		this->i_base[i] = bi/(card + ci);
	}
	*/	
}

void Baseline::user_bias(Baseline *instance, const vector<int> &users,
																const int &start, const int &end) {
		double g_mean = instance->M.get_global_mean();
		for(int k = start; k < end; k++) {
			int user = users[k];
			double bu = 0;
			int card = instance->M.get_num_items_rated(user);
			for(auto &i: instance->M.get_items_rated_by(user)) {
				double rating = i.second.first;
				bu += (rating - g_mean);
			}
			b_barrier.lock();
			instance->u_base[user] = (bu/(card + instance->cu));
			b_barrier.unlock();
		}
}

void Baseline::item_bias(Baseline *instance, const vector<int> &items,
																const int &start, const int &end) {
		double g_mean = instance->M.get_global_mean();
		for(int k = start; k < end; k++) {
			int item = items[k];
			double bi = 0;
			int card = instance->M.get_num_user_rated(item);
			for(auto& u: instance->M.get_who_rated(item)) {
				int user = u.first;
				double bu = instance->u_base[user];
				bi += (instance->M.get_rating(user, item) - g_mean - bu);
			}
		  b_barrier.lock();
			instance->i_base[item] = bi/(card + instance->ci);
			b_barrier.unlock();
		}
}


void Baseline::train() {
}

double Baseline::predict(const int &u, const int &i) {
	
	double g_mean = this->M.get_global_mean();
	
	double bu = (M.find_user(u) ? u_base[u] : 0.0);
	double bi = (M.find_item(i) ? i_base[i]: 0.0);

	double  res = g_mean + bu + bi;
	
	res = min(res, 10.0);
	res = max(res, 0.0);
	return (res); 
}




