#include "../include/Model.h"

mutex item_based_barrier;

ItemBasedCF::ItemBasedCF(Data &D) {
	this->max_neighbour = 50;
	this->min_sim = 1e-4;
	this->min_user = 3;
	this->max_user = 700;
	this->B = new Baseline(D);
	this->M = new Matrix(D);

	this->remove_bias();
	
	this->compute_candidates();

	this->pre_compute_norm();
	//vector<int>& items = this->M->get_items();

	//for(auto& item: this->M->get_items()) {
	//		int count = this->M->get_num_user_rated(item);
	//		if(count >= this->min_user && count <= this->max_user) {
	//			candidates.push_back(item);
	//		}
	//}
	

//	this->compute_norm();
//	vector<Bound> bnd1 = bounds(NUM_THREAD, (int)items.size());
//	vector<thread> t1; 
//	for(int i = 0; i < NUM_THREAD; i++) {
//		t1.push_back(thread(compute_items_norm, this, ref(items), ref(bnd1[i].start), ref(bnd1[i].end)));
//	}

//	for(auto& t: t1) {
//		t.join();
//	}

} 

ItemBasedCF::~ItemBasedCF() {
	delete this->B;
	delete this->M;
}


void ItemBasedCF::pre_process(Data& H) {
	vector<int> it = H.get_items();
	vector<Bound> bnd2 = bounds(NUM_THREAD, (int)it.size());
	vector<thread> t2;
	for(int i = 0; i < NUM_THREAD; i++) {
		t2.push_back(thread(build_cos_neigh, this, ref(it), ref(candidates), ref(bnd2[i].start), 
														ref(bnd2[i].end)));
	}

	for(auto& t: t2) {
		t.join();
	}
}

void ItemBasedCF::remove_bias() {
	for(auto& user: M->get_users()) {
		for(auto& i: M->get_items_rated_by(user)) {
			int item = i.first;
			double rec = M->get_rating(user, item) - B->i_base[item];
			M->set_rating(user, item, rec);
		}			
	}
}


void ItemBasedCF::compute_candidates(){

	vector<int>& items = this->M->get_items();
	for(auto& item: items) {
			int count = this->M->get_num_user_rated(item);
			if(count >= this->min_user && count <= this->max_user) {
				this->candidates.push_back(item);
			}
	}
}

void ItemBasedCF::pre_compute_norm() {

	vector<int>& items = this->M->get_items();
	vector<Bound> bnd1 = bounds(NUM_THREAD, (int)items.size());
	vector<thread> t1; 
	
	for(int i = 0; i < NUM_THREAD; i++) {
		t1.push_back(thread(compute_items_norm, this, ref(items), ref(bnd1[i].start), ref(bnd1[i].end)));
	}

	for(auto& t: t1) {
		t.join();
	}
}



void ItemBasedCF::compute_items_norm(ItemBasedCF *instance, const vector<int>  &items,
								const int &start, const int &end) {

	for(int k = start; k < end; k++) {
		int item = items[k];
		user_set Ik = instance->M->get_who_rated(item);
		double sum = 0;

		for(const auto& u: Ik) {
			 int user = u.first;
			// can take user mean off
			sum += pow(instance->M->get_rating(user, item), 2);
		}
		
		double norm = sqrt(sum);
		item_based_barrier.lock();
		instance->item_norm[item] = norm;
		item_based_barrier.unlock();
	}		
}

void ItemBasedCF::build_cos_neigh(ItemBasedCF *instance, const vector<int> &items,
																	const vector<int> &candidates, const int &start, const int &end) {

	for(int k = start; k < end; k++) {
		int item = items[k];
		user_set& Iu = instance->M->get_who_rated(item);
		vector<ItemSim> res; 

		for(auto& v: candidates) {
			if(item == v) continue;
			user_set& Iv = instance->M->get_who_rated(v);
			long double sum = 0.0;	
			for(const auto &user: Iv) {
				if(Iu.find(user.first) != Iu.end() && Iv.find(user.first) != Iv.end()) {
					long double rating_u = instance->M->get_rating(user.first, item);
					long double rating_v = instance->M->get_rating(user.first, v);
					sum += (rating_u * rating_v);
				}
			}
			
			long double sim = sum/(instance->item_norm[item] * instance->item_norm[v]);
			if(isnan(sim) ) {
				continue;
			}
			if(sim >= instance->min_sim) res.push_back(ItemSim(v, sim));
		}

		sort(res.begin(), res.end());
		item_based_barrier.lock();
		instance->sim[item] = res;
		item_based_barrier.unlock();
	}
}


void ItemBasedCF::build_item_nei(int& item) {
	user_set& Iu = this->M->get_who_rated(item);
	vector<ItemSim> res; 

	for(auto& v: candidates) {
		if(item == v) continue;
		user_set& Iv = this->M->get_who_rated(v);
		long double sum = 0.0;	
		for(const auto &user: Iv) {
			if(Iu.find(user.first) != Iu.end() && Iv.find(user.first) != Iv.end()) {
				long double rating_u = this->M->get_rating(user.first, item);
				long double rating_v = this->M->get_rating(user.first, v);
				sum += (rating_u * rating_v);
			}
		}
			
		long double sim = sum/(this->item_norm[item] * this->item_norm[v]);
		if(isnan(sim) ) {
			continue;
		}
		if(sim >= this->min_sim) res.push_back(ItemSim(v, sim));
	}

	sort(res.begin(), res.end());
	this->sim[item] = res;
}


double ItemBasedCF::predict(const int &u,const int &i) {
	if(!M->find_user(u) || !M->find_item(i)) {
		return this->B->predict(u, i);
	}

	// if the item neighbourhood has 
	// not been pre-processed it is built 
	// on demand
	if(sim.find(i) == sim.end()) {
		int item = i;
		this->build_item_nei(item);
	}

	vector<ItemSim>& q = this->sim[i];

	long double sum = 0.0;
	long double sum_sim = 0.0;
	double i_base = this->B->i_base[i];

	item_set& rated = M->get_items_rated_by(u);
	int added = 0;

	for(int count = 0; count < (int)q.size(); count++) {
		ItemSim tmp = q[count];
		int j = tmp.item;
		long double sim_uj = tmp.similarity;

	//	cout << u <<  " " << i << ": " << sim_uj << endl;
		if(rated.find(j) != rated.end()) {
			sum += (sim_uj * (this->M->get_rating(u, j) - this->B->i_base[j]));
			sum_sim += abs(sim_uj);
			added++;
		}
		if(added > this->max_neighbour) break;
	}
	if(added < 3) return B->predict(u, i);
	long double res = (sum/sum_sim) + i_base;
	
	if(isnan(res)) return this->B->predict(u, i);
	
	//res = min((double)res, 10.0);
	//res = max((double)res, 0.0);
	return res;
}
