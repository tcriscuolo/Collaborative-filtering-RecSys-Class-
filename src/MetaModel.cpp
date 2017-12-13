#include "../include/Model.h"

/*
void train_svd(MetaModel *instance) {


}

void train_userbased(MetaModel *instance) {

}
*/

MetaModel::MetaModel(Data &D, Data &H) {
	int dim = 3;
	int num_it = 200;
	double l_rate = 1e-3;
	double lambda = 8e-2;


	thread t1([this, &D, &dim, &num_it, &l_rate, &lambda, &H]() {
									this->svd = new Svd(D, dim, num_it, l_rate, lambda);
									});

	thread t2([this, &D, &H] () {
									this->user_based = new UserBasedCF(D);
									this->user_based->pre_process(H);
							});

	thread t3([this, &D, &H] () {
								this->item_based = new ItemBasedCF(D);
								this->item_based->pre_process(H);
						});
					
	t1.join();
	t2.join();
	t3.join();
}


double MetaModel::predict(const int&u, const int &i) {
	//weight for each model
	double a = 0.40; 
	double b = 0.30; 
	double c = 0.30;
	double pred = a*svd->predict(u, i) + 
								b*item_based->predict(u, i) + 
								c*user_based->predict(u, i);

	pred = min(pred, 10.0);
	pred = max(pred, 0.0);
	return pred;
}



