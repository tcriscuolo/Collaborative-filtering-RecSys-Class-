#include "../include/Model.h"


// Construct a Validade object
Validate::Validate(Data &D) {
	this->D = D;	
}


void Validate::predict(Baseline &model) {
	vector<Tuple> samples = this->D.get_samples();

	cout << "UserId:ItemId,Prediction"<< endl;
	for (auto& samp : samples) {

		double pred = model.predict(samp.user, samp.item);

	//	pred = this->to_valid_range(pred);

		cout << this->D.get_user_hash(samp.user) << ":" << this->D.get_item_hash(samp.item)
						<< "," << pred << endl;
	}
}

void Validate::predict(UserBasedCF &model) {
	vector<Tuple> samples = this->D.get_samples();

	cout << "UserId:ItemId,Prediction"<< endl;
	for (auto& samp : samples) {

		double pred = model.predict(samp.user, samp.item);

		//pred = this->to_valid_range(pred);

		cout << this->D.get_user_hash(samp.user) << ":" << this->D.get_item_hash(samp.item)
						<< "," << pred  << endl;
	}
}

void Validate::predict(ItemBasedCF &model) {
	vector<Tuple> samples = this->D.get_samples();

	cout << "UserId:ItemId,Prediction"<< endl;
	for (auto& samp : samples) {
		double pred = model.predict(samp.user, samp.item);
	
		//pred = this->to_valid_range(pred);

		cout << this->D.get_user_hash(samp.user) << ":" << this->D.get_item_hash(samp.item)
						<< "," << pred << endl;
	}

	//cout << (int)samples.size() << endl;
}

void Validate::predict(Svd &model) {
	vector<Tuple> samples = this->D.get_samples();

	cout << "UserId:ItemId,Prediction"<< endl;
	for (auto& samp : samples) {
		
		double pred = model.predict(samp.user, samp.item);
		
		//pred = this->to_valid_range(pred);
		cout << this->D.get_user_hash(samp.user) << ":" << this->D.get_item_hash(samp.item)
						<< "," << pred << endl;
	}

	//cout << (int)samples.size() << endl;
}


void Validate::predict(MetaModel &model) {
	vector<Tuple> samples = this->D.get_samples();

	cout << "UserId:ItemId,Prediction"<< endl;
	for (auto& samp : samples) {
		
		double pred = model.predict(samp.user, samp.item);
		
		pred = this->to_valid_range(pred);
		cout << this->D.get_user_hash(samp.user) << ":" << this->D.get_item_hash(samp.item)
						<< "," << pred << endl;
	}

	//cout << (int)samples.size() << endl;
}





double Validate::to_valid_range(const double &p){
	double r = p;
	r = (r > 10.0 ? 10 : r);
	r = (r < 0.0 ? 0 : r);

	return r;
}
