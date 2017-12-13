#include "../include/Matrix.h"

Data::Data() {

}

void Data::read_train(string input_path) {	
	string spc = " ";
	string nl = "nl";
	ifstream file(input_path);
	string user, item, prediction, timestamp, scratch;
	
	this->num_of_user = 0;
	this->num_of_item = 0;
	if(file.is_open()) {

		int line = 0;

		// reads the header line
		getline(file, scratch);

		while(getline(file, user , ':')) {
			getline(file, item, ',');
			getline(file, prediction, ',');
			getline(file, timestamp);

			//cout << "user: " << user << endl;
			//cout << "item: " << item << endl;
			//cout << "pred: " << prediction << endl;
			//cout << "time: " << timestamp << endl;		
			// for user i add item j to its list with rating and timestamp
			Tuple sample;
			
			// makes a pre-hash for usersid from 1 to number of users	
			sample.initialize(user, item, stof(prediction), stoi(timestamp));
		
			if(user_hash.find(sample.user) == user_hash.end()) users_id.push_back(sample.user);
			if(item_hash.find(sample.item) == item_hash.end()) items_id.push_back(sample.item);

			this->user_hash[sample.user] = user;
			this->item_hash[sample.item] = item;
			
			data.push_back(sample);

			//cout << "Line: " << line << endl;
			line++;
		}	
		this->is_train = true;
		file.close();
	} else {
		cout << "Could't open input file"<< endl;
	}
}

void Data::read_test(std::string input_path) {
	ifstream file;
	file.open(input_path);

	string scratch;
	string user;
	string item;
	if(file.is_open()) {
		if(file.is_open()) {		
			getline(file, scratch);

			while(getline(file, user, ':')) {
				getline(file, item);
				Tuple sample;
				sample.initialize(user, item);
				data.push_back(sample);
				
				if(user_hash.find(sample.user) == user_hash.end()) users_id.push_back(sample.user);
				if(item_hash.find(sample.item) == item_hash.end()) items_id.push_back(sample.item);
				
				this->user_hash[sample.user] = user;
				this->item_hash[sample.item] = item;
				//cout << sample.user << " " << sample.item << endl;
			}
		}
		this->is_train = false;
	} else {
		cout << "Could't open test file" << endl;
	}

}


int Data::get_num_of_user() {
	return (this->users_id.size());
}

int Data::get_num_of_item() {
	return (this->items_id.size());
}

int Data::get_num_of_sample() {
	return (this->data.size());
}

// Returns cell at the given idx
Tuple& Data::get_sample(const int &idx) {
	return (this->data[idx]);
}

vector<int>& Data::get_users() {
	return (this->users_id);
}

vector<int>& Data::get_items() {
	return (this->items_id);
}

string Data::get_user_hash(const int &u) {
	return (this->user_hash[u]);
}

string Data::get_item_hash(const int &i) {
	return (this->item_hash[i]);
}

// Returns all tuples
std::vector<Tuple>& Data::get_samples() {
	return (this->data);
}







