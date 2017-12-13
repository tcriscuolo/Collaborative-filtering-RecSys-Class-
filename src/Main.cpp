#include <iostream>
#include <string>
#include "../include/Data.h"
#include "../include/Matrix.h"
#include "../include/Model.h"

using namespace std;

int main(int argc, char *argv[]) {
	// check if number of arguments is correct
	if(argc != 3) {
		cout << "Wrong number of arguments" << endl;
		return 0;
	}

	string train_path = string(argv[1]);
	string test_path 	=	string(argv[2]);

	Data train;
	train.read_train(train_path);
	
	Data test;
	test.read_test(test_path);
	
	// If you use the userbased or itembased
	// it is strongly recommended to
	// pre process the test input

	//	Baseline model = Baseline(train);
	//ItemBasedCF model = ItemBasedCF(train);
	//	UserBasedCF model = UserBasedCF(train);
	//model.pre_process(test);
	//	Svd model = Svd(train);

	MetaModel model = MetaModel(train, test);

  Validate val = Validate(test);
	val.predict(model);
	return 0;
}
