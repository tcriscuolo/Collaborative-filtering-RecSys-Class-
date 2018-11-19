#ifndef THREAD_H_INCLUDED
#define THREAD_H_INCLUDED
#include <thread>
#include <mutex>
#include <vector>

#define NUM_THREAD 4

// Auxliaries for multi thread processing
using namespace std;

typedef struct Bound{
	Bound(int a, int b) {
		this->start = a;
		this->end = b;
	}

	int start;
	int end;
} Bound;

// Create a partition for threads to work on
static vector<Bound> bounds(const int &num_thread, const int &total_job){
	int block_size = total_job/num_thread;

	vector<Bound> res;
	int last_start = 0;
	res.push_back(Bound(0, block_size));
	int i = 1;

	while(i < num_thread - 1) {
		int last_end = res[i - 1].end;
		res.push_back(Bound(last_end, last_end + block_size));
		i++;
	}

	int last_end = res[i - 1].end;
	res.push_back(Bound(last_end, total_job));

	return res;
}

#endif 
