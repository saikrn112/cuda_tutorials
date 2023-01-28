#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <iostream>
#include <vector>

struct is_it_one
{
__host__ __device__ bool operator() (int x) { return x == 1; }
};

struct transpose{
  const int _n;
  transpose(int n):_n(n){};

  __host__ __device__
  int operator()( const int& y) const
  {
    int tmp = y;
    int i = tmp/_n;
    int j = tmp%_n;
    return j*_n + i;
  }
};



struct diagCheck
{

  private:
    int _a,_b;
  public:
    diagCheck(int a, int b): _a(a),_b(b){}
    __host__ __device__
    int operator()(int x,int y){
        if((x-_a)==(y-_b)||(x-_a)==(_b-y))
          return 1;
        else
          return 0;
    }
};



int main(void) {
  std::ifstream myfile;
  int n;
   /* File Operations */
  myfile.open("myfile.txt"); // opening the file
  if (myfile.is_open()) {
    //getline(myfile,line); n = std::stoi(line); // First element of the file : Number of workshops
    myfile >> n ;
    // cout << n <<" " <<  m << endl;;
  } else {
    std::cout << "file couldn't open" << std::endl;
    return 1;
  }


  thrust::device_vector<int> hMat;

  int tmp;
  for(int i=0; i<n*n; i++){
    myfile >> tmp;
    hMat.push_back(tmp);
  }

  std::vector<int> pairs;

  thrust::device_vector<int>::iterator iter1;
  thrust::device_vector<int>::iterator iter2;
  thrust::device_vector<int> coordX,coordY;
  int d;
  // for checking rowwise
  for(int i=0; i<n; i++){
    int result  = thrust::reduce(hMat.begin()+i*n, hMat.begin()+(i+1)*n,0,thrust::plus<int>()  );
    if(result >1){
      std::cout << "NO" << std::endl;
      iter1    = hMat.begin();
      iter2    = thrust::find_if(hMat.begin() + i*n, hMat.begin() + (i+1)*n, is_it_one());
      d = thrust::distance(iter1, iter2);
      pairs.push_back(d);
      thrust::device_vector<int>::iterator iter3    = thrust::find_if(hMat.begin() + d + 1, hMat.begin() + (i+1)*n, is_it_one());
      d = thrust::distance(iter1, iter3);
      pairs.push_back(d);
      for(int i=0; i<pairs.size(); i++) std::cout << "("<< pairs[i]%n<<","<< pairs[i]/n <<")"<< std::endl;
      return 0;
    } else{
  		iter1=hMat.begin()+i*n;
  		iter2=thrust::find_if(hMat.begin()+i*n, hMat.begin()+i*n+n,is_it_one());
  		d = thrust::distance(iter1, iter2);
  		coordX.push_back(i);
      coordY.push_back(d);
    }
  }

  // for checking columnwise
  thrust::device_vector<int> indices(n*n);
  thrust::sequence(indices.begin(), indices.end());
  thrust::device_vector<int> tindices(n*n);

  thrust::transform(indices.begin(),indices.end(), tindices.begin(), transpose(n));
  thrust::device_vector<int> thMat = hMat;

  thrust::sort_by_key(tindices.begin(), tindices.begin() + n*n, thMat.begin());

  for(int i=0; i<n; i++){
    int result  = thrust::reduce(thMat.begin()+i*n, thMat.begin()+(i+1)*n,0,thrust::plus<int>()  );
    if(result >1){
      std::cout << "NO" << std::endl;
      thrust::device_vector<int>::iterator iter1    = thMat.begin();
      thrust::device_vector<int>::iterator iter2    = thrust::find_if(thMat.begin() + i*n, thMat.begin() + (i+1)*n, is_it_one());
      int d = thrust::distance(iter1, iter2);
      pairs.push_back(d);
      thrust::device_vector<int>::iterator iter3    = thrust::find_if(thMat.begin() + d + 1, thMat.begin() + (i+1)*n, is_it_one());
      d = thrust::distance(iter1, iter3);
      pairs.push_back(d);
      for(int i=0; i<pairs.size(); i++) std::cout << "("<< pairs[i]%n<<","<< pairs[i]/n <<")"<< std::endl;
      return 0;
    }
  }


  //Check diagonal elements
  thrust::device_vector<int> out;
  // thrust::fill
  for(int i=0; i<n; i++)
      out.push_back(0);


  for(int i=0;i<n;i++){
    int a=*(coordX.begin()+i),b=*(coordY.begin()+i);
    thrust::transform(coordX.begin()+i+1,coordX.end(),coordY.begin()+i+1,out.begin(),diagCheck(a,b));
    if(thrust::reduce(out.begin(), out.end(), (int) 0, thrust::plus<int>())!=0){
      std::cout<<"NO"<<std::endl;
      std::cout<<coordX[i]<<" "<<coordY[i]<<std::endl;
      iter1=out.begin();
      iter2=thrust::find(out.begin(), out.end(),(int) 1);
      d = thrust::distance(iter1, iter2);
      std::cout<<coordX[i+d+1]<<" "<<coordY[i+d+1]<<std::endl;
      return 0;
    }
    else
      // thrust::fill(out_vec.begin(),N,0);
      for(int i=0; i<n; i++)
        out.push_back(0);
  }


  std::cout << "YES" <<std::endl;
  return 0;
}
