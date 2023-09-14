#include "NeuralNetwork.hpp"
#include<iostream>

using namespace std;
int interpret_prediction(float p,float output);

int main() {
    vector<vector<float>> in = {{0,0},{0,1},{1,0},{1,1}};
    vector<vector<float>> out = {{0},{0},{0},{1}};
    float lr = 0.001;
    float c = 0;
    for(int i=0; i<100; i++)
    {
        NeuralNetwork n(lr,in,out,{(int)in[0].size(),3,5,(int)out[0].size()});
        n.train();

        int j = interpret_prediction(n.predict({1,1})[0],1);
        if(j==1)
        {
            c+=1.0;
        }
    }
    cout<<"Model accuracy: "<<c/100<<endl;
    


    return 0;
}

int interpret_prediction(float p,float output) {
    if(output==1)
    {
        return (fabs(p - 1) < fabs(p) ? 1 : 0);
    }
    if(output==0)
    {
        return (fabs(p - 1) > fabs(p) ? 1 : 0);
    }

    return -1;
}