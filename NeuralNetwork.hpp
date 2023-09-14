#include<iostream>
#include<vector>
#include <fstream>
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>  
#include <math.h>
using namespace std;

template <typename S>
ostream& operator<<(ostream& os,
                    const vector<S>& vector)
{
    // Printing all the elements
    // using <<
    for (auto element : vector) {
        os << element << " ";
    }
    os << endl;
    return os;
}

class NeuralNetwork{
    float learning_rate;
    vector<vector<float>> inputs,outputs; // learning set
    // first is an input layer, last layer is an output one 
    vector<vector<float>> nodes,biases; 
    vector<vector<vector<float>>> weights;

    vector<int> topology; 
    //topology vector assigned values:
    //[0] - inputs number
    //[1] - layers number
    //[2] - nodes per layer number
    //[3] - outputs number
    public:
    NeuralNetwork(float lr, vector<vector<float>> inputs,vector<vector<float>> outputs, vector<int> topology);

    void train();

    vector<float> predict(vector<float> in);

    void initialize_weights_and_biases();

    float random() { return ((float) rand()) / ((float) RAND_MAX);}

    //activation function
    float sigmoid(float x) { return 1 / (1 + exp(-x)); }

    //derivative of activation function
    float dSigmoid(float x) { return x * (1 - x); }

    
};
vector<int> shuffle(vector<int> array, int n)
{
    if (n > 1) 
    {
        int i;
        for (i = 0; i < n - 1; i++) 
        {
          int j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
    return array;
}

NeuralNetwork::NeuralNetwork(float lr, vector<vector<float>> inputs,
                            vector<vector<float>> outputs, vector<int> topology) :
                            learning_rate(lr),inputs(inputs), outputs(outputs),
                            topology(topology) 
{
    initialize_weights_and_biases(); 
}

void NeuralNetwork::train() {
    int epochs = 1000;
    // Iterate through the entire training for a number of epochs
    for (int n=0; n < epochs; n++) {
        // As per SGD, shuffle the order of the training set
        vector<int> trainingSetOrder;
        for(int i=0; i<inputs.size() ; i++) {trainingSetOrder.push_back(i);}
        vector<int> c = shuffle(trainingSetOrder,inputs.size());
        trainingSetOrder.clear();
        for(int i : c)
        {
            trainingSetOrder.push_back(i);
        }

        // Cycle through each of the training set elements
        for (int x=0; x<inputs.size(); x++) {
            int i = trainingSetOrder[x];
            // Compute hidden layers activation
            for(int j=0; j<topology[1] ; j++) // layer
            {
                vector<float> v;
                for (int k=0; k<topology[2]; k++) // node
                {
                    float activation=biases[j][k];
                    int z = (j==0 ? topology[0] : topology[2]);
                    for (int l=0; l<z; l++) //weight
                    {
                        activation+= (j==0 ? inputs[i][l] : nodes[j-1][l])*weights[j][k][l];
                    }
                    v.push_back(sigmoid(activation));
                }
                nodes.push_back(v);
            }

            // Compute output layer activation
            vector<float> d;
            for (int j=0; j<topology[3]; j++)
            {
                float activation=biases[topology[1]][j];
                for (int k=0; k<topology[2]; k++) 
                {
                    activation+=nodes[topology[1]-1][k]*weights[topology[1]][j][k];
                }
                d.push_back(sigmoid(activation));
            }
            nodes.push_back(d);
            // Compute change in output weights
            vector<float> deltaOutput;
            for (int j=0; j<topology[3]; j++) {
                float dError = (outputs[i][j]-nodes[topology[1]][j]);
                deltaOutput.push_back(dError*dSigmoid(nodes[topology[1]][j]));
            }
            // Compute change in hidden weights
            vector<vector<float>> deltaHidden;
            for(int j=topology[1]-1; j>=0 ; j--)
            {
                vector<float> v;
                for (int k=0; k<topology[2]; k++) 
                {
                    
                    float dError = 0.0f;
                    int z = (j==topology[1]-1 ? topology[3] : topology[2]); // if last layer then num of outputs else num of nodes in layer
                    for(int l=0; l<z; l++) {
                        
                        dError+= (j==topology[1]-1 ? deltaOutput[l]*nodes[topology[1]][l] : deltaHidden[0][l]*nodes[topology[1]-j-1][l]);
                    }
                    v.push_back(dError*dSigmoid(nodes[j][k]));
                }
            deltaHidden.push_back(v);
            }
            // Apply change in output weights
            for (int j=0; j<topology[3]; j++) {
                biases[topology[1]][j] += deltaOutput[j]*learning_rate;
                for (int k=0; k<topology[2]; k++) {
                    weights[topology[1]][j][k]+=nodes[topology[1]][k]*deltaOutput[j]*learning_rate;
                }
            }
            // Apply change in hidden weights
            for(int j=0; j<topology[1]; j++)
            {
                for (int k=0; k<topology[2]; k++) {
                    biases[j][k] += deltaHidden[j][k]*learning_rate;
                    int z = (j==0 ? topology[0] : topology[2]); // if first layer then inputs else nodes number
                    for(int l=0; l<z; l++) {
                        weights[j][k][l]+= (j==0 ? inputs[i][l]*deltaHidden[j][k]*learning_rate : nodes[j-1][l]*deltaHidden[j][k]);
                    }
                }
            }
            
        }
    }
}

void NeuralNetwork::initialize_weights_and_biases() {
    for(int i=0 ; i<topology[1] + 1 ; i++) {
        vector<float> v;
        vector<vector<float>> u;
        int z = (i==topology[1] ? 3 : 2); // if last layer, then number of nodes is as in output layer
        for(int j=0 ; j<topology[z]; j++)
        {
            vector<float> s;
            int w = (i==0 ? 0 : 2); // if first layer, number of weights for each node is inputs number, else it's number of nodes in a layer
            for(int k=0; k<topology[w]; k++)
            {
                s.push_back(random());
            }
            v.push_back(random());
            u.push_back(s);
        }
        biases.push_back(v);
        weights.push_back(u);
    }
}

vector<float> NeuralNetwork::predict(vector<float> in) {
    vector<vector<float>> nodes_predictions;
    for(int i=0; i<topology[1]+1 ; i++)
    {
        vector<float>v;
        int z = (i==topology[1] ? 3 : 2);
        for(int j=0; j<topology[z] ; j++) 
        {
            float sum = biases[i][j];
            int w = (i==0 ? 0 : 2);
            for(int k=0; k<topology[w] ; k++)
            {
                sum += (i==0 ? in[k] : nodes_predictions[i-1][k])*weights[i][j][k];
            }
            v.push_back(sum);
        }
        nodes_predictions.push_back(v);
    }
    return nodes_predictions[topology[1]];
}
