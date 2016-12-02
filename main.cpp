#include <iostream>
#include <iterator>
#include <Array/Array.h>
#include <ReClaM/FFNet.h>
#include <ReClaM/createConnectionMatrix.h>
#include <ReClaM/CrossEntropy.h>
#include <ReClaM/Rprop.h>
#include <ReClaM/ClassificationError.h>

//Modified from the Shark library tutorial
int main()
{
  //Construct XOR problem input and target output
  Array<double> trainInput( 4,2);
  Array<double> trainTarget(4,1);
  for(int k=0, i=0; i!=2; ++i)
  {
    for(int j=0; j!=2; ++j)
    {
      trainInput(k,0) = i;
      trainInput(k,1) = j;
      trainTarget(k, 0) = (i+j) % 2;
      ++k;
    }
  }

  //Define neural net topology
  const int n_inputs = 2;
  const int n_hidden = 2;
  const int n_outputs = 1;
  //Create neural net connection matrix
  Array<int> connection_matrix;
  createConnectionMatrix(connection_matrix,n_inputs, n_hidden, n_outputs);

  //Display the connection matrix
  std::cout << "Display the connection matrix:\n";
  std::copy(connection_matrix.begin(),connection_matrix.end(),
    std::ostream_iterator<int>(std::cout," "));
  std::cout << '\n';

  //Create the feed-forward neural network
  FFNet net(n_inputs, n_outputs, connection_matrix);
  std::cout << "Display the neural network (note that there are no weights set yet):\n";
  net.write(std::cout);
  std::cout << '\n';

  std::cout << "Initializing the weights (uniformly <-0.1,0.1>)...\n";
  net.initWeights(-0.1, 0.1);

  std::cout << "Display the neural net:\n";
  net.write(std::cout);
  std::cout << '\n';

  //Error function
  CrossEntropy error;
  ClassificationError accuracy(.5);

  //Optimizer
  IRpropPlus optimizer;
  optimizer.init(net);

  //Training loop
  const int n_learning_cycles = 100;
  std::cout << "Start training for " << n_learning_cycles << " learning cycles.\n";
  for (int i = 0; i!= n_learning_cycles; ++i)
  {
    //Train the network
    optimizer.optimize(net, error, trainInput, trainTarget);

    //Show results
    std::cout << i << "\t"
        << accuracy.error(net, trainInput, trainTarget) << "\t"
        << error.error(net, trainInput, trainTarget) << std::endl;
  }

  std::cout << "Display the neural network after training:\n";
  net.write(std::cout);
  std::cout << '\n';
}
