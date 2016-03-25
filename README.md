# Neural Nets

  This is my main focus at the Recurse Center.

The classical neural networks are built upon the MNIST data set for digit recognition (http://yann.lecun.com/exdb/mnist/). 
These implementations rely on Michael Nielsen's book on neural networks and deep learning.

You will find a lot of different networks that utilize the same basis, but are optimized differently (L2, weights initialization, Nesterov Momentum, Momentum). See the performance differences in the graphs folder.

My initial goal is not necessarily to optimize for the best possible performance, but to understand neural networks in depth without using additional libraries, and to get a grasp of optimization methods.

The files have confusing names (I am sorry!) The idea behind them are as follows:
  
  **Network.py**
  
  >> Classical raw implementation of MNIST neural net
  See batches_graph.py in the graphs folder for graphing the performance with different batch sizes


  **Network1.py**
  
 >>Neural net adjusted by momentum. Accuracy goes up quickly to nearly 96 percent, slightly better than with the classical model. 
  Time is still about the same: 15 seconds each epoch.
  Run by unit_test1.py, play with accuracy by adjusting the learning rate and epochs.

  **Network2.py**
  
  >>Neural net adjusted by Nesterov momentum. Accuracy goes up quickly to nearly 96 percent, same as with classical momentum.
  Time is still about the same: 15 seconds each epoch. 
  There is also learning rate adoption with step decay, resulting in no significant differences in performance.
  Run by unit_test2.py
  See eta_graph2.py for graphing performance with different learning rates.


  **Network3.py**
  
  >>Neural network adjusted by L2 regulaizaton against overfitting -> weight decay. See L2_graphs for visualizing different values for regularization
  
  
Each implementation has a funky draw() function that shows you the ascii drawing of the digit. To see this uncomment the draw() function in the validation.
