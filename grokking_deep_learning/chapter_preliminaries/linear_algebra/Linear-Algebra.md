## Linear Algebra
Linear algebra is foundational to deep learning because it provides the mathematical framework 
for representing and manipulating data and model parameters. Vectors, matrices, and tensors—core 
structures in linear algebra—are used to express inputs, weights, activations, and transformations 
in neural networks. Operations like matrix multiplication power forward and backward passes, 
making linear algebra essential for both computation and optimization in deep learning models.



### Scalars
**scalars** are single numerical values used to represent 
magnitude or weights. They serve as the most basic building blocks—used to scale vectors and matrices, 
define loss values, learning rates, and perform element-wise operations. While deep learning models 
operate on high-dimensional tensors, every computation ultimately breaks down to operations involving scalars.




### Vectors
**vectors** are 1-dimensional arrays that represent quantities 
with both magnitude and direction. They are used to model data samples, feature representations, 
gradients, and weights within neural networks. Vectors enable operations like dot products, 
transformations, and projections, which are fundamental to tasks such as computing layer 
activations and optimizing loss functions during training.




### Matrices
**matrices** are 2-dimensional arrays used to represent structured data, 
transformations, and parameters in neural networks. They enable efficient computation of linear operations
such as matrix multiplication, which powers the forward and backward propagation of activations and 
gradients. Matrices are essential for representing layers, encoding weights between neurons, 
and performing batch operations on multiple data samples simultaneously.





### Tensors

**Tensors** are multi-dimensional arrays that generalize scalars, vectors, and matrices, serving as 
the foundational data structure in deep learning. They represent everything from inputs and 
weights to intermediate activations across layers. Their shape and dimensionality allow deep 
learning models to process complex, high-dimensional data efficiently, enabling parallelized 
operations during training and inference.


Tensors will become more important when we start working with images. Each image arrives as a 
3rd-order tensor with axes corresponding to the height, width, and channel. 
At each spatial location, the intensities of each color (red, green, and blue) are 
stacked along the channel.





### Reduction
**Reduction** refers to the process of collapsing one or more dimensions of a tensor by applying 
operations like summation, mean, or maximum. It’s commonly used in deep learning to aggregate 
information—for example, computing the loss across a batch, summing gradients, or reducing outputs 
to scalar values for evaluation. Reduction operations help simplify high-dimensional data while 
preserving essential information needed for learning and optimization.





### Dot Products
**Dot products** are fundamental operations that measure the similarity or alignment between two vectors. 
In deep learning, they are used to compute weighted sums in neurons, project inputs onto learned directions, 
and calculate similarities between feature representations. Dot products enable transformations within 
layers, and are key to operations in fully connected networks, attention mechanisms, and gradient calculations during backpropagation.





### Matrix–Vector Products
**Matrix–vector products** apply a linear transformation to a vector by multiplying it with a matrix, 
producing a new vector. In deep learning, this operation is central to computing the output of a neural
layer, where the matrix represents learnable weights and the vector is the input or activation from 
the previous layer. It enables efficient mapping of input features to higher-level representations, 
forming the core of forward propagation in neural networks.





### Matrix–Matrix Multiplication
**Matrix–matrix multiplication** combines two matrices to produce a new matrix that represents the 
composition of linear transformations. In deep learning, this operation is essential for processing 
entire batches of input data simultaneously, where one matrix contains the inputs and the other 
contains the model's weights. It enables efficient computation of activations across layers and 
is heavily optimized for performance on GPUs during both training and inference.





### Norms
**Norms** measure the magnitude or length of vectors or tensors and are used to quantify size, 
distance, or deviation in deep learning. They play a key role in regularization techniques 
like `L1` and `L2` penalties, helping prevent overfitting by constraining weight magnitudes. 
Norms are also used in optimization (e.g., gradient clipping) and in evaluating model behavior 
through metrics such as vector similarity or feature scaling.



















