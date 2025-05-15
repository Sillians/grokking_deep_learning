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












































