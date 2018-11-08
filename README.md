# Handwritten-letters-Identifier-with-Module-based-Automatic-Differentiation-Neural-Net-from-Sketch
Label images of handwritten letters by implementing a Module Based AD Neural Network with one hidden layer and arbitrary hidden nodes from scratch.

# Dataset
We will be using a subset of an Optical Character Recognition (OCR) dataset. This data includes
images of all 26 handwritten letters; our subset will include only the letters “a,” “e,” “g,” “i,” “l,” “n,” “o,”
“r,” “t,” and “u.” The handout contains three datasets drawn from this data: a small dataset with 60 samples
per class (50 for training and 10 for test), a medium dataset with 600 samples per class (500 for training and
100 for test), and a large dataset with 1000 samples per class (900 for training and 100 for test). Figure 2.1
shows a random sample of 10 images of few letters from the dataset.

# File Format 
Each dataset (small, medium, and large) consists of two csv files—train and test. Each row
contains 129 columns separated by commas. The first column contains the label and columns 2 to 129
represent the pixel values of a 16  8 image in a row major format. Label 0 corresponds to “a,” 1 to “e,” 2
to “g,” 3 to “i,” 4 to “l,” 5 to “n,” 6 to “o,” 7 to “r,” 8 to “t,” and 9 to “u.” Because the original images are
black-and-white (not grayscale), the pixel values are either 0 or 1. However, you should write your code
to accept arbitrary pixel values in the range [0,1]. The images in Figure 2.1 were produced by converting
these pixel values into .png files for visualization. Observe that no feature engineering has been done here;
instead the neural network you build will learn features appropriate for the task of character recognition.

# Module-based Method of Implementation
Algorithm Forward Computation  
1: procedure NNFORWARD(Training example (x, y), Parameters alpha, beta)  
2: a = LINEARFORWARD(x;alpha)  
3: z = SIGMOIDFORWARD(a)  
4: b = LINEARFORWARD(z; beta)  
5: ^y = SOFTMAXFORWARD(b)  
6: J = CROSSENTROPYFORWARD(y; ^y)  
7: o = object(x; a; z; b; ^y; J)  
8: return intermediate quantities o  

# Algorithm Backpropagation
1: procedure NNBACKWARD(Training example (x, y), Parameters Alpha, Beta, Intermediates o)  
2: Place intermediate quantities x; a; z; b; ^y; J in o in scope  
3: gJ = dJ/dJ = 1 . Base case  
4: g^y = CROSSENTROPYBACKWARD(y; ^y; J; gJ )  
5: gb = SOFTMAXBACKWARD(b; ^y; g^y)  
6: gbeta; gz = LINEARBACKWARD(z; b; gb)  
7: ga = SIGMOIDBACKWARD(a; z; gz)  
8: galpha; gx = LINEARBACKWARD(x; a; ga) . We discard gx  
9: return parameter gradients galpha; gbeta  
