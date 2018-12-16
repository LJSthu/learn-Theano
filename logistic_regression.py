import numpy
import theano
import theano.tensor as T

N = 3000
feat = 784

# make the dataset X = (400,784) Y = (400,1)
D = (numpy.random.randn(N,feat), numpy.random.randint(size = N, low=0, high=2))
training_step = 3000

# declare the x,y
x = T.dmatrix("x")
y = T.dvector("y")
learning_rate = T.dscalar("lr")

# declare the weight w and b
w = theano.shared(value=numpy.random.rand(feat), name="w")
b = theano.shared(value=0., name="b")

print("initialized weights \n")
print(w.get_value())
print(b.get_value())

# build the graph
output = 1/(1+T.exp(-T.dot(x, w)-b))
prediction = output > 0.5
cross_entropy = -y * T.log(output) - (1-y)*T.log(1-output)
loss = cross_entropy.mean() + 0.01*(w**2).sum()
gradW, gradb = T.grad(loss, [w, b])

# train function
train = theano.function(inputs=[x,y,learning_rate], outputs=[prediction, cross_entropy,loss, learning_rate], \
                        updates=((w,w-learning_rate*gradW), (b,b-learning_rate*gradb)))
# predict function
predict = theano.function(inputs=[x], outputs=prediction)

for i in range(training_step):
    if (i < 1000):
        learning_rate = 0.1
    else:
        learning_rate =0.01
    pred, cro, l,lr = train(D[0], D[1], learning_rate)
    print(i, l,lr)


print("target values for D:")
print(D[1])
print("prediction on D:")

result = predict(D[0])
print(result)
accuracy = (result == D[1]).mean()
print("final accuracy is: ")
print(accuracy)




