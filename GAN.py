import theano
import theano.tensor as T
import numpy as np
from utils import *
from updates import *
import matplotlib.pyplot as plt

# can be reproduced
seed = 12345
np.random.seed(seed)


class DataDistribution:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    def sample(self, sum):
        samples = np.random.multivariate_normal(self.mean, self.cov, sum)
        # print(samples.shape)
        samples = samples.reshape((sum, 2))
        return floatX(samples)


'''
# test the data
mean = [0,0]
cov = [[1,0],[0,10]]
data = DataDistribution(mean, cov)
data = data.sample(1000)
plt.plot(data[:,0], data[:,1], 'x')
plt.axis('equal')
plt.show()
'''


class GeneratorDistribution:
    def __init__(self, range):
        self.range = range

    def sample(self, sum):
        # samples = np.linspace(-self.range, self.range, sum) + \
        #         np.random.random(sum) * 0.01
        samples = np.random.uniform(low=-self.range, high=self.range,size=(sum, 20)) + \
                  np.random.random((sum, 20)) * 0.01
        samples = np.asmatrix(samples)
        samples = samples.reshape((sum, 20))
        # print(samples)
        return floatX(samples)


'''
# test the data
gen = GeneratorDistribution(10)
gen.sample(10)
'''

class Generator:
    def __init__(self, input_size, hidden_size, output_size):
        prefix = "gen_"
        self.in_size = input_size
        self.out_size = output_size
        self.hidden_size = hidden_size
        self.W1 = init_weights((self.in_size, self.hidden_size), prefix + "W1")
        self.b1 = init_bias(self.hidden_size, prefix + "b1")
        self.W11 = init_weights((self.hidden_size, self.hidden_size), prefix + "W11")
        self.b11 = init_bias(self.hidden_size, prefix + "b11")
        self.W2 = init_weights((self.hidden_size, self.out_size), prefix + "W2")
        self.b2 = init_bias(self.out_size, prefix + "b2")
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W11, self.b11]

    def generate(self, z):  # simple full-connected layer
        h = T.tanh(T.dot(z, self.W1) + self.b1)
        h2 = T.tanh(T.dot(h, self.W11) + self.b11)
        y = T.dot(h2, self.W2) + self.b2
        return y

class Discriminator:
    def __init__(self, input_size, hidden_size, output_size):
        prefix = "dis"
        self.in_size = input_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.W1 = init_weights((self.in_size, self.hidden_size), prefix + "W1")
        self.b1 = init_bias(self.hidden_size, prefix + "b1")
        #self.W11 = init_weights((self.hidden_size, self.hidden_size), prefix + "W11")
        #self.b11 = init_bias(self.hidden_size, prefix + "b11")
        self.W2 = init_weights((self.hidden_size, self.out_size), prefix + "W2")
        self.b2 = init_bias(self.out_size, prefix + "b2")
        self.params = [self.W1, self.b1, self.W2, self.b2]

    def discriminate(self, x):
        h = T.tanh(T.dot(x, self.W1) + self.b1)
        # h2 = T.tanh(T.dot(h, self.W11) + self.b11)
        y = T.nnet.sigmoid(T.dot(h, self.W2) + self.b2)
        return y

class GAN:
    def __init__(self, genSize, disSize, optimizer):   # genSize and disSize are lists
        self.genSize = genSize
        self.disSize = disSize
        self.optimizer = optimizer
        self.X = T.matrix("X")
        self.Z = T.matrix("Z")
        self.Generator = Generator(genSize[0], genSize[1], genSize[2])
        self.Discriminator = Discriminator(disSize[0], disSize[1], disSize[2])
        self.params_dis = self.Generator.params
        self.params_gen = self.Discriminator.params
        self.defineGraph()

    def defineGraph(self):  # define the calulation graph
        g = self.Generator.generate(self.Z)
        d_realData = self.Discriminator.discriminate(self.X)
        d_fakeData = self.Discriminator.discriminate(g)

        # calculate the discriminator's loss
        loss_d = -T.mean(T.log(d_realData) + T.log(1 - d_fakeData))
        grad_dis = []
        for param in self.params_dis:
            gparam = T.grad(loss_d, param)
            grad_dis.append(gparam)

        # calculate the generator's loss
        loss_g = T.mean(T.log(1 - d_fakeData))
        grad_gen = []
        for param in self.params_gen:
            gparam = T.grad(loss_g, param)
            grad_gen.append(gparam)

        # perform the optimizer
        learning_rate = T.dscalar("lr")
        optimizer = eval(self.optimizer)
        updates_gen = optimizer(self.params_gen, grad_gen, learning_rate)
        updates_dis = optimizer(self.params_dis, grad_dis, learning_rate)

        # train the network
        self.train_gen = theano.function(inputs=[self.X, self.Z, learning_rate],
                                     outputs=[loss_d, loss_g, g], updates=updates_gen)
        self.train_dis = theano.function(inputs=[self.X, self.Z, learning_rate],
                                         outputs=[loss_d, loss_g, g], updates=updates_dis)


def main():
    # set the hyperparams
    learning_rate = 1e-4
    batch_size = 100
    hidden_size = 100
    optimizer = "adam"
    dis_num = 1
    gen_num = 2

    # size
    gen_size = [20, hidden_size, 2]
    dis_size = [2, hidden_size, 1]

    # data
    data_real = DataDistribution([0,0], [[1,0],[0,20]])
    data_gen = GeneratorDistribution(10)

    print "compiling..."
    model = GAN(gen_size, dis_size, optimizer)

    print "training..."
    for epoch in xrange(1000):
        for i in range(dis_num):
            x = data_real.sample(batch_size)
            z = data_gen.sample(batch_size)
            # print(x.shape)
            # print(z.shape)
            loss_d, loss_g, _= model.train_dis(x, z, learning_rate)
            print "loss_d = ", loss_d
            print "loss_g = ", loss_g
        for i in range(gen_num):
            x = data_real.sample(batch_size)
            z = data_gen.sample(batch_size)
            # print(x.shape)
            # print(z.shape)
            loss_d, loss_g, _= model.train_gen(x, z, learning_rate)
            print "                                     loss_d = ", loss_d
            print "                                     loss_g = ", loss_g


    test_z = data_gen.sample(5000)
    test_x = data_real.sample(5000)
    _, _, output = model.train_gen(test_x, test_z, learning_rate)
    print(output.shape)
    real_data = data_real.sample(5000)
    plt.plot(real_data[:, 0], real_data, 'cx')
    plt.plot(output[:, 0], output[:, 1], 'rx')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()