import numpy as np

def Shifting(x):
    shifup = np.roll(x, -1, axis=1)[:,np.newaxis,:]
    shifdn = np.roll(x, 1, axis=1)[:,np.newaxis,:]
    output = np.hstack((x[:, np.newaxis, :], shifup, shifdn))
    return output

def Densifing(x):
    output = np.repeat(x, 2, axis=0)
    output = (output+np.roll(output,1, axis=0))/2.
    return output

if __name__ == '__main__':
    input = np.random.rand(8, 2000)
    output = Shifting(input)
    print(input)
    print(output)
    print(output.shape)

    output = Densifing(input)
    print(input)
    print(output)
    print(output.shape)
