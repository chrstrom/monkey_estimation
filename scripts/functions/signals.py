from math import exp

class Signals:

    N = 513
    T = 10e-6


    def F(self, w0):
        
        sum = 0
        for n in range(0, self.N):
            sum += self.x(n)*exp(-1j*w0*n*self.T)

        return sum / self.N


if __name__ == '__main__':
    sig = Signals()

    print(sig.F(0.1))
