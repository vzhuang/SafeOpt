import GPy.kern.src.stationary
from scipy.special import kv
from scipy.special import gamma
import numpy as np

class Matern(GPy.kern.src.stationary.Stationary):

    def __init__(self, input_dim, variance=1., nu=1.2, lengthscale=None, ARD=False, active_dims=None, name='Mat32'):
        super(Matern, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
        self.nu=nu

    def to_dict(self):
        input_dict = super(Matern, self)._to_dict()
        input_dict["class"] = "GPy.kern.Matern"
        return input_dict

    @staticmethod
    def _from_dict(kernel_class, input_dict):
        useGPU = input_dict.pop('useGPU', None)
        return Matern32(**input_dict)

    def K_of_r(self, r):
        # print('test1')
        # print(self.variance * (1. + np.sqrt(3.) * r) * np.exp(-np.sqrt(3.) * r))
        m = np.sqrt(2 * self.nu) * r
        res = self.variance * (np.power(2, 1-self.nu) / gamma(self.nu)) * \
            np.power(m, self.nu) * kv(self.nu, m)
        res[r==0] = self.variance
        # print('test2')
        # print(res)
        return res

    def dK_dr(self,r):
        return -3.*self.variance*r*np.exp(-np.sqrt(3.)*r)

    def Gram_matrix(self, F, F1, F2, lower, upper):
        """
        Return the Gram matrix of the vector of functions F with respect to the
        RKHS norm. The use of this function is limited to input_dim=1.
        :param F: vector of functions
        :type F: np.array
        :param F1: vector of derivatives of F
        :type F1: np.array
        :param F2: vector of second derivatives of F
        :type F2: np.array
        :param lower,upper: boundaries of the input domain
        :type lower,upper: floats
        """
        assert self.input_dim == 1
        def L(x, i):
            return(3. / self.lengthscale ** 2 * F[i](x) + 2 * np.sqrt(3) / self.lengthscale * F1[i](x) + F2[i](x))
        n = F.shape[0]
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                G[i, j] = G[j, i] = integrate.quad(lambda x : L(x, i) * L(x, j), lower, upper)[0]
        Flower = np.array([f(lower) for f in F])[:, None]
        F1lower = np.array([f(lower) for f in F1])[:, None]
        return(self.lengthscale ** 3 / (12.*np.sqrt(3) * self.variance) * G + 1. / self.variance * np.dot(Flower, Flower.T) + self.lengthscale ** 2 / (3.*self.variance) * np.dot(F1lower, F1lower.T))

    def sde(self):
        """
        Return the state space representation of the covariance.
        """
        variance = float(self.variance.values)
        lengthscale = float(self.lengthscale.values)
        foo  = np.sqrt(3.)/lengthscale
        F    = np.array([[0, 1], [-foo**2, -2*foo]])
        L    = np.array([[0], [1]])
        Qc   = np.array([[12.*np.sqrt(3) / lengthscale**3 * variance]])
        H    = np.array([[1, 0]])
        Pinf = np.array([[variance, 0],
        [0,              3.*variance/(lengthscale**2)]])
        # Allocate space for the derivatives
        dF    = np.empty([F.shape[0],F.shape[1],2])
        dQc   = np.empty([Qc.shape[0],Qc.shape[1],2])
        dPinf = np.empty([Pinf.shape[0],Pinf.shape[1],2])
        # The partial derivatives
        dFvariance       = np.zeros([2,2])
        dFlengthscale    = np.array([[0,0],
        [6./lengthscale**3,2*np.sqrt(3)/lengthscale**2]])
        dQcvariance      = np.array([12.*np.sqrt(3)/lengthscale**3])
        dQclengthscale   = np.array([-3*12*np.sqrt(3)/lengthscale**4*variance])
        dPinfvariance    = np.array([[1,0],[0,3./lengthscale**2]])
        dPinflengthscale = np.array([[0,0],
        [0,-6*variance/lengthscale**3]])
        # Combine the derivatives
        dF[:,:,0]    = dFvariance
        dF[:,:,1]    = dFlengthscale
        dQc[:,:,0]   = dQcvariance
        dQc[:,:,1]   = dQclengthscale
        dPinf[:,:,0] = dPinfvariance
        dPinf[:,:,1] = dPinflengthscale

        return (F, L, Qc, H, Pinf, dF, dQc, dPinf)    
