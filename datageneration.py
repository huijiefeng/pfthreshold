import numpy as np

def _coeff_generator(d,s,normalize = True, low = 1, high = 2, norm = 1):
    coeff = np.zeros(d)
    coeff[:s] = np.random.uniform(low,high,s)
    return coeff/np.sqrt(np.sum(coeff**2))*norm if normalize else coeff

def _Z_generator(n,d,rho,bias = 0):
    Z = np.random.normal(loc = bias, size=(n,d))
    for i in range(1,d):
        Z[:,i] = rho * (Z[:,i-1]-bias) + Z[:,i]
    return Z

class DataGeneration:
    def __init__(self,
                 model: str,
                 n: int,
                 d: int,
                 s: int,
                 rho: float = 0.5,
                 normalize: bool = True,
                 norm = 1,
                **kargs):
        if model == 'Manski':
            sig = kargs.get('sig',1.0)
            self.Model = Heteroskedastic(n,d,s,rho,normalize,norm,sig)
        elif model == 'Logistic':
            self.Model = Logistic(n,d,s,rho,normalize,norm)
        elif model =='ConditionalMean':
            sig = kargs.get('sig',1.0)
            ratio = kargs.get('ratio',0.5)
            mid = kargs.get('mid', 2.0)
            bias = kargs.get('bias', 0)
            self.Model = ConditionalMean(n,d,s,rho,normalize,norm,
                                        sig,ratio,mid,bias)
        else:
            raise Exception('Unknown model')
    
    def create(self):
        return self.Model

class Heteroskedastic:
    def __init__(self,
                n: int,
                d: int,
                s: int,
                rho: float = 0.5,
                normalize: bool = True,
                norm: float = 1.0,
                sig: float = 1.0):
        self.n = n
        self.d = d
        self.s = s
        self.rho = rho
        self.normalize = normalize
        self.norm = norm
        self.sig = sig
        self.coeff = _coeff_generator(d,s,normalize,norm=norm)

    def generate(self):
        X = np.random.normal(size = self.n)
        Z = _Z_generator(self.n,self.d,self.rho)
        mu = X - Z.dot(self.coeff)
        var_vec = 1 + 2*mu**2
        epsilon = np.random.normal(0,np.sqrt(var_vec),size=self.n)
        Y = np.sign(mu + self.sig * epsilon)
        return {'X': X,
                'Y': Y,
                'Z': Z,
                'true_coeff' : self.coeff}
    
    def __str__(self):
        return 'Manski model with n:{},d:{},s:{},rho:{},normalize:{},sig:{}'.format(
            self.n,self.d,self.s,self.rho,self.normalize,self.sig)

# different from other generating processes where the coeff is random every time
class Logistic:
    def __init__(self,
                n: int,
                d: int,
                s: int,
                rho: float = 0.5,
                normalize: bool = True,
                norm: float = 1.0):
        self.n = n
        self.d = d
        self.s = s
        self.rho = rho
        self.norm = norm
        self.normalize = normalize
        self.coeff = _coeff_generator(d,s,normalize,norm=norm)

    def generate(self):
        X = np.random.normal(size = self.n)
        Z = _Z_generator(self.n,self.d,self.rho)
        mu = X - Z.dot(self.coeff)
        epsilon = np.random.logistic(size=self.n)
        Y = np.sign(mu + epsilon)
        return {'X': X,
                'Y': Y,
                'Z': Z,
                'true_coeff' : self.coeff}
    
    def __str__(self):
        return 'Logistic model with n:{},d:{},s:{},rho:{},normalize:{}'.format(
            self.n,self.d,self.s,self.rho,self.normalize)

class ConditionalMean:
    def __init__(self,
                n: int,
                d: int,
                s: int,
                rho: float = 0.5,
                normalize: bool = True,
                norm: float = 1.0,
                sig: float = 1.0,
                ratio: float = 0.5,
                mid: float = 2.0,
                bias: float = 0.0):
        self.n = n
        self.d = d
        self.s = s
        self.rho = rho
        self.normalize = normalize
        self.norm = norm
        self.sig = sig
        self.ratio = ratio
        self.mid = mid
        self.bias = bias
        self.coeff = _coeff_generator(d,s,normalize,norm=norm)

    def generate(self):
        Y = 2*np.random.binomial(1,self.ratio,self.n)-1
        Z = _Z_generator(self.n,self.d,self.rho,bias=self.bias)
        #epsilon = np.random.normal(0,self.sig,size=self.n)
        epsilon = np.random.normal(0,self.sig*(1+np.abs(Z.dot(self.coeff))),size=self.n)
        X = Z.dot(self.coeff) + self.mid*Y + epsilon
        
        return {'X': X,
                'Y': Y,
                'Z': Z,
                'true_coeff' : self.coeff}
    
    def __str__(self):
        return 'ConditionalMean model with n:{},d:{},s:{},rho:{},normalize:{},sig:{},ratio:{},mid:{},bias:{}'.format(
            self.n,self.d,self.s,self.rho,self.normalize,self.sig,self.ratio,self.mid,self.bias)




    
def SplitData(dat,ratio = 0.5):
    mid = int(ratio * dat['X'].shape[0])
    dat_train = {'X': dat['X'][:mid],
                'Y': dat['Y'][:mid],
                'Z': dat['Z'][:mid,:],
                'true_coeff' : dat['true_coeff']}
    dat_test = {'X': dat['X'][mid:],
                'Y': dat['Y'][mid:],
                'Z': dat['Z'][mid:,:],
                'true_coeff' : dat['true_coeff']}
    return dat_train,dat_test