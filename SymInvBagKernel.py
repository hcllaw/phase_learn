import numpy as np
from kerpy.BagKernel import BagKernel
from tools.GenericTests import GenericTests
from kerpy.GaussianKernel import GaussianKernel
from abc import abstractmethod

class SymInvBagKernel(BagKernel):
    def __init__(self,data_kernel, mdata=100, dim = 1):
        '''this kernel only works on rff features since it does not have a closed form'''
        BagKernel.__init__(self,data_kernel)

    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "" + BagKernel.__str__(self)
        s += "]"
        return s
    
    def rff_generate(self,mdata=100,dim=3):
        '''
        mdata:: number of random features for data kernel
        dim:: data dimensionality
        '''
        self.data_kernel.rff_generate(mdata,dim=dim)
        self.rff_num=mdata
    
    def rff_expand(self,bagX):
        nx=len(bagX)
        m = self.data_kernel.rff_num
        featuremeans=np.zeros((nx, m))
        for ii in range(nx):
            '''return the scaling before normalisation, as we need to work out the scale at the bottm without normalising'''
            # get our expectation of the mean (rff_expand is two matrix concatenated corresponding to the cos and sin matrix, with frequnecy along top.)
            featuremeans[ii]=np.sqrt(m/2.)*np.mean(self.data_kernel.rff_expand(bagX[ii]),axis=0)
            # Now we have a row corresponding to different frequencies (above) (cos vector followed by sin vector)
            '''now normalise the feature means to strip the symmetric contribution: (Ecos)^2+(Esin)^2'''
            # Now to strip symmetry, we take the cos^2 and add to sin^2 and take sqrt
            normalisers = np.sqrt(featuremeans[ii,:m/2]**2+featuremeans[ii,m/2:]**2)
            # construct the vector to renormalise with 
            normalisersc = np.concatenate( ( normalisers,normalisers ) , axis=0)
            '''rescale again and division'''
            featuremeans[ii]=np.sqrt(2./m)*featuremeans[ii]/normalisersc
            #This is the explicit feature map for each bag now (i.e. for each distribution)
        return featuremeans
    
    def kernel(self, bagX, bagY=None):
        featuresX=self.rff_expand(bagX)
        if bagY is not None:
            featuresY=self.rff_expand(bagY)
        else:
            featuresY=featuresX
        #K = np.dot(featuresX,featuresY.T)
        #print np.shape(K)
        return np.dot(featuresX,featuresY.T)
        
if __name__ == '__main__':
	print('Test units to be updated')
