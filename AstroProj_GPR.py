
# coding: utf-8

# In[ ]:


from astropy.io import fits
import pylab as p
import fitsio
import numpy as np
import os
import matplotlib.pyplot as plt

#%matplotlib inline
#----------------------------------------------------------------------------------------------------------------
#To get the magnitude of each epoch from the spectrum using the mjd's(32 observations)
#----------------------------------------------------------------------------------------------------------------
#Read the RM(target) file in hand
tiled = fits.open('tiled_sample.fits')
t = tiled[1].data
mjd = t.MJD[0]
plate = t.PLATE[0]
fiberid = t.FIBERID[0]
z = t.Z[0]

#Create path for the spectrum directory to read in based on target mjd's
spec_dir =  '/uufs/chpc.utah.edu/common/home/sdss02/ebosswork/eboss/spectro/redux/v5_10_0/spectra/lite'
mag_range = [1430, 1470]
mag = np.empty(len(mjd))
sig = np.empty(len(mjd))

for i in range(len(mjd)):
    spec_name = os.path.join(spec_dir, str(plate[i]), 'spec-%04d-%05d-%04d.fits' % (plate[i], mjd[i], fiberid[i]))
    try:
        a = fitsio.read(spec_name, 1)
        flux, loglam,ivar = a['flux'], a['loglam'], a['ivar']
        lam = 10 ** loglam
        #plt.plot(lam, flux, lw=0.6)
        #plt.show()
#Get the magnitudes for the wavelength and flux in range:        
        ixs = (lam > mag_range[0] * (1 + z)) & (lam < mag_range[1] * (1 + z))
        good = (ivar[ixs] > 0)
        mag[i] = flux[ixs][good].sum()
        sig[i] = (1/ivar[ixs[good]]).sum()
    except IOError as e:
      mag[i] = -1
      sig[i] = -1
        

np.savetxt('dataset.dat', np.array([mjd, mag, sig]))

#-----------------------------------------------------------------------------------------------------------------
#Use the set of magnitudes obtained with the mjd's as time to generate the GPR model:
#-----------------------------------------------------------------------------------------------------------------
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import fitsio
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


#Read in the file from above
mjd, mag, sig = np.loadtxt('dataset.dat')

# select good pixels
sig = np.sqrt(sig)
mjd -= mjd[0]
# select pixels to fit the model

ixs = mag > 0

X = mjd[ixs][:, None]
err = sig[ixs]

y = mag[ixs]
y -= y.mean()
#----------------------------------------------------------------------------------
#Mag vs time plot
plt.plot(X[:,0], y, '*')
plt.xlabel('Time(days)')
plt.ylabel('Magnitude')
#plt.savefig('Mag-time.jpg',dpi = 300)
plt.show()

#----------------------------------------------------------------------------------
# select pixels to fit the model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, np.vstack((y, err)).T, test_size=0.5, random_state=0)
#Create the model based on selecting 50% of the pixels
import george
from george import kernels

# Set up the Gaussian process:
#k1 = np.mean(err) ** 2 * kernels.ExpKernel(1.0)
#k1 = np.mean(err) ** 2 * kernels.ExpSquaredKernel(1.0)
k1 = np.mean(err) ** 2 * kernels.Matern32Kernel(1.0)
k2 = np.mean(err) ** 2 * kernels.Matern52Kernel(15.0)
kernel = k1 + k2
gp = george.GP(kernel)

# Pre-compute the factorization of the matrix.
gp.compute(X_train, y_train[:, 1])

# Compute the log likelihood.
print(gp.lnlikelihood(y_train[:,0]))

# predict
t = np.linspace(0, 200, 500)
mu, cov = gp.predict(y_train[:,0], t)
std = np.sqrt(np.diag(cov))

#Plot
fig = plt.figure(figsize=(12, 8))

plt.plot(X_test[:,0], y_test[:, 0], '*', c='k', label = 'Test points')
plt.errorbar(X_train[:,0], y_train[:,0], y_train[:, 1], fmt='o', label = 'Trained points')#the points from the train as well as test set

plt.plot(t, mu) #the line that fits the model

plt.fill_between(t, mu + std, mu - std, color='gray')#The error on the model
plt.title('Matern32 + Matern52 Kernel')
plt.xlabel('Time(days)')
plt.ylabel('Magnitude')
#plt.savefig('matern32-52kernel.jpg',dpi = 300)
plt.show()

