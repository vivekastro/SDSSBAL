import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy import io
from astropy.io import fits 
from astropy.io import ascii
from scipy.stats.distributions import chi2
from matplotlib.backends.backend_pdf import PdfPages


def computeSN1700(wave,flux,err):
    ww=np.where((wave >=1650) & (wave <= 1750))[0]
    return np.median(flux[ww])/np.median(err[ww])


df = np.genfromtxt('Inputfile_for_continuum_fitting_85_sources.txt',names=['ra','dec','z','pmf1','pmf2','pmf3'],dtype=(float,float,float,'|S35','|S35','|S35'))
#pp=PdfPages('SN1700_estimates_plot.pdf')
jxvf=open('SN1700_estimates_85_sources.txt','w')
for i in range(len(df)):
#for i in range(10):
    rootdir='/Users/vzm83/DR14QSO/Structure_Function/Spectra/AllSpectra'
    filename1 = os.path.join(rootdir,'spec-'+df['pmf1'][i]+'.fits')
    filename2 = os.path.join(rootdir,'spec-'+df['pmf2'][i]+'.fits')
    filename3 = os.path.join(rootdir,'spec-'+df['pmf3'][i]+'.fits')
    data1 = fits.open(filename1)[1].data
    data2 = fits.open(filename2)[1].data
    data3 = fits.open(filename3)[1].data
    wave1 = (10**(data1.loglam))/(1.0+df['z'][i])
    wave2 = (10**(data2.loglam))/(1.0+df['z'][i])
    wave3 = (10**(data3.loglam))/(1.0+df['z'][i])
    flux1 = data1.flux ; error1 = 1.0/np.sqrt(data1.ivar)
    flux2 = data2.flux ; error2 = 1.0/np.sqrt(data2.ivar)
    flux3 = data3.flux ; error3 = 1.0/np.sqrt(data3.ivar)
    # Check by plotting
    #fig,ax=plt.subplots(figsize=(20,10))
    #ax.plot(wave2,flux2,color='black',alpha=0.5,label=os.path.basename(filename2))
    #ax.plot(wave3,flux3,color='red',alpha=0.5,label=os.path.basename(filename3))
    #ax.plot(wave1,flux1,color='blue',alpha=0.5,label=os.path.basename(filename1))
    #
    #ax.plot(wave2,error2,color='black',alpha=0.15)
    #ax.plot(wave3,error3,color='red',alpha=0.15)
    #ax.plot(wave1,error1,color='blue',alpha=0.15)
    #ax.axvspan(1650,1750,alpha=0.25,color='green')
    #ax.set_xlabel('Rest Wavelength ($\AA$)')
    #ax.set_ylabel=('Flux')
    #allflux= np.concatenate((flux1,flux2,flux3))
    #ax.set_ylim(np.median(allflux)-3*np.std(allflux),np.median(allflux)+3*np.std(allflux))
    #ylim = plt.ylim()
    #xlim = plt.xlim()
    #labelstr = '{0}:  {1:5.4f}\n{2}:  {3:5.4f}\n{4}:  {5:5.4f}'.format(os.path.basename(filename2),computeSN1700(wave2,flux2,error2),os.path.basename(filename3),computeSN1700(wave3,flux3,error3),os.path.basename(filename1),computeSN1700(wave1,flux1,error1))
    #ax.text(xlim[0]+0.5*(xlim[1]-xlim[0]),ylim[1]-0.2*(ylim[1]-ylim[0]),labelstr,fontsize=18)
    #print labelstr
    #ax.set_title('SN1700 Estimates')
    #ax.legend(loc=1)
    #fig.tight_layout()
    #fig.savefig(pp,format='pdf')
    #plt.show()
    print>>jxvf,'{0}\t{1}\t\t{2:5.2f}'.format(os.path.basename(filename1),df['pmf1'][i],computeSN1700(wave1,flux1,error1))
    print>>jxvf,'{0}\t{1}\t\t{2:5.2f}'.format(os.path.basename(filename2),df['pmf2'][i],computeSN1700(wave2,flux2,error2))
    print>>jxvf,'{0}\t{1}\t\t{2:5.2f}'.format(os.path.basename(filename3),df['pmf3'][i],computeSN1700(wave3,flux3,error3))
#pp.close()
jxvf.close()
