#!/usr/bin/python

import numpy as np
from astropy.io import fits
import scipy as sp
import glob
import os
import matplotlib.pyplot as plt
from astropy.time import Time
from datetime import date
from scipy.ndimage.filters import gaussian_filter1d,uniform_filter1d
import calendar
import sys
from empca import empca 
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages
import scipy
from astropy.stats import sigma_clip as sigmaclip
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit 
import pandas as pd

def powerlawFunc(xdata,  amp,index):
    return amp*np.power(xdata,index)


def fitPowerlaw(wave,flux,weight, amp=1,index=1): 
    from lmfit import minimize, Parameters, fit_report, Minimizer
    import numpy as np
    import scipy.optimize as optimization
    wav_range= [(1250,1350),(1700,1800),(1950,2500),(2650,2710),(2950,3700),(3950,4050)]
    xx = np.where( ((wave > 1250 ) & (wave < 1300))  | ((wave > 1590 ) & (wave < 1750))  |((wave > 2000 ) & (wave < 2100)) | ((wave > 2350 ) & (wave < 2500)) | ((wave > 2650 ) & (wave < 2700)) | ((wave > 2950 ) & (wave < 3700)) | ((wave > 3950 ) & (wave < 4050))   )
    x0= [amp,index]
    xdata=np.asarray(wave[xx])  
    ydata=np.asarray(flux[xx])
    sigma=np.asarray(weight[xx])
    #print len(xdata),len(ydata),len(sigma) 
    try:    
        popt, pcov = optimization.curve_fit(powerlawFunc, xdata, ydata, x0, 1.0/np.sqrt(sigma))
    except (RuntimeError, TypeError):
        popt,pcov = (1,1),1
    #print popt
    #popt, pcov = optimization.curve_fit(func, xdata, ydata, x0)
    model = powerlawFunc(wave,popt[0],popt[1])
    chi2 = ((flux - model)*np.sqrt(weight))**2
    rchi2 = np.sum(chi2)/(len(xdata) - 2)
    print 'Reduced Chi Square : {0}  Number of points: {1}'.format(rchi2,len(xdata))
    return (popt,pcov)   

def maskOutliers(wave,flux,weight,amp,index):
    model = powerlawFunc(wave,amp,index)
    std =np.std(flux[weight > 0])
    fluxdiff = flux - model
    ww = np.where (np.abs(fluxdiff) > 3*std)[0]
    print 'Testing Fraction of cut pixels',len(wave),len(ww),ww,amp,index
    nwave = np.delete(wave,ww)
    nflux = np.delete(flux,ww)
    nweight = np.delete(weight,ww)
    print len(nwave),len(nflux),len(nweight) 
    return nwave,nflux,nweight

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except (ValueError, msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = np.asarray(y[0]) - np.abs( y[1:half_window+1][::-1] - np.asarray(y[0]) )
    lastvals = np.asarray(y[-1]) + np.abs(y[-half_window-1:-1][::-1] - np.asarray(y[-1]))
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def _solve(A, b, w):
    """
    Solve Ax = b with weights w; return x
    
    A : 2D array
    b : 1D array length A.shape[0]
    w : 1D array same length as b
    """
  
    #- Apply weights
    # nvar = len(w)
    # W = dia_matrix((w, 0), shape=(nvar, nvar))
    # bx = A.T.dot( W.dot(b) )
    # Ax = A.T.dot( W.dot(A) )
    
    b = A.T.dot( w*b )
    A = A.T.dot( (A.T * w).T )

    if isinstance(A, scipy.sparse.spmatrix):
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        # x = np.linalg.solve(A, b)
        x = np.linalg.lstsq(A, b)[0]
        
    return x



def maskAbsorption(wave, flux,weight,pspectra,pmodel,usigma=5,lsigma=2,nloop=0,debug=0,totalloop=9):
    diff = flux - pspectra
    sdiff = flux[weight >0] - pspectra[weight > 0]
    oweight = weight
    #filter to clip the  lowest five percentile flux; naturally cuts the deep absorptions
    botcut = np.percentile(flux,5)
    bwm = np.where(flux < botcut)
    print  (nloop*totalloop)-1,usigma,lsigma
    if (nloop*totalloop)-1 == 0:
        #filter to reject first wavelengths & last wavelengths; removes 
        wavcut = np.percentile(wave[weight>0],[1,99])
        awm = np.where((wave < wavcut[0]) | (wave > wavcut[1]))
        weight[awm] = 0

        #filter to cut low SNR pixels;current threshold is at 5; may be good to reduce it
        sn = flux*np.sqrt(weight)
        snc = np.where((sn<0) | (sn > 80))
        weight[snc] = 0
    #sn = np.where(flux/np.sqrt(weight) <0.1)
    #plt.plot(wave,flux/np.sqrt(weight))
    #plt.axhspan(np.median(flux/np.sqrt(weight)),0,ls='--')
    #plt.show()
    #print 'Standard deviation Flux, Diff : ',np.std(flux),np.std(diff)
    
    #wma = np.where((diff <2* np.std(diff)) & (flux < np.median(sigmaclip(flux).data[~sigmaclip(flux).mask])) )
    #wme = np.where((diff >5* np.std(diff)) & (flux > np.median(sigmaclip(flux).data[~sigmaclip(flux).mask])) )
    #ws = np.where( ((flux/pspectra) < 0.9) & (flux < np.median(sigmaclip(flux).data[~sigmaclip(flux).mask])))
    #wma = np.where((diff <lsigma* np.std(diff)) & (flux < 0.9*pmodel) & (wave < 2000) )
    #wmb = np.where((diff <lsigma* np.std(diff)) & (flux < 0.75*pmodel) & (wave > 2000) )
    #wme = np.where((diff >usigma* np.std(diff)) & (flux > 0.8*pmodel) )
    
    #filter to clip absorptions in the blue part of the spectrum < 1500
    wma = np.where((-1*diff >lsigma* np.std(sdiff)) & (flux < 0.9*pmodel)  & (wave < 1550)  )
    #filter to clip absorptions in the red part of the spectrum < 1500
    wmb = np.where((-1*diff >lsigma* np.std(sdiff)) & (flux < 0.9*pmodel)  & (wave > 1550)  )
    
    #filter to bad sky substraction residuals in the red part of the spectrum
    wmc = np.where((np.abs(diff) >2.5*np.std(sdiff)) & (wave > 2000) & ~((wave > 2785) & (wave < 2815)) )
    
    #filter to clip the deviant pixels above the median filter; excludes the prominent emission lines ; not employed
    wme = np.where((diff >8* np.std(sdiff)) & (flux > 0.8*sp.ndimage.filters.median_filter(flux,551,mode='nearest'))  & ~((wave > 1540) & (wave < 1560))  \
            & ~((wave > 1880) & (wave < 1915))  & ~((wave > 2785) & (wave < 2815))  )
    
    #low pass filter to filter out the narrow absorption systems
    lp = np.where( sp.ndimage.filters.median_filter(flux,11,mode='nearest')- flux > 3*np.std(sdiff))
    
    #ws = np.where( ((flux/pspectra) < 0.9) & (flux < pmodel))

    
    #weight[ws] = 0
    # Imeplement savitzy golay
    sav_diff = savitzky_golay(bnflux[i],25,1) - flux
    sma = np.where((sav_diff < 3*np.std(sdiff)) & (flux < 0.9*sp.ndimage.filters.median_filter(flux,551,mode='nearest') ))
    sma1 = np.where((sav_diff > 3*np.std(sdiff)) & (flux > 0.9*sp.ndimage.filters.median_filter(flux,551,mode='nearest') ))
    print '******NOTE*******', np.std(sdiff),np.std(sav_diff) 
    smb = np.where((diff > 10*np.std(sdiff)) & (flux > pmodel))
    
    nm=[]
    for l in range(len(flux)-1):
        if ((flux[l] - flux[l+1] ) > 10*np.std(sdiff)):
            nm.append(l+1) 
        if (-(flux[l] - flux[l+1] ) > 5*np.std(sdiff)):
            nm.append(l)
    im = []
    for ll in range(20,len(flux) - 20):
        medfl = np.median(diff[ll-20:ll+20])
        if medfl > 1*np.std(diff[ll-20:ll+20]):
            im.append(ll)
    start = -1
    contigous_segments=[]
    for jj,jx in enumerate(im[0:-1]):
        if start < 0:
            start = jx
        else :
            if (im[jj+1] - im[jj]) !=1 :
                end = jx
                contigous_segments.append((start,end))
                start = -1

    nim =[]
    for bounds in contigous_segments:
        nim.append(np.arange(bounds[0]-11,bounds[1]+11))
    print 'contigous Check:',contigous_segments
    print 'nim Check:',nim
    nnim =  np.array([item for sublist in nim for item in sublist])
    print 'Nnim Check:',nnim
    weight[sma] = 0 # savitzky-golay mask 
    weight[sma1] = 0 # savitzky-golay mask 
    weight[smb] = 0 # savitzky-golay mask 
    
    weight[nm] = 0 # to catch sharp turnovers in absortion edges
    
    weight[wma] = 0 # clip blue absorptions
   
    weight[wmb] = 0 # clip red absorptions
    
    weight[wmc] = 0 # bad sky residuals
    weight[bwm] = 0 # lowest 5 percentile flux
    weight[lp] = 0 # low pass narrow line filter
    #if len(nnim)>0:
    #    weight[nnim] = 0 # contigous segments where median flux over 20 pixels are above 2 sigma  of flux difference
    weight[wme] = 0
    if debug==1:
        print '**'*51
        print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in Sav gal {}'.format(nloop*9,len(flux),len(sma[0]))
        print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in NM loop {}'.format(nloop*9,len(flux),len(nm))
        print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in lo lsig {}'.format(nloop*9,len(flux),len(wma[0]))
        print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in hi lsig {}'.format(nloop*9,len(flux),len(wmb[0]))
        print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in sky sub {}'.format(nloop*9,len(flux),len(wmc[0]))
        print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in percent {}'.format(nloop*9,len(flux),len(bwm[0]))
        if (nloop*totalloop)-1==0:
            print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in Wav cut {}'.format(nloop*9,len(flux),len(awm[0]))
            print 'iteration:{}\tTotalpixels:{}\t N_pixels masked in SNR cut {}'.format(nloop*9,len(flux),len(snc[0]))
        print '**'*51
        tfig,tax = plt.subplots(3,3,figsize=(15,8),sharex=True)
        tweight = oweight ; tweight[sma] = 0
        tax[0,0].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(sma[0])))
        tax[0,0].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
        tax[0,0].plot(wave[sma],flux[sma],'.',color='blue')
        tax[0,0].set_title('Implementing the 3 sigma savitzy golay  filter ')
        tax[0,0].legend(loc=1) 
        
        tweight = oweight ; tweight[nm] = 0
        tax[0,1].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(nm)))
        tax[0,1].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
        tax[0,1].plot(wave[nm],flux[nm],'.',color='blue')
        tax[0,1].set_title('Implementing the nm ')
        tax[0,1].legend(loc=1) 
        
        tweight = oweight ; tweight[wma] = 0
        tax[0,2].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(wma[0])))
        tax[0,2].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
        tax[0,2].plot(wave[wma],flux[wma],'.',color='blue')
        tax[0,2].set_title('Implementing the absorption <  1500 and lsigma')
        tax[0,2].legend(loc=1) 
        
        tweight = oweight ; tweight[wmb] = 0
        tax[1,0].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(wmb[0])))
        tax[1,0].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
        tax[1,0].plot(wave[wmb],flux[wmb],'.',color='blue')
        tax[1,0].set_title('Implementing the absorption >  1500 and lsigma')
        tax[1,0].legend(loc=1) 

        tweight = oweight ; tweight[wmc] = 0
        tax[1,1].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(wmc[0])))
        tax[1,1].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
        tax[1,1].plot(wave[wmc],flux[wmc],'.',color='blue')
        tax[1,1].set_title('Implementing  sky residuals in the red part ')
        tax[1,1].legend(loc=1) 
        
        tweight = oweight ; tweight[bwm] = 0
        tax[1,2].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(bwm[0])))
        tax[1,2].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
        tax[1,2].plot(wave[bwm],flux[bwm],'.',color='blue')
        tax[1,2].set_title('Implementing  5 and 99 percentile fluxes  cut')
        tax[1,2].legend(loc=1) 
        if (nloop*totalloop)-1 == 0: 
            tweight = oweight ; tweight[awm] = 0
            tax[2,0].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(awm[0])))
            tax[2,0].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
            tax[2,0].plot(wave[awm],flux[awm],'.',color='blue')
            tax[2,0].set_title('Implementing the cut of first and last wavelengths')
            tax[2,0].legend(loc=1) 
        
        tweight = oweight ; tweight[lp] = 0
        tax[2,1].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(lp[0])))
        tax[2,1].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
        tax[2,1].plot(wave[lp],flux[lp],'.',color='blue')
        tax[2,1].set_title('Implementing the cut of lowpass filter')
        tax[2,1].legend(loc=1) 
        
        if (nloop*totalloop)-1 == 0: 
            tweight = oweight ; tweight[snc] = 0
            tax[2,2].plot(wave,flux,color='black',label='T:'+str(len(flux))+'-M:'+str(len(snc[0])))
            tax[2,2].plot(wave,tweight/np.median(tweight)*np.median(flux),color='red',alpha=0.3)
            tax[2,2].plot(wave[snc],flux[snc],'.',color='blue')
            tax[2,2].set_title('Implementing the cut based on SNR between 0 and 80')
            tax[2,2].legend(loc=1)
        tfig.tight_layout()    
        plt.show()
    rbotcut = np.percentile(flux[weight>0],2)
    awm = np.where(flux < rbotcut)
    #print nloop
    
    if nloop==1:
        print 'nloop = 1, beginning to cut by nsigma'
        nsigma = (flux - pspectra)*np.sqrt(weight)
        ns = np.where((np.abs(nsigma)>2.5) & ~((wave > 1545) & (wave < 1565))  \
            & ~((wave > 1860) & (wave < 1925))  & ~((wave > 2785) & (wave < 2815)) )[0]
        #weight[ns]=0
        print 'No: of pixels masked in the final based on the nsigma: ', len(ns)
    return flux,weight,np.median(sigmaclip(flux).data[~sigmaclip(flux).mask])


from matplotlib import rc
rc('font',**{'family':'Times New Roman'})
#rc('text', usetex=True)

#Output file for saving 
pp1 = PdfPages('EMPCA_BALreconstruction_eBOSS_final_nmask-IV_48635.pdf')
pp2 = PdfPages('EMPCA_BALreconstruction_Nsigma_eBOSS_final_nMask-IV_48635.pdf')
saveChi2 = open('EMPCA_BALreconstruction_new_eBOSS_final_chi2_nMask-IV_48635.txt','w')
savecoeff = open('EMPCA_BALreconstruction_new_eBOSS_final_coeff_nMask-IV_48635.txt','w')
#

fsm = np.genfromtxt('AllPMF_info_fullsample.txt',names=['name','ra','dec','plate','mjd','fiber','Z','Ztag'],dtype=('|S12',float,float,int,int,int,float,'|S7'))

iteration = 10
#balrmid = np.loadtxt('../../Downloads/pat_list.csv',usecols=(0),dtype=int, delimiter=',')
#balrminfo = np.genfromtxt('../../Downloads/pat_list.csv',usecols=(0,1,2,3,4,5,6,7),names=['id','class','c4','al3','mg2','notes','vwidth'],dtype=(int,int,'|S10','|S10','|S10','|S50','S20'), delimiter=',')
#Read in the eigenvectors and wavelengths
comp = np.load('EMPCA_components_ncomp_8z_DR12q_48365.npz')
evecs = comp['evecs']
p=len(evecs)
lll = comp['wave']
rpcacomp = np.load('RandomizedPCA_components_ncomp_8z.npz')
rpcaevecs = rpcacomp['evecs']

nmfcomp=np.load('NMF_components_ncomp_8z.npz')
nmfevecs = nmfcomp['evecs']

nmfcomp1=np.load('GBZNMF_components_ncomp_8z_DR12q.npz')
nmfevecs1 = nmfcomp1['evec']

mdata = fits.open('SDSSIV_BALQSO_BigFits.fits')
mdatafmap = fits.open('SDSSIV_BALQSO_BigFits_fibermap.fits')

start,end = 0,len(mdata)-1

#mdata=mdata[start:end]
#mdatafmap= mdatafmap[start:end]

baldataf = np.zeros((len(mdata)-1,4700))
baldatav = np.zeros((len(mdata)-1,4700))
baldatam = np.zeros((len(mdata)-1,4700))
baldataw = np.zeros((len(mdata)-1,4700))

zarray = []
platearray = []
mjdarray=[]
fiberarray=[]
namearray=[]
raarray=[]
decarray=[]



#for i in range(0,len(mdata)-1):
for i in range(0,end-start):
    td = mdata[start+i+1].data
    tdf = mdatafmap[start+i+1].data
    print 'Master file',i
    for j in range(len(td)):
        baldataf[i,j]=td['flux'][j]
        baldatav[i,j]=td['ivar'][j]
        baldatam[i,j]=td['and_mask'][j]
        baldataw[i,j]=10**td['loglam'][j]

    platearray.append(tdf['PLATE'][0])
    mjdarray.append(tdf['MJD'][0])
    fiberarray.append(tdf['FIBERID'][0])
    raarray.append(tdf['PLUG_RA'][0])
    decarray.append(tdf['PLUG_DEC'][0])
    yx = np.where((fsm['plate'] == tdf['PLATE']) & (fsm['mjd'] == tdf['MJD']) & (fsm['fiber'] == tdf['FIBERID']))[0]
    zarray.append(fsm['Z'][yx[0]])
    namearray.append(fsm['name'][yx[0]])

print '************ Completed creating master file *************************************************************************'
balinfo=pd.DataFrame()
balinfo['RA'] = np.array(raarray)
balinfo['DEC'] = np.array(decarray)
balinfo['PLATE'] = np.array(platearray)
balinfo['MJD'] = np.array(mjdarray)
balinfo['FIBER'] = np.array(fiberarray)
balinfo['Z'] = np.array(zarray)
balinfo['name'] = np.array(namearray)
print balinfo['name']
#lksdjf=raw_input()
#Read in BAL data
#baldataf = fits.open('RmQSO_BAL.fits')[0].data
#baldatav = fits.open('RmQSO_BAL.fits')[1].data
#baldatam = fits.open('RmQSO_BAL.fits')[2].data
#balinfo = fits.open('RmQSO_BAL.fits')[3].data
#balhead = fits.open('RmQSO_BAL.fits')[1].header





#Same wavelength grid as that of eigenvectors
bbins = len(lll)
bnflux = np.zeros((len(baldataf),bbins))
bnweight = np.zeros((len(baldataf),bbins))

#for kk in range(len(baldataf)):
for kk in range(0,end-start):
    try:
        bbflux = np.trim_zeros(baldataf[kk])
        bbwave =  np.trim_zeros(baldataw[kk])/(1.0+balinfo['Z'][kk])#10**(balhead['CRVAL1'] + np.arange(len(bbflux))*balhead['CD1_1'])/(1.0+balinfo['Z'][kk])
        bbvar = np.trim_zeros(baldatav[kk])
        bbmas = baldatam[kk][0:len(bbflux)]
        bbweight = (bbvar*(bbmas == 0)).copy()
        bws = np.searchsorted(lll, bbwave)
        print 'interpolate',kk,len(bbflux),len(bbweight),len(bbvar),len(bbmas),len(bbwave)
        #print len(bbflux),len(bws),len(bbweight)
        for j, k in enumerate(bws):
            if k == 0 or k == bbins:
                continue
            bnflux[kk,k] += bbweight[j]*bbflux[j]
            bnweight[kk,k] += bbweight[j]
        wa = bnweight[kk] > 0
        bnflux[kk,wa]/= bnweight[kk,wa]
    except:
        print 'interpolation failed at ',kk
print '********** Completed interpolation step ************************************************************************'
print '********** Begin reconstruction ********************************************************************************'
#gkahs=raw_input()
#Reconstruct the spectra using components
#for i in range(len(bnflux)):
for i in range(0,end-start):
    try:
        #first iteration to exclude prominent absorption line regions
        lm = np.where( (lll <= 1240) | ((lll >=1295) & ( lll <=1400)) | ((lll >=1430) & (lll <=1546)) | ((lll >=1780) & (lll <=1880)))[0]
        #
        mmflux= bnflux[i][:] ; mmweight = bnweight[i][:]
        mnflux= bnflux[i][:] ; mnweight = bnweight[i][:]
        mnflux1= bnflux[i][:] ; mnweight1 = bnweight[i][:]
        mpmflux= bnflux[i][:] ; mpmweight = bnweight[i][:]
        mpowflux = bnflux[i][:] ; mpowweight = bnweight[i][:]
        mwmflux= bnflux[i][:] ; mwmweight = bnweight[i][:]
        #mmweight[lm]=0 ; mnweight[lm] =0; mnweight1[lm] = 0; mpmweight[lm] = 0 ; mpowweight[lm] = 0; mwmweight[lm] = 0 ; 
    
        #plt.plot(lll,mmflux)
        #plt.plot(lll,mmweight)
        #plt.show()
        coeff = _solve(evecs.T, mmflux,mmweight)
        rpcacoeff = _solve(rpcaevecs.T, mpmflux,mpmweight)
        nmfcoeff = _solve(nmfevecs.T, mnflux,mnweight)
        nmfcoeff1 = _solve(nmfevecs1.T, mnflux1,mnweight1)
    
        pcoeff = np.dot(evecs,mmflux) 
        pca_spectra =   np.dot(pcoeff[:p], evecs[:p])
        wpca_spectra =   np.dot(coeff[:p], evecs[:p])
        rpca_spectra =   np.dot(rpcacoeff[:p], rpcaevecs[:p])
        nmf_spectra =   np.dot(nmfcoeff[:p], nmfevecs[:p])
        nmf_spectra1 =   np.dot(nmfcoeff1[:p], nmfevecs1[:p])
    
        ##first iteration 
        #coeff = _solve(evecs.T, bnflux[i],bnweight[i])
        #rpcacoeff = _solve(rpcaevecs.T, bnflux[i],bnweight[i])
        #nmfcoeff = _solve(nmfevecs.T, bnflux[i],bnweight[i])
        #nmfcoeff1 = _solve(nmfevecs1.T, bnflux[i],bnweight[i])
        #pcoeff = np.dot(evecs, bnflux[i]) 
        #pca_spectra =   np.dot(pcoeff[:p], evecs[:p])
        #rpca_spectra =   np.dot(rpcacoeff[:p], rpcaevecs[:p])
        #nmf_spectra =   np.dot(nmfcoeff[:p], nmfevecs[:p])
        #nmf_spectra1 =   np.dot(nmfcoeff1[:p], nmfevecs1[:p])
        #wpca_spectra =   np.dot(coeff[:p], evecs[:p])
        mflux= bnflux[i] ; mweight = bnweight[i]
        nflux= bnflux[i] ; nweight = bnweight[i]
        nflux1= bnflux[i] ; nweight1 = bnweight[i]
        pmflux= bnflux[i] ; pmweight = bnweight[i]
        powflux = bnflux[i] ; powweight = bnweight[i]
        wmflux= bnflux[i] ; wmweight = bnweight[i]
        #plt.plot(lll,bnflux[i])
        #plt.plot(lll[bnweight[i]>0],bnflux[i][bnweight[i]>0])
        #plt.plot(lll[wmweight>0],mflux[wmweight>0])
        titletxt= 'Training Sample: DR12q     name: {0}    RA: {1}    DEC: {2}    PLATE: {3}    MJD: {4}    FIBER: {5}    Z:{6}'.format(balinfo['name'][i],balinfo['RA'][i],balinfo['DEC'][i],balinfo['PLATE'][i],balinfo['MJD'][i],balinfo['FIBER'][i],balinfo['Z'][i])
        print titletxt
        powopt,powcov=fitPowerlaw(lll,savitzky_golay(powflux,15,2),powweight)
        pbwave,pbflux,pbweight = maskOutliers(lll,powflux,powweight,powopt[0],powopt[1])
        print 'Flux',powflux
        print 'PowerlAW',powopt
        powopt,powcov=fitPowerlaw(pbwave,savitzky_golay(pbflux,15,2),pbweight)
        pmodel =  powerlawFunc(lll,powopt[0],powopt[1])
        #if i in [0,1,2,3,4,5,10,50,34,23,56]:
        #    plt.plot(lll,bnflux[i],color='black',alpha=0.3)
        #    plt.plot(lll,pmodel,ls='--',color='red')
        #    plt.show()
    
        
        for it in range(1,iteration):
            #print 'Nloop inside loop:',it,iteration,it/(iteration-1)
            #mflux,mweight,mmed  = maskAbsorption(lll,mflux,mweight,pca_spectra,pmodel,5,5-0.2*i)
            #pmflux,pmweight,pmmed  = maskAbsorption(lll,pmflux,pmweight,rpca_spectra,pmodel,5,5-0.2*i)
            #nflux,nweight,nmed  = maskAbsorption(lll,nflux,nweight,nmf_spectra,pmodel,5,5-0.2*i)
            #nflux1,nweight1,nmed1  = maskAbsorption(lll,nflux1,nweight1,nmf_spectra1,pmodel,8,2.8-0.2*i,it/(iteration-1.))
            #wmflux,wmweight,wmmed  = maskAbsorption(lll,wmflux,wmweight,wpca_spectra,pmodel,8,2.8-0.2*i,it/(iteration-1.))
            nflux1,nweight1,nmed1  = maskAbsorption(lll,nflux1,nweight1,nmf_spectra1,pmodel,8,5-0.2*it,it/(iteration-1.))
            wmflux,wmweight,wmmed  = maskAbsorption(lll,wmflux,wmweight,wpca_spectra,pmodel,8,5-0.2*it,it/(iteration-1.))
            coeff = _solve(evecs.T, wmflux,wmweight)
            rpcacoeff = _solve(rpcaevecs.T, pmflux,pmweight)
            nmfcoeff = _solve(nmfevecs.T, nflux,nweight)
            nmfcoeff1 = _solve(nmfevecs1.T, nflux1,nweight1)
    
            pcoeff = np.dot(evecs, mflux) 
            pca_spectra =   np.dot(pcoeff[:p], evecs[:p])
            wpca_spectra =   np.dot(coeff[:p], evecs[:p])
            rpca_spectra =   np.dot(rpcacoeff[:p], rpcaevecs[:p])
            nmf_spectra =   np.dot(nmfcoeff[:p], nmfevecs[:p])
            nmf_spectra1 =   np.dot(nmfcoeff1[:p], nmfevecs1[:p])
        wm = np.where((wmweight > 0) & (wpca_spectra > 0) & (bnflux[i] > 0)& (lll > min(bbwave)) & (lll < max(bbwave)) & ~((lll > 1525) & (lll < 1565))  \
                & ~((lll > 1860) & (lll < 1925))  & ~((lll > 2785) & (lll < 2815)) )[0]
        wmn = np.where((nweight1 > 0) & (nmf_spectra1 > 0) & (bnflux[i] > 0)& (lll > min(bbwave)) & (lll < max(bbwave))& ~((lll > 1525) & (lll < 1565))  \
                & ~((lll > 1860) & (lll < 1925))  & ~((lll > 2785) & (lll < 2815)) )[0]
        #wm = np.where((wmweight > 0) &(wpca_spectra > 0) )
        #wmn = np.where((nweight1 > 0) & (nmf_spectra1 > 0) )
        
        #----------------------------------------------------------------------------------
        print>>savecoeff,'{0}\t{1:5.4f}\t{2:5.4f}\t{3:5.4f}\t{4:5.4f}\t{5:5.4f}\t{6:5.4f}\t{7:5.4f}\t{8:5.4f}\t{9:5.4f}\t{10:5.4f}\t{11:5.4f}\t{12:5.4f}\t{13:5.4f}\t{14:5.4f}\t{14:5.4f}\t{15}\t{16}\t{17}'.format(balinfo['name'][i],\
                coeff[0],coeff[1],coeff[2],coeff[3],coeff[4],coeff[5],coeff[6],coeff[7],\
                nmfcoeff1[0],nmfcoeff1[1],nmfcoeff1[2],nmfcoeff1[3],nmfcoeff1[4],nmfcoeff1[5],nmfcoeff1[6],nmfcoeff1[7],balinfo['PLATE'][i],balinfo['MJD'][i],balinfo['FIBER'][i])
        #----------------------------------------------------------------------------------
        #save the normalized files
        if balinfo['PLATE'][i] >= 10000 :
            savefilename = 'normPCA-{0:05d}-{1:05d}-{2:04d}.txt'.format(balinfo['PLATE'][i],balinfo['MJD'][i],balinfo['FIBER'][i])
        else:
            savefilename = 'normPCA-{0:04d}-{1:05d}-{2:04d}.txt'.format(balinfo['PLATE'][i],balinfo['MJD'][i],balinfo['FIBER'][i])
        np.savetxt('EMPCA_norm_48365/'+savefilename, zip(lll,bnflux[i]/wpca_spectra,bnweight[i]/wpca_spectra,bnflux[i]/nmf_spectra,bnweight[i]/nmf_spectra,wpca_spectra,nmf_spectra,bnflux[i],bnweight[i]), fmt='%10.5f') 
        
        print '*****************************************************************************'
        print '                 Testing the Chi2 values'
        print '*****************************************************************************'
    
        #print 'Flux Difference: ',bnflux[i][wm] - wpca_spectra[wm]
        #print 'Weight',bnweight[i][wm]
        print np.sum((bnflux[i][wm]-wpca_spectra[wm])**2*bnweight[i][wm]),len(bnflux[i][wm])- p
        print np.sum((bnflux[i][wmn]-nmf_spectra1[wmn])**2*bnweight[i][wmn]),len(bnflux[i][wmn])- p
        #for jl in range(len(wm)):
        #    print jl,(bnflux[i][wm[jl]]-wpca_spectra[wm[jl]])**2*bnweight[i][wm[jl]],\
        #    (bnflux[i][wmn[jl]]-nmf_spectra1[wmn[jl]])**2*bnweight[i][wmn[jl]]
        print '*****************************************************************************'
    
        rchi2 = np.sum((bnflux[i][wm]-wpca_spectra[wm])**2*bnweight[i][wm])/(len(bnflux[i][wm])- p)
        rchi2s = r'EMPCA $\chi^2_\nu$: {0:4.3f}'.format(rchi2)
        nrchi2 = np.sum((bnflux[i][wmn]-nmf_spectra1[wmn])**2*bnweight[i][wmn])/(len(bnflux[i][wmn])- p)
        nrchi2s = r'NMF $\chi^2_\nu$: {0:4.3f}'.format(nrchi2)
    
        fig1,(ax1,ax2,ax3)= plt.subplots(3,1,figsize=(12,8),sharex=True)
        ax1.plot(lll,bnflux[i],color='black',alpha=0.4,label='Spectra')
        ax3.plot(lll,bnflux[i],color='black',alpha=0.4,label='Spectra')
        ax3.plot(lll,pmodel,color='red',ls='--',alpha=0.4,label='Power law')
        #ax1.plot(lll[wm],bnflux[i][wm],'.',color='green',alpha=0.6,label='Masked Spectra')
        ax2.plot(lll,bnflux[i],color='black',alpha=0.3,label='Spectra')
        #ax2.plot(lll[wm],bnflux[i][wm],'.',color='green',alpha=0.6,label='Masked Spectra')
        ax3.plot(lll[wmweight > 0],bnflux[i][wmweight > 0],'.',color='green',alpha=0.6,label='Masked Spectra')
        #ax3.plot(lll,wmweight/1e2,color='red',alpha=0.6,label='Weight')
        #ax1.plot(lll,pca_spectra,color='red',label='Model')
        #ax1.plot(lll,rpca_spectra,color='orange',label='Randomized PCA Model')
        ax1.plot(lll,wpca_spectra,color='blue',label='EMPCA Model')
        #ax2.plot(lll,nmf_spectra,color='cyan',label='Sklearn NMF Model')
        ax2.plot(lll,nmf_spectra,color='brown',label='H-NMF Model')
        #ax1.plot(lll,nmf_spectra,color='brown',label='GBZNMF Model')
        #ax1.set_xlabel(r'Wavelength')
        #ax2.set_xlabel(r'Wavelength')
        ax3.set_xlabel(r'Wavelength')
        ax1.set_ylabel(r'Flux')
        ax2.set_ylabel(r'Flux')
        ax3.set_ylabel(r'Flux')
        ax1.legend(loc=1)
        ax2.legend(loc=1)
        ax3.legend(loc=1)
        ax1.set_ylim(np.median(bnflux[i][wm] - 4*np.std(bnflux[i][wm])),np.median(bnflux[i][wm] + 15*np.std(bnflux[i][wm])))
        ax2.set_ylim(np.median(bnflux[i][wm] - 4*np.std(bnflux[i][wm])),np.median(bnflux[i][wm] + 15*np.std(bnflux[i][wm])))
        ax3.set_ylim(np.median(bnflux[i][wm] - 4*np.std(bnflux[i][wm])),np.median(bnflux[i][wm] + 15*np.std(bnflux[i][wm])))
        ylim = ax1.get_ylim()
        xlim = ax1.get_xlim()
        ylim2 = ax2.get_ylim()
        xlim2 = ax2.get_xlim()
        ax1.text(xlim[0]+0.1*(xlim[1] - xlim[0]),ylim[1]-0.2*(ylim[1] - ylim[0]),rchi2s,fontsize=18)
        ax2.text(xlim2[0]+0.1*(xlim2[1] - xlim2[0]),ylim2[1]-0.2*(ylim2[1] - ylim2[0]),nrchi2s,fontsize=18)
        #ax3.axhline(wmmed,ls='--',color='blue')
        #ax3.axhline(nmed1,ls='--',color='brown')
        titletxt= 'Training Sample: DR12q     name: {0}    RA: {1}    DEC: {2}    PLATE: {3}    MJD: {4}    FIBER: {5}    Z:{6}'.format(balinfo['name'][i],balinfo['RA'][i],balinfo['DEC'][i],balinfo['PLATE'][i],balinfo['MJD'][i],balinfo['FIBER'][i],balinfo['Z'][i])
        #comment = 'Pat\'s Note: {}'.format(balrminfo['notes'][i])
        print titletxt
        ax1.set_title(titletxt)
        ylim3 = ax3.get_ylim()
        xlim3 = ax3.get_xlim()
        #ax3.text(xlim3[0]+0.1*(xlim3[1] - xlim3[0]),ylim3[1]-0.2*(ylim3[1] - ylim3[0]),comment,fontsize=12)
        fig1.tight_layout()
        #fig1.canvas.manager.window.wm_geometry("+%d+%d" % (100,0))
        fig1.savefig(pp1,format='pdf')
        
        fig2,(bx1,bx2,bx3)= plt.subplots(3,1,figsize=(12,8),sharex=True)
        bx1.plot(lll[wmn],(bnflux[i][wmn] -nmf_spectra1[wmn])*np.sqrt(bnweight[i][wmn]),'.' ,color='red',alpha=0.3,label='H-NMF')
        bx1.plot(lll[wm],(bnflux[i][wm] -wpca_spectra[wm])*np.sqrt(bnweight[i][wm]),'+',color='black',alpha=0.9,label='EMPCA')
        bx2.plot(lll[wmn],(bnflux[i][wmn] -nmf_spectra1[wmn])*np.sqrt(bnweight[i][wmn]),'+' ,color='red',alpha=0.4,label='H-NMF')
        bx3.plot(lll,bnflux[i],color='black',alpha=0.4,label='Spectra')
        #bx3.plot(lll,pmodel,color='red',ls='--',alpha=0.4,label='Power law')
        bx3.plot(lll[wm],bnflux[i][wm],'.',color='cyan',alpha=0.4,label='Masked Spectra')
        bx3.plot(lll,wpca_spectra,color='blue',ls ='--',lw=2,label='EMPCA Model')
        bx3.plot(lll,nmf_spectra,color='red',ls=':',lw=2,label='H-NMF Model')
        bx1.axhline(-1, ls=':',alpha=0.5, color='red')
        bx1.axhline(1, ls=':',alpha=0.5, color='red')
        bx2.axhline(1, ls=':',alpha=0.5, color='red')
        bx2.axhline(-1, ls=':',alpha=0.5, color='red')
        bx3.set_ylim(np.median(bnflux[i][wm] - 4*np.std(bnflux[i][wm])),np.median(bnflux[i][wm] + 10*np.std(bnflux[i][wm])))
        ylima = bx1.get_ylim()
        xlima = bx1.get_xlim()
        ylima2 = bx2.get_ylim()
        xlima2 = bx2.get_xlim()
        bx1.legend(loc=1)
        bx2.legend(loc=1)
        bx3.legend(loc=1)
        estring = r'EMPCA--> $\Sigma \chi^2$:{0:5.3f}    dof:{1:4d}    $\chi^2_\nu$:{2:5.2f}'.format((np.sum(((bnflux[i][wm] -wpca_spectra[wm])*np.sqrt(bnweight[i][wm]))**2)),(len(bnflux[i][wm])- p), np.sum(((bnflux[i][wm] -wpca_spectra[wm])*np.sqrt(bnweight[i][wm]))**2)/(len(bnflux[i][wm])- p))
        nstring = r'NMF--> $\Sigma \chi^2$:{0:5.3f}    dof:{1:4d}    $\chi^2_\nu$:{2:5.2f}'.format((np.sum(((bnflux[i][wmn] -nmf_spectra1[wmn])*np.sqrt(bnweight[i][wmn]))**2)),(len(bnflux[i][wmn])- p), np.sum(((bnflux[i][wmn] -nmf_spectra1[wmn])*np.sqrt(bnweight[i][wmn]))**2)/(len(bnflux[i][wmn])- p))
        bx1.text(xlima[0]+0.1*(xlima[1] - xlima[0]),ylima[1]-0.2*(ylima[1] - ylima[0]),estring,fontsize=18)
        bx2.text(xlima2[0]+0.1*(xlima2[1] - xlima2[0]),ylima2[1]-0.2*(ylima2[1] - ylima2[0]),nstring,fontsize=18)
        bx3.set_xlabel(r'Wavelength')
        bx1.set_ylabel(r'N$_{\sigma}$')
        bx2.set_ylabel(r'N$_{\sigma}$')
        bx3.set_ylabel(r'Flux')
        bx1.set_title(titletxt)
        fig2.tight_layout()
        #fig2.canvas.manager.window.wm_geometry("+%d+%d" % (1500,0))
        #plt.show()
        fig2.savefig(pp2,format='pdf')
        plt.close('all')
        print>>saveChi2, '{0}\t{1:5.4f}\t{2}\t{3:4.3f}\t{4:5.4f}\t{5}\t{6:4.3f}\t{7}\t{8}\t{9}'.format(balinfo['name'][i],(np.sum(((bnflux[i][wm] -wpca_spectra[wm])*np.sqrt(bnweight[i][wm]))**2)),(len(bnflux[i][wm])- p), np.sum(((bnflux[i][wm] -wpca_spectra[wm])*np.sqrt(bnweight[i][wm]))**2)/(len(bnflux[i][wm])- p),(np.sum(((bnflux[i][wmn] -nmf_spectra1[wmn])*np.sqrt(bnweight[i][wmn]))**2)),(len(bnflux[i][wmn])- p), np.sum(((bnflux[i][wmn] -nmf_spectra1[wmn])*np.sqrt(bnweight[i][wmn]))**2)/(len(bnflux[i][wmn])- p),balinfo['PLATE'][i],balinfo['MJD'][i],balinfo['FIBER'][i])
        #    ljljlj=raw_input()
    except:
        dele = open('Unable2fitPCAfits_fullsample_48365.txt','a')
        print 'Check on writing pandas', i
        print>>dele,'Training Sample: DR12q     name: {0}    RA: {1}    DEC: {2}    PLATE: {3}    MJD: {4}    FIBER: {5}    Z:{6}'.format(balinfo['name'].iloc[i],balinfo['RA'].iloc[i],balinfo['DEC'].iloc[i],balinfo['PLATE'].iloc[i],balinfo['MJD'].iloc[i],balinfo['FIBER'].iloc[i],balinfo['Z'].iloc[i])
        dele.close()   

        
pp1.close()
pp2.close()
saveChi2.close()
savecoeff.close()

