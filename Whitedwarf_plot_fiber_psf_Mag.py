import numpy as np
import matplotlib.pyplot as plt
import glob 
import astropy.io.fits as fits
import os
import urllib2

skydata = fits.open('Skyserver_CrossIDWhitedwarfs_DR14.fits')[1].data
wdata=np.genfromtxt('Whitedwarfs_in_SDSS_with_3epochs.txt',names=['name','ra','dec','pmf1','pmf2','pmf3'],dtype=('|S10',float,float,'|S20','|S20','|S20'))


sdiff_r0 =[]; sdiff_g0=[]; sdiff_i0=[]
sdiff_r1 =[]; sdiff_g1=[]; sdiff_i1=[]
sdiff_r2 =[]; sdiff_g2=[]; sdiff_i2=[]

for i in range(len(wdata)):                                                                                                                 
	print os.path.isfile('Whitedwarf_Spectra/spec-'+wdata['pmf1'][i]+'.fits'),os.path.isfile('Whitedwarf_Spectra/spec-'+wdata['pmf2'][i]+'.fits'),os.path.isfile('Whitedwarf_Spectra/spec-'+wdata['pmf3'][i]+'.fits')     
	data0 = fits.open('Whitedwarf_Spectra/spec-'+wdata['pmf1'][i]+'.fits')[2].data
	data1 = fits.open('Whitedwarf_Spectra/spec-'+wdata['pmf2'][i]+'.fits')[2].data
	data2 = fits.open('Whitedwarf_Spectra/spec-'+wdata['pmf3'][i]+'.fits')[2].data
	#print data0['SPECTROFLUX'],data1['SPECTROFLUX'],data2['SPECTROFLUX']
        
        sx = np.where(skydata['Name'] == wdata['name'][i])[0]
        model_g=skydata['modelMag_g'][sx[0]] 
        model_r=skydata['modelMag_r'][sx[0]] 
        model_i=skydata['modelMag_i'][sx[0]]



        sfiber_r0 = 22.5 - 2.5*np.log10(data0['SPECTROFLUX'][0][2])
        sfiber_g0 = 22.5 - 2.5*np.log10(data0['SPECTROFLUX'][0][1])
        sfiber_i0 = 22.5 - 2.5*np.log10(data0['SPECTROFLUX'][0][3])

        sfiber_r1 = 22.5 - 2.5*np.log10(data1['SPECTROFLUX'][0][2])
        sfiber_g1 = 22.5 - 2.5*np.log10(data1['SPECTROFLUX'][0][1])
        sfiber_i1 = 22.5 - 2.5*np.log10(data1['SPECTROFLUX'][0][3])
    
        sfiber_r2 = 22.5 - 2.5*np.log10(data2['SPECTROFLUX'][0][2])
        sfiber_g2 = 22.5 - 2.5*np.log10(data2['SPECTROFLUX'][0][1])
        sfiber_i2 = 22.5 - 2.5*np.log10(data2['SPECTROFLUX'][0][3])
        print 'G-mag',model_g,sfiber_g0,sfiber_g1,sfiber_g2
        print 'R-mag',model_r,sfiber_r0,sfiber_r1,sfiber_r2
        print 'I-mag',model_i,sfiber_i0,sfiber_i1,sfiber_i2

        sdiff_r0.append(model_r - sfiber_r0)
    
        sdiff_g0.append(model_g - sfiber_g0)
    
        sdiff_i0.append(model_i - sfiber_i0)

        sdiff_r1.append(model_r - sfiber_r1)
    
        sdiff_g1.append(model_g - sfiber_g1)
    
        sdiff_i1.append(model_i - sfiber_i1)

        sdiff_r2.append(model_r - sfiber_r2)
    
        sdiff_g2.append(model_g - sfiber_g2)
    
        sdiff_i2.append(model_i - sfiber_i2)


sdiff_g0=np.array(sdiff_g0);sdiff_r0=np.array(sdiff_r0);sdiff_i0=np.array(sdiff_i0)
sdiff_g1=np.array(sdiff_g1);sdiff_r1=np.array(sdiff_r1);sdiff_i1=np.array(sdiff_i1)
sdiff_g2=np.array(sdiff_g2);sdiff_r2=np.array(sdiff_r2);sdiff_i2=np.array(sdiff_i2)
sdiff_g0 =sdiff_g0[np.isfinite(sdiff_g0)];sdiff_r0 =sdiff_r0[np.isfinite(sdiff_r0)];sdiff_i0 =sdiff_i0[np.isfinite(sdiff_i0)]
sdiff_g1 =sdiff_g1[np.isfinite(sdiff_g1)];sdiff_r1 =sdiff_r1[np.isfinite(sdiff_r1)];sdiff_i1 =sdiff_i1[np.isfinite(sdiff_i1)]
sdiff_g2 =sdiff_g2[np.isfinite(sdiff_g2)];sdiff_r2 =sdiff_r2[np.isfinite(sdiff_r2)];sdiff_i2 =sdiff_i2[np.isfinite(sdiff_i2)]


fig,((bx0,bx,bx1)) = plt.subplots(1,3,figsize=(20,6))
bins = np.linspace(-1.5,1.,20)






bx0.hist(sdiff_g0,bins=bins,histtype='step',color='black',lw=3,label='SDSS' )
bx0.hist(sdiff_g1,bins=bins,histtype='step',color='red',lw=3,label='BOSS' )
bx0.hist(sdiff_g2,bins=bins,histtype='step',color='blue',lw=3,label='eBOSS' )
bx0.set_xlabel('Magnitude Difference g$_{model}$ - g$_{fiber}$')
bx0.set_ylabel('Histogram Density')
bx0.axvline(np.median(sdiff_g0),ls='--',color='black')
bx0.axvline(np.median(sdiff_g1),ls='--',color='red')
bx0.axvline(np.median(sdiff_g2),ls='--',color='blue')
bstring_g = 'median $\Delta$g(Phot - SDSS)  : {0:4.3f} \nmedian $\Delta$g(Phot - BOSS)  : {1:4.3f}\nmedian $\Delta$g(Phot - eBOSS) : {2:4.3f} '.format(np.median(sdiff_g0),np.median(sdiff_g1),np.median(sdiff_g2))
print bstring_g

bx0.text(-1.45,120,bstring_g,fontsize=10)
bx0.legend(loc=1)

bx.hist(sdiff_r0,bins=bins,histtype='step',color='black',lw=3,label='SDSS' )
bx.hist(sdiff_r1,bins=bins,histtype='step',color='red',lw=3,label='BOSS' )
bx.hist(sdiff_r2,bins=bins,histtype='step',color='blue',lw=3,label='eBOSS' )
bx.set_xlabel('Magnitude Difference r$_{model}$ - r$_{fiber}$')
bx.set_ylabel('Histogram Density')
bstring_r = 'median $\Delta$r(Phot - SDSS)  : {0:4.3f} \nmedian $\Delta$r(Phot - BOSS)  : {1:4.3f}\nmedian $\Delta$r(Phot - eBOSS) : {2:4.3f} '.format(np.median(sdiff_r0),np.median(sdiff_r1),np.median(sdiff_r2))
print bstring_r
bx.axvline(np.median(sdiff_r0),ls='--',color='black')
bx.axvline(np.median(sdiff_r1),ls='--',color='red')
bx.axvline(np.median(sdiff_r2),ls='--',color='blue')
bx.text(-1.45,120,bstring_r,fontsize=10)
bx.set_title("219 Whitedwarf spectra in SDSS, BOSS, eBOSS")

bx.legend(loc=1)

bx1.hist(sdiff_i0,bins=bins,histtype='step',color='black',lw=3,label='SDSS' )
bx1.hist(sdiff_i1,bins=bins,histtype='step',color='red',lw=3,label='BOSS' )
bx1.hist(sdiff_i2,bins=bins,histtype='step',color='blue',lw=3,label='eBOSS' )
bx1.set_xlabel('Magnitude Difference i$_{model}$ - i$_{fiber}$')
bx1.set_ylabel('Histogram Density')
bstring_i = 'median $\Delta$i(Phot - SDSS)  : {0:4.3f} \nmedian $\Delta$i(Phot - BOSS)  : {1:4.3f}\nmedian $\Delta$i(Phot - eBOSS) : {2:4.3f} '.format(np.median(sdiff_i0),np.median(sdiff_i1),np.median(sdiff_i2))
print bstring_i
bx1.axvline(np.median(sdiff_i0),ls='--',color='black')
bx1.axvline(np.median(sdiff_i1),ls='--',color='red')
bx1.axvline(np.median(sdiff_i2),ls='--',color='blue')
bx1.text(-1.45,120,bstring_i,fontsize=10)

bx1.legend(loc=1)

fig.tight_layout()
plt.show()
