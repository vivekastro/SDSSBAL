import numpy as np
import astropy.io.fits as fits
import scipy as sp
import matplotlib.pyplot as plt
import astroML 
from matplotlib.ticker import NullFormatter
import pandas as pd
import os,glob
import urllib2
from scipy import   stats
from pydl.pydl.pydlutils.spheregroup import *
from astroML.plotting import hist
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from astropy.modeling.models import Voigt1D
from astropy import constants as const
from astropy import units as U
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
from specutils import extinction 
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pydl.pydl.pydlutils import yanny
from astropy.table import Table
from astropy.time import Time
from scipy.ndimage.filters import gaussian_filter1d,uniform_filter1d
import palettable
import sfdmap
from PyAstronomy import pyasl
"""
Program to explore the full eBOSS BALQSO sample. 
@author : Vivek M.
@date   : 22/Feb/2019
@version: 1.0
"""
spallversion='v5_13_0'
params = {
   'axes.labelsize': 18,
   'axes.linewidth': 1.5,
   #'text.fontsize': 8,
   'legend.fontsize': 15,
   'xtick.labelsize': 18,
   'ytick.labelsize': 18,
   'text.usetex': True,
   #'figure.figsize': [16, 5]
   'legend.frameon': False,
   'font.family': 'Times New Roman'
   }
plt.rcParams.update(params)
c15_1=palettable.tableau.Tableau_20.mpl_colors
hfont = {'fontname':'Times New Roman'}
def plot_hline(y,**kwargs):
    data = kwargs.pop("data") #get the data frame from the kwargs
    plt.axhline(y=y, c='black',linestyle='--',zorder=-1) #zorder places the line underneath the other points

n_v = 1240.81
si_iv = 1393.755
c_iv = 1549.48
al_iii = 1857.4
mg_ii = 2799.117
p_v = 1122.9925#1128.008#1117.977 #
def negativeRAs(ralist):
    newralist=[]
    for ra in ralist:
        if ra >= 300 :
            t=ra - 360.0
            ra = t
        newralist.append(ra)
    return newralist
def _DeRedden(lam,flux,ra,dec,dustmap_path='/Users/vzm83/Softwares/sfddata-master'):
        """Correct the Galatical extinction"""          
        m = sfdmap.SFDMap(dustmap_path) 
        flux_unred = pyasl.unred(lam,flux,m.ebv(ra,dec))
        return flux_unred

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


def download_spectra(plate, mjd, fiber, dirname='.'):
    '''  Downloads SDSS spectra from DR14 and puts it in dirname
         Change the SDSS URL to download from a different location
    '''
    FITS_FILENAME = 'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits'
    try :
        SDSS_URL = ('https://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/%(plate)04i/'
                'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits')
        urllib2.urlopen(SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber))
        print 'Downloadin from dr14'
    except urllib2.HTTPError as err:
        if err.code == 404 :
            SDSS_URL = ('https://data.sdss.org/sas/dr8/sdss/spectro/redux/26/spectra/%(plate)04i/'
            		'spec-%(plate)04i-%(mjd)05i-%(fiber)04i.fits')
            print 'Downloadin from dr8'
    print SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
    download_url = 'wget   '+SDSS_URL % dict(plate=plate,mjd=mjd,fiber=fiber)
    print download_url
    os.system(download_url)
    mv_cmd='mv '+FITS_FILENAME % dict(plate=plate,mjd=mjd,fiber=fiber) + ' '+dirname+'/.'
    #print mv_cmd
    os.system(mv_cmd)


def onclick(event):
    '''
    Record the points by clicking on the plots.
    '''
    global ix, iy
    ix, iy = event.xdata, event.ydata

    print 'x = %d, y = %d'%(ix, iy)

    # assign global variable to access outside of function
    #global coords
    #coords.append((ix, iy))

    # Disconnect after 20 clicks
    #if len(coords) == 20:
   #     fig.canvas.mpl_disconnect(cid)
   #     plt.close(1)
    return (ix,iy)


def ray_tracing_method(x,y,poly):
    '''
    Check if the point is inside the polygon
    '''
    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside







def merge2initialSamples():
    
    #For Old file used for SDSS-III containing 2109 tatgets
    #baltargets = yanny.read_table_yanny(filename='master-BAL-targets-yanny-format1.dat.txt',tablename='TARGET')

    #For SDSS-IV USe the following file containing 2958 sources
    baltargets = yanny.read_table_yanny(filename='green01-TDSS_FES_VARBALmaster1.par.txt',tablename='TARGET')
    newtargets=yanny.read_table_yanny('targeting13-explained_more_TDSS_FES_VARBAL_201605.dat',tablename='TARGET')
    out = open('Master_initial_sample.txt','w') 
    baltargetsra = np.concatenate((baltargets['ra'],newtargets['ra']))
    baltargetsdec = np.concatenate((baltargets['dec'],newtargets['dec']))
    for i in range(len(baltargetsra)):
        print>>out, 'Target{0:04d}\t{1:10.5f}\t{2:10.5f}'.format(i+1,baltargetsra[i],baltargetsdec[i])
    print 'Initial sample has {} sources; {} from initial and {} from later'.format(len(baltargetsra),len(baltargets),len(newtargets))
    print 'Initial sample has unique {} sources; {} from initial and {} from later'.format(len(np.unique(baltargetsra)),len(np.unique(baltargets)),len(np.unique(newtargets)))
    
    out.close()

def getMulti_epochInfo():

    #odata = fits.open('Skyserver_CrossID_DR14_gibson.fits')[1].data
    #ndata = fits.open('topcatMatch_spAll_v5_13_0_gibson_BAL_sample.fits')[1].data
    #master = np.genfromtxt('Master_gibson_2005_targets_cor.txt',names=['name','ra','dec'],dtype=('|S15',float,float),skip_header=1)
    
    odata = fits.open('Skyserver_CrossID_DR14_master.fits')[1].data
    ndata = fits.open('topcatMatch_spAll_v5_13_0_master_BAL_sample_allmatches.fits')[1].data
    master = np.genfromtxt('Master_initial_sample.txt',names=['name','ra','dec'],dtype=('|S15',float,float),skip_header=1)

    masterpmf = [] ; masterplate = []; mastermjd = []; masterfiber=[];mastername = [] ; masterra=[]; masterdec=[]
    for i in range(len(master)):
    #for i in range(20):
        xx = np.where(ndata['col1'] == master['name'][i])[0]
        
        #yy = np.where(odata['col1'] == master['name'][i])[0]
        yy = np.where(odata['Name'] == master['name'][i])[0] #******CHANGE HERE for 3028***********


        plate = [] ; mjd = []; fiber = []
        plate_mjd_fiber = []

        if len(yy) > 0 :
            yodata = odata[yy]
            for j in range(len(yodata)):
                pmf = '{0:05d}-{1:05d}-{2:04d}'.format(yodata['plate'][j],yodata['mjd'][j],yodata['fiberID'][j])
                if pmf not in plate_mjd_fiber :
                    plate_mjd_fiber.append(pmf)
                    plate.append(yodata['plate'][j])
                    mjd.append(yodata['mjd'][j])
                    fiber.append(yodata['fiberID'][j])

        if len(xx) > 0 :
            xndata = ndata[xx]
            for k in range(len(xndata)):
                pmf = '{0:05d}-{1:05d}-{2:04d}'.format(xndata['PLATE'][k],xndata['MJD'][k],xndata['FIBERID'][k])
                if pmf not in plate_mjd_fiber :
                    plate_mjd_fiber.append(pmf)
                    plate.append(xndata['PLATE'][k])
                    mjd.append(xndata['MJD'][k])
                    fiber.append(xndata['FIBERID'][k])
        if ((master['name'][i] == 'nTarget0106') |(master['name'][i] ==  'Target0453')) :
             pmf = '{0:05d}-{1:05d}-{2:04d}'.format(707,52177,392)
             if pmf not in plate_mjd_fiber :
                plate_mjd_fiber.append(pmf)
                plate.append(707)
                mjd.append(52177)
                fiber.append(392)
        if ((master['name'][i] == 'nTarget0164') |(master['name'][i] ==  'Target0517')) :
             pmf = '{0:05d}-{1:05d}-{2:04d}'.format(431,51877,550)
             if pmf not in plate_mjd_fiber :
                plate_mjd_fiber.append(pmf)
                plate.append(431)
                mjd.append(51877)
                fiber.append(550)
        if ((master['name'][i] == 'nTarget1984') |(master['name'][i] ==  'Target2870')) :
            pmf = '{0:05d}-{1:05d}-{2:04d}'.format(645,52203,480)
            if pmf not in plate_mjd_fiber :
                plate_mjd_fiber.append(pmf)
                plate.append(645)
                mjd.append(52203)
                fiber.append(480)
        if ((master['name'][i] == 'nTarget0153') |(master['name'][i] ==  'Target0505')) :
            pmf = '{0:05d}-{1:05d}-{2:04d}'.format(1865,53312,261)
            if pmf not in plate_mjd_fiber :
                plate_mjd_fiber.append(pmf)
                plate.append(1865)
                mjd.append(53312)
                fiber.append(261)
        if ((master['name'][i] == 'nTarget0346') |(master['name'][i] ==  'Target0731')) :
            pmf = '{0:05d}-{1:05d}-{2:04d}'.format(467,51901,192)
            if pmf not in plate_mjd_fiber :
                plate_mjd_fiber.append(pmf)
                plate.append(467)
                mjd.append(51901)
                fiber.append(192)
        if ((master['name'][i] == 'nTarget0774') |(master['name'][i] ==  'Target1230')) :
            pmf = '{0:05d}-{1:05d}-{2:04d}'.format(1002,52646,297)
            if pmf not in plate_mjd_fiber :
                plate_mjd_fiber.append(pmf)
                plate.append(1002)
                mjd.append(52646)
                fiber.append(297)


        if len(mjd)>0:
            print '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(i+1,master['name'][i],len(xx),len(yy),min(mjd),max(mjd),max(plate),len(plate))
            masterpmf.append(plate_mjd_fiber)
            masterplate.append(plate)
            mastermjd.append(mjd)
            masterfiber.append(fiber)
            mastername.append(master['name'][i])
            masterra.append(master['ra'][i])
            masterdec.append(master['dec'][i])
        if len(plate)>0:
            print plate_mjd_fiber#,masterpmf
    data = {'name': np.array(master['name']),'pmf': np.array(masterpmf),'plate':np.array(masterplate),'mjd':np.array(mastermjd), 'fiber':np.array(masterfiber)}
    #pdata = Table(data)
    print len(data)
    #c1 = fits.Column(name='name', array=np.array(master['name']), format='15A')
    #c2 = fits.Column(name='pmf', array=np.array(masterpmf,dtype=np.object), format='PA(16)')
    #c3 = fits.Column(name='plate', array=np.array(masterplate,dtype=np.object), format='PI()')
    #c4 = fits.Column(name='mjd', array=np.array(mastermjd,dtype=np.object), format='PI()')
    #c5 = fits.Column(name='fiber', array=np.array(masterfiber,dtype=np.object), format='PI()')
    #tfits = fits.BinTableHDU.from_columns([c1,c2,  c3, c4, c5])
    #tfits.writeto('MasterList_Plate-MJD-Fiber.fits')
    #np.savez('MasterList_Plate-MJD-Fiber_2005.npz',
    #        name = np.array(mastername) ,
    #        ra = np.array(masterra) ,
    #        dec = np.array(masterdec) ,
    #        pmf = np.array(masterpmf) ,
    #        plate = np.array(masterplate) ,
    #        mjd = np.array(mastermjd) ,
    #        fiber = np.array(masterfiber) ,
    #        )
    #data = np.load('MasterList_Plate-MJD-Fiber_2005.npz')
    #out = open('Master_multi-epoch_information_numberofEpochs_v5_13_2005.txt','w')
    #*********** CHANGE HERE for 3028
    np.savez('MasterList_Plate-MJD-Fiber.npz',
            name = np.array(mastername) ,
            ra = np.array(masterra) ,
            dec = np.array(masterdec) ,
            pmf = np.array(masterpmf) ,
            plate = np.array(masterplate) ,
            mjd = np.array(mastermjd) ,
            fiber = np.array(masterfiber) ,
            )
#
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    out = open('Master_multi-epoch_information_numberofEpochs_v5_13.txt','w')
    for i in range(len(data['ra'])):
        print>>out, '{0}\t{1:10.5f}\t{2:10.5f}\t{3}'.format(data['name'][i],data['ra'][i],data['dec'][i],len(data['mjd'][i]))
    out.close()

def plot_nepochs():
    #master = np.genfromtxt('Master_gibson_2005_targets_cor.txt',names=['name','ra','dec'],dtype=('|S15',float,float),skip_header=1)
    #msum = np.genfromtxt('Master_multi-epoch_information_numberofEpochs_v5_13_2005.txt',names=['name','ra','dec','nepochs'])
    
    master = np.genfromtxt('Master_initial_sample.txt',names=['name','ra','dec'],dtype=('|S15',float,float),skip_header=1)
    msum = np.genfromtxt('Master_multi-epoch_information_numberofEpochs_v5_13.txt',names=['name','ra','dec','nepochs'])
    
    ra1,dec1=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_NGC.txt').T
    ra2,dec2=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_SGC.txt').T
    
    fig,ax = plt.subplots(figsize=(10,8))
    x1 = np.where(msum['nepochs'] == 1)[0]
    x2 = np.where(msum['nepochs'] == 2)[0]
    x3 = np.where(msum['nepochs'] >= 3)[0]
    print len(x1),len(x2)
    ax.plot(ra1,dec1,'-',color='black',alpha=0.5)
    ax.plot(ra2,dec2,'-',color='black',alpha=0.5)
    ax.plot(negativeRAs(master['ra']),master['dec'],'.',markersize=3,color='black',label='Parent Sample'+'(\#'+str(len(master['ra']))+')')#label='Parent Sample(\#2005)')#
    ax.plot(negativeRAs(msum['ra'][x1]),msum['dec'][x1],'o',color='red',markersize=3,label='1 epoch'+'(\#'+str(len(x1))+')')
    ax.plot(negativeRAs(msum['ra'][x2]),msum['dec'][x2],'s',color='blue',markersize=3,label='2  epochs'+'(\#'+str(len(x2))+')')
    ax.plot(negativeRAs(msum['ra'][x3]),msum['dec'][x3],'s',color='magenta',markersize=3,label='3 or more epochs'+'(\#'+str(len(x3))+')')
    ax.set_xlim(-55,300)
    ax.set_ylim(-15,100)
    ax.legend(loc=1)
    ax.grid()
    xlim,ylim=ax.get_xlim(),ax.get_ylim()
    throughdate=Time(58528,format='mjd')
    throughdate.format = 'fits'
    print throughdate.value[0:10]
    ax.text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.05*(ylim[1] - ylim[0]),'Through '+str(throughdate.value[0:10]),fontsize=20)

    ax.set(xlabel='RA',ylabel='DEC')
    fig.tight_layout()
    #fig.savefig('SDSS_IV_BALquasar_sample_nepochs_plot_v5_13_2005.jpg')
    fig.savefig('SDSS_IV_BALquasar_sample_nepochs_plot_v5_13.jpg')
   #plt.show()
    

def download_data(print_cmd=False):
    if print_cmd :
        #cptxt = open('copyspectrumfromBOSS_SPECTRO_REDUX_v5_13_2005.txt','w')
        cptxt = open('copyspectrumfromBOSS_SPECTRO_REDUX_v5_13.txt','w')
    else:
        checkdownload=[]
    #data = np.load('MasterList_Plate-MJD-Fiber_2005.npz')
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    specdir = 'SDSSIV_BALdata'
    for i in range(len(data['name'])):
    #for i in range(30):
        plates = data['plate'][i];mjds=data['mjd'][i];fibers=data['fiber'][i]
        for j in range(len(plates)):
            print data['name'][i], plates[j],mjds[j],fibers[j]
            plate = plates[j]; mjd = mjds[j]; fiber = fibers[j]
            if plate >= 10000:
                FITS_FILENAME = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(plate,mjd,fiber)
            else:
                FITS_FILENAME = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(plate,mjd,fiber)
            if not print_cmd:
                if not ((os.path.isfile(FITS_FILENAME)) | (os.path.isfile(os.path.join(specdir,FITS_FILENAME)))):
                    download_spectra(plates[j],mjds[j],fibers[j],specdir)
                else : 
                    print 'Spectrum already downloaded'

            if print_cmd:
                print>>cptxt, 'cp   /uufs/chpc.utah.edu/common/home/sdss/ebosswork/eboss/spectro/redux/{0}/spectra/lite/{1}/{2}  ~/BOSS_BALDATA/.'.format(spallversion,plate,FITS_FILENAME)        
            else :
                if not ((os.path.isfile(FITS_FILENAME)) & (os.path.isfile(os.path.join(specdir,FITS_FILENAME)))):
                    checkdownload.append(FITS_FILENAME)
            print '--'*31
    print checkdownload
    if print_cmd:
        cptxt.close()

def hist_multiepochs():
    #data = np.genfromtxt('Master_multi-epoch_information_numberofEpochs_v5_13_2005.txt',names=['name','ra','dec','nepochs'],dtype=('|S15',float,float,int))
    data = np.genfromtxt('Master_multi-epoch_information_numberofEpochs_v5_13.txt',names=['name','ra','dec','nepochs'],dtype=('|S15',float,float,int))
    fig,ax=plt.subplots(figsize=(5,5))
    bins=np.arange(1,11,1)
    hist(data['nepochs'],normed=False,bins=bins,histtype="step", linewidth= 2,alpha= 1, linestyle="-",color='red',align='mid', ax=ax)
    #ax.set(xlabel='Number of epochs',ylabel='Number of quasars',title='SDSS-IV: Initial sample 2005 sources')
    ax.set(xlabel='Number of epochs',ylabel='Number of quasars',title='SDSS-IV: Full sample 3014 sources')
    ax.grid()
    fig.tight_layout()
    #fig.savefig('eBOSS_fullsample_number_epochs_hist_v5_13_2005.jpg')
    fig.savefig('eBOSS_fullsample_number_epochs_hist_v5_13.jpg')
    #plt.show()

def cumulative_MJD():
    #data = np.load('MasterList_Plate-MJD-Fiber_2005.npz')
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    SDSSIV_MJDs = []; uSDSSIV_MJDs = []
    for i in range(len(data['name'])):
        tmjd = np.array(data['mjd'][i])
        xx=np.where(tmjd > 56777)[0]
        #print tmjd[xx]
        if len(xx) >0:
            uSDSSIV_MJDs.append(min(tmjd[xx]))
            for j in xx:
                print i,j,tmjd,xx
                SDSSIV_MJDs.append(tmjd[j])

    SDSSIV_MJDs = np.array(SDSSIV_MJDs)
    uSDSSIV_MJDs = np.array(uSDSSIV_MJDs)

    mjdbounds = [56870,57235,57601,57966,58331,58543]
    yr0 = np.where((SDSSIV_MJDs > 56777) & (SDSSIV_MJDs <= 56870))[0]
    yr1 = np.where((SDSSIV_MJDs > 56870) & (SDSSIV_MJDs <= 57235))[0]
    yr2 = np.where((SDSSIV_MJDs > 57235) & (SDSSIV_MJDs <= 57601))[0]
    yr3 = np.where((SDSSIV_MJDs > 57601) & (SDSSIV_MJDs <= 57966))[0]
    yr4 = np.where((SDSSIV_MJDs > 57966) & (SDSSIV_MJDs <= 58331))[0]
    yr5 = np.where((SDSSIV_MJDs > 58331) & (SDSSIV_MJDs <= 58543))[0]

    uyr0 = np.where((uSDSSIV_MJDs > 56777) & (uSDSSIV_MJDs <= 56870))[0]
    uyr1 = np.where((uSDSSIV_MJDs > 56870) & (uSDSSIV_MJDs <= 57235))[0]
    uyr2 = np.where((uSDSSIV_MJDs > 57235) & (uSDSSIV_MJDs <= 57601))[0]
    uyr3 = np.where((uSDSSIV_MJDs > 57601) & (uSDSSIV_MJDs <= 57966))[0]
    uyr4 = np.where((uSDSSIV_MJDs > 57966) & (uSDSSIV_MJDs <= 58331))[0]
    uyr5 = np.where((uSDSSIV_MJDs > 58331) & (uSDSSIV_MJDs <= 58543))[0]
    print "No. of quasars:",len(uyr1),len(uyr2),len(uyr3),len(uyr4),len(uyr5)
    print "No. of spectra:",len(yr1),len(yr2),len(yr3),len(yr4),len(yr5)
    print "No. of  quasars in sequels (56777-56870)",len(uyr0)
    print "No. of  spectra in sequels (56777-56870)",len(yr0)
    fig,ax=plt.subplots(figsize=(10,5))
    bins = np.arange(56820,58600)
    hist(SDSSIV_MJDs,normed=False,bins='knuth',histtype="step", linewidth= 2,alpha= 1, linestyle="-",color='red',align='mid', ax=ax)
    hist(SDSSIV_MJDs,normed=False,bins=bins,cumulative=True,histtype="step", linewidth= 3,alpha= 0.5, linestyle="--",color='blue',align='mid', ax=ax,label='spectra')
    hist(uSDSSIV_MJDs,normed=False,bins=bins,cumulative=True,histtype="step", linewidth= 3,alpha= 0.5, linestyle=":",color='magenta',align='mid', ax=ax,label='quasars')
    #ax.set(xlabel='MJDs',ylabel='Number of spectra',title='SDSS-IV rate of data collection: Initial 2005 sample')
    ax.set(xlabel='MJDs',ylabel='Number of spectra',title='SDSS-IV rate of data collection: Full 3014 sample')
    for mb in mjdbounds:
        ax.axvline(mb,ls=':',linewidth=3)
    xlim,ylim=ax.get_xlim(),ax.get_ylim()
    ax.text(56870+0.4*(57235-56870),ylim[1]-0.1*(ylim[1] - ylim[0]),'Year1')
    ax.text(57235+0.4*(57601-57235),ylim[1]-0.1*(ylim[1] - ylim[0]),'Year2')
    ax.text(57601+0.4*(57966-57601),ylim[1]-0.1*(ylim[1] - ylim[0]),'Year3')
    ax.text(57966+0.4*(58331-57966),ylim[1]-0.1*(ylim[1] - ylim[0]),'Year4')
    ax.text(58331+0.4*(58543-58331),ylim[1]-0.1*(ylim[1] - ylim[0]),'Year5')
    ax.text(56870+0.4*(57235-56870),ylim[1]-0.15*(ylim[1] - ylim[0]),str(len(yr1)))
    ax.text(57235+0.4*(57601-57235),ylim[1]-0.15*(ylim[1] - ylim[0]),str(len(yr2)))
    ax.text(57601+0.4*(57966-57601),ylim[1]-0.15*(ylim[1] - ylim[0]),str(len(yr3)))
    ax.text(57966+0.4*(58331-57966),ylim[1]-0.15*(ylim[1] - ylim[0]),str(len(yr4)))
    ax.text(58331+0.4*(58543-58331),ylim[1]-0.15*(ylim[1] - ylim[0]),str(len(yr5)))

    ax.text(56750+0.05*(58543-58331),ylim[1]-0.25*(ylim[1] - ylim[0]),'Spectra',color='blue',alpha=0.5,fontsize=18)
    ax.text(56750+0.05*(58543-58331),ylim[1]-0.31*(ylim[1] - ylim[0]),'Quasars',color='magenta',alpha=0.5,fontsize=18)
    #fig.legend(loc=2)
    fig.tight_layout()
    #fig.savefig('eBOSS_fullsample_rate_of_data_hist_2005.jpg')
    fig.savefig('eBOSS_fullsample_rate_of_data_hist.jpg')
    #plt.show()

def plot_spPlate():
    sp = fits.open('spAll-v5_13_0.fits')[1].data
    data = fits.open('spAll-v5_10_10.fits')[1].data
    yy=np.where(data['PROGRAMNAME'] == 'sequels')[0]
    sequels = data[yy]

    ra1,dec1=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_NGC.txt').T
    ra2,dec2=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_SGC.txt').T
    elgngc = np.where(sp['PROGRAMNAME'] == 'ELG_NGC')[0]
    elgsgc = np.where(sp['PROGRAMNAME'] == 'ELG_SGC')[0]
    eboss = np.where(sp['PROGRAMNAME'] == 'eboss')[0]
    speboss = sp[eboss]
    g = np.where(speboss['CLASS'] == 'GALAXY')[0]
    g1 = np.where(sequels['CLASS'] == 'GALAXY')[0]
    q = np.where(speboss['CLASS'] == 'QSO')[0]
    q1 = np.where(sequels['CLASS'] == 'QSO')[0]
    
    fig,(ax,ax1,ax2) = plt.subplots(1,3,figsize=(20,8))
    ax.plot(ra1,dec1,'-',color='black',alpha=0.5)
    ax.plot(ra2,dec2,'-',color='black',alpha=0.5)
    ax1.plot(ra1,dec1,'-',color='black',alpha=0.5)
    ax1.plot(ra2,dec2,'-',color='black',alpha=0.5)
    
    ax2.plot(ra1,dec1,'-',color='black',alpha=0.5)
    ax2.plot(ra2,dec2,'-',color='black',alpha=0.5)
    
 
    ax.plot(negativeRAs(speboss['RA'][g]),speboss['DEC'][g],'.',markersize=3,color='black',label='GALAXY(\#'+str(len(g)+len(g1))+')')
    ax.plot(negativeRAs(sequels['RA'][g1]),sequels['DEC'][g1],'.',markersize=3,color='black')
    ax1.plot(negativeRAs(speboss['RA'][q]),speboss['DEC'][q],'.',markersize=3,color='red',label='QSO(\#'+str(len(q)+len(q1))+')')
    ax1.plot(negativeRAs(sequels['RA'][q1]),sequels['DEC'][q1],'.',markersize=3,color='red')
    ax2.plot(negativeRAs(sp['RA'][elgngc]),sp['DEC'][elgngc],'.',markersize=3,color='blue',alpha=0.3,label='ELG(\#'+str(len(elgngc)+len(elgsgc))+')')
    ax2.plot(negativeRAs(sp['RA'][elgsgc]),sp['DEC'][elgsgc],'.',markersize=3,color='blue',alpha=0.3)
    ax.set(xlabel='RA',ylabel='DEC',title='eBOSS Galaxy footprint')
    ax.set_xlim(-55,300)
    ax.set_ylim(-15,100)
    ax.legend(loc=1)
    ax.grid()
    xlim,ylim=ax.get_xlim(),ax.get_ylim()
    throughdate=Time(58528,format='mjd')
    throughdate.format = 'fits'
    print throughdate.value[0:10]
    ax.text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.05*(ylim[1] - ylim[0]),'Through '+str(throughdate.value[0:10]),fontsize=20)
    
    ax1.set(xlabel='RA',ylabel='DEC',title='eBOSS QSO footprint')
    ax1.set_xlim(-55,300)
    ax1.set_ylim(-15,100)
    ax1.legend(loc=1)
    ax1.grid()
    xlim,ylim=ax1.get_xlim(),ax1.get_ylim()
    throughdate=Time(58528,format='mjd')
    throughdate.format = 'fits'
    print throughdate.value[0:10]
    ax1.text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.05*(ylim[1] - ylim[0]),'Through '+str(throughdate.value[0:10]),fontsize=20)

    ax2.set(xlabel='RA',ylabel='DEC',title='eBOSS ELG footprint')
    ax2.set_xlim(-55,300)
    ax2.set_ylim(-15,100)
    ax2.legend(loc=1)
    ax2.grid()
    xlim,ylim=ax2.get_xlim(),ax2.get_ylim()
    throughdate=Time(58528,format='mjd')
    throughdate.format = 'fits'
    print throughdate.value[0:10]
    ax2.text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.05*(ylim[1] - ylim[0]),'Through '+str(throughdate.value[0:10]),fontsize=20)

    fig.tight_layout()

    fig.savefig('SDSSIV_eBOSS_final_sample_footprint.jpg')
    #plt.show()

def three_epochs_count():
    data = np.load('MasterList_Plate-MJD-Fiber_2005.npz')
    out = open('Fullsample_3epoch_counts_2005.txt','w')
    #data = np.load('MasterList_Plate-MJD-Fiber.npz')
    #out = open('Fullsample_3epoch_counts.txt','w')
    for i in range(len(data['name'])):
        tmjd = np.array(data['mjd'][i])
        s1 = np.where((tmjd < 55176 ))[0]
        s2 = np.where((tmjd >= 55176 ) & (tmjd <= 56777))[0]
        s3 = np.where((tmjd > 56777 ))[0]
        
        if len(s1) > 0 :
            sdss1 = len(s1)
        else :
            sdss1 = 0
        if len(s2) > 0 :
            sdss2 = len(s2)
        else :
            sdss2 = 0
        if len(s3) > 0 :
            sdss3 = len(s3)
        else :
            sdss3 = 0

        print>>out,'{0}\t{1:10}\t{2:10}\t{3:10}\t{4:10}\t{5:10}'.format(data['name'][i],data['ra'][i],data['dec'][i],sdss1,sdss2,sdss3)
    out.close()

    mcount = np.genfromtxt('Fullsample_3epoch_counts_2005.txt',names=['name','ra','dec','sdss1','sdss2','sdss3'],dtype=('|S15',float,float,int,int,int))
    #mcount = np.genfromtxt('Fullsample_3epoch_counts.txt',names=['name','ra','dec','sdss1','sdss2','sdss3'],dtype=('|S15',float,float,int,int,int))
    xx111 = np.where((mcount['sdss1'] > 0) & (mcount['sdss2'] > 0) & (mcount['sdss3']>0))[0]
    xx110 = np.where((mcount['sdss1'] > 0) & (mcount['sdss2'] > 0) & (mcount['sdss3']==0))[0]
    xx101 = np.where((mcount['sdss1'] > 0) & (mcount['sdss2'] == 0) & (mcount['sdss3']>0))[0]
    xx011 = np.where((mcount['sdss1'] == 0) & (mcount['sdss2'] > 0) & (mcount['sdss3']>0))[0]
    print 'Sources with atleast 1 epoch each in SDSS-I/II, SDSS-III, SDSS-IV',len(xx111)
    print 'Sources with atleast 1 epoch in SDSS-I/II, SDSS-III, and no SDSS-IV',len(xx110)
    print 'Sources with atleast 1 epoch in SDSS-I/II, SDSS-IV, and no SDSS-III',len(xx101)
    print 'Sources with atleast 1 epoch in SDSS-III, SDSS-IV, and no SDSS-I/II',len(xx011)


def restFrame_timescales():
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))#fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    psm = np.genfromtxt('Master_gibson_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))#fits.open('Initialsample_2005_BAL_sources_crossmatch-DR12Q.fits')[1].data
    data = np.load('MasterList_Plate-MJD-Fiber.npz')    
    data1 = np.load('MasterList_Plate-MJD-Fiber_2005.npz')  

    Flbal = fits.open('LoBAL_sample_fullsample.fits')[1].data
    Ilbal = fits.open('LoBAL_sample_initialsample.fits')[1].data
    bins = np.arange(0,14,0.25)
    fmaxt = [] ;pmaxt=[] ;fmint=[] ; pmint=[] ;fzvi=[] ; pzvi=[] ;fhb=[];phb=[];flb=[];plb=[]
    for i in range(len(fsm)):
        xx=np.where(data['name'] == fsm['name'][i])[0]
        #fdata= data[xx]
        if len(xx) > 0:
            tmjd = sorted(np.array(data['mjd'][xx]))[0]
            if len(tmjd) > 1:
                maxtime = (max(tmjd) - min(tmjd))/(1.0+fsm['Z_VI'][i])/365.
                mintime = (tmjd[1] - tmjd[0])/(1.0+fsm['Z_VI'][i])/365.
                fmaxt.append(maxtime)
                fmint.append(mintime)
                fzvi.append(fsm['Z_VI'][i])
                if fsm['name'][i] in Flbal['col1']:
                    flb.append(maxtime)
                else:
                    fhb.append(maxtime)
                print 'Fmaxt', i

    for ii in range(len(psm)):
        pxx=np.where(data1['name'] == psm['name'][ii])[0]
        #pdata= data[pxx]
        if len(pxx) > 0:
            ttmjd = sorted(np.array(data1['mjd'][pxx]))[0]
            print ttmjd
            if len(ttmjd) > 1:
                pmaxtime = (max(ttmjd) - min(ttmjd))/(1.0+psm['Z_VI'][ii])/365
                pmintime = np.abs(ttmjd[1] - ttmjd[0])/(1.0+psm['Z_VI'][ii])/365
                pmaxt.append(pmaxtime)
                pmint.append(pmintime)
                pzvi.append(psm['Z_VI'][ii])
                if psm['name'][ii] in Ilbal['col1']:
                    plb.append(pmaxtime)
                else:
                    phb.append(pmaxtime)

                print 'Pmaxt', ii

    fmaxt = np.array(fmaxt) ;pmaxt=np.array(pmaxt) ;fmint=np.array(fmint) ; pmint=np.array(pmint) ;fzvi = np.array(fzvi);pzvi=np.array(pzvi)
    fhb=np.array(fhb)
    phb=np.array(phb)
    flb=np.array(flb)
    plb=np.array(plb)

    #fhb = np.where((fzvi> 1.6 ) & (fzvi < 5.6))[0]
    #phb = np.where((pzvi > 1.6 ) & (pzvi < 5.6))[0]
    #flb = np.where((fzvi> 0.25 ) & (fzvi < 3.95))[0]
    #plb = np.where((pzvi> 0.25 ) & (pzvi < 3.95))[0]
    print fmaxt
    fig,(ax,ax1)=plt.subplots(1,2,figsize=(10,5))
    hist(fmaxt,normed=False,bins=bins,histtype="step", linewidth= 2,alpha= 1, linestyle="-",color='black',align='mid',label='Full sample', ax=ax)
    hist(pmaxt,normed=False,bins=bins,histtype="step", linewidth= 2,alpha= 1, linestyle="-",color='red',align='mid',label='Initial sample', ax=ax)
    hist(fhb,normed=False,bins=bins,histtype="step", linewidth= 2,alpha= 1, linestyle="-",color='black',align='mid',label='Full sample-HiBAL', ax=ax1)
    hist(flb,normed=False,bins=bins,histtype="step", linewidth= 2,alpha= 1, linestyle="-",color='red',align='mid',label='Full sample-LoBAL', ax=ax1) 
    hist(phb,normed=False,bins=bins,histtype="step", linewidth= 2,alpha= 1, linestyle=":",color='black',align='mid',label='Initial sample-HiBAL', ax=ax1)
    hist(plb,normed=False,bins=bins,histtype="step", linewidth= 2,alpha= 1, linestyle=":",color='red',align='mid',label='Initial sample-LoBAL', ax=ax1) 
    ax.set(xlabel='Rest-frame time (years)', ylabel='Histogram',title='Maximum probed timescales')
    ax1.set(xlabel='Rest-frame time (years)', ylabel='Histogram',title='Maximum probed timescales')
    ax.legend(loc=1)
    ax1.legend(loc=1)
    ax.grid()
    ax1.grid()
    fig.tight_layout()
    fig.savefig('SDSSIV_BALQSO_restframetimescales.jpg')
    #plt.show()

def plot_redshift():
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))#fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    mg = np.where((fsm['Z_VI'] > 0.28) & (fsm['Z_VI'] < 2.28))[0] 
    siiv = np.where((fsm['Z_VI'] > 1.96) & (fsm['Z_VI'] < 5.55))[0] 
    civ = np.where((fsm['Z_VI'] > 1.68) & (fsm['Z_VI'] < 4.93))[0]
    al = np.where((fsm['Z_VI'] > 1.23) & (fsm['Z_VI'] < 3.93))[0]
    xx = np.where(fsm['Z_VI'] > 0)[0]
    bins = np.arange(0,6.5,0.25)

    fig,ax=plt.subplots(figsize=(5,5))
    hist(fsm['Z_VI'][xx],bins=bins,normed=0,histtype='step',color='black',lw=3,ax=ax)
    ax.set(xlabel='Z_VI',ylabel='Histogram',title='Redshift: full sample')
    xlim,ylim =ax.get_xlim(),ax.get_ylim()
    ax.text(xlim[0]+0.4*(xlim[1]-xlim[0]),ylim[1]-0.25*(ylim[1]-ylim[0]),'Z_VI between 1.96 and 5.55 :'+str(len(siiv)))
    ax.text(xlim[0]+0.4*(xlim[1]-xlim[0]),ylim[1]-0.2*(ylim[1]-ylim[0]),'Z_VI between 1.68 and 4.93 :'+str(len(civ)))
    ax.text(xlim[0]+0.4*(xlim[1]-xlim[0]),ylim[1]-0.1*(ylim[1]-ylim[0]),'Z_VI between 0.28 and 2.28 :'+str(len(mg)))
    ax.text(xlim[0]+0.4*(xlim[1]-xlim[0]),ylim[1]-0.15*(ylim[1]-ylim[0]),'Z_VI between 1.23 and 3.93 :'+str(len(al)))
    ax.grid()
    fig.tight_layout()
    fig.savefig('SDSSIV_BALs_fullsample_redshift.jpg')
    #plt.show()


def LoHIBAL_plot_redshift():
    ofsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))#fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    y=np.where(ofsm['Z_VI'] > 0)[0]
    fsm= ofsm[y]
    psm = np.genfromtxt('Master_gibson_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))#fits.open('Initialsample_2005_BAL_sources_crossmatch-DR12Q.fits')[1].data
    Flbal = fits.open('LoBAL_sample_fullsample.fits')[1].data
    Ilbal = fits.open('LoBAL_sample_initialsample.fits')[1].data
    Flzvi = []; Fhzvi=[] ; Ilzvi=[] ; Ihzvi=[]
    bins = np.arange(0,6.5,0.25)
    for i in range(len(fsm)):
        if fsm['name'][i] in Flbal['col1']:
            Flzvi.append(fsm['Z_VI'][i])
        else:
            Fhzvi.append(fsm['Z_VI'][i])
    for ii in range(len(psm)):
        if psm['name'][ii] in Ilbal['col1']:
            Ilzvi.append(psm['Z_VI'][ii])
        else:
            Ihzvi.append(psm['Z_VI'][ii])


    Flzvi = np.array(Flzvi); Fhzvi=np.array(Fhzvi) ; Ilzvi=np.array(Ilzvi) ; Ihzvi=np.array(Ihzvi)

    mg = np.where((fsm['Z_VI'] > 0.28) & (fsm['Z_VI'] < 2.28))[0] 
    siiv = np.where((fsm['Z_VI'] > 1.96) & (fsm['Z_VI'] < 5.55))[0] 
    civ = np.where((fsm['Z_VI'] > 1.68) & (fsm['Z_VI'] < 4.93))[0]
    al = np.where((fsm['Z_VI'] > 1.23) & (fsm['Z_VI'] < 3.93))[0] 
    fig,ax=plt.subplots(figsize=(5,5))
    hist(Fhzvi,bins=bins,normed=0,histtype='step',color='black',lw=3,ax=ax,label='Full sample-HiBAL: '+str(len(Fhzvi)))
    hist(Ihzvi,bins=bins,normed=0,histtype='step',color='red',lw=3,ax=ax,label='Initial sample-HiBAL: '+str(len(Ihzvi)))
    hist(Flzvi,bins=bins,normed=0,histtype='step',color='black',ls=':',lw=3,ax=ax,label='Full sample-LoBAL: '+str(len(Flzvi)))
    hist(Ilzvi,bins=bins,normed=0,histtype='step',color='red',ls=':',lw=3,ax=ax,label='Initial sample-LoBAL: '+str(len(Ilzvi)))
    ax.set(xlabel='Z\_VI',ylabel='Histogram',title='Redshift distribution')
    xlim,ylim =ax.get_xlim(),ax.get_ylim()
    #ax.text(xlim[0]+0.5*(xlim[1]-xlim[0]),ylim[1]-0.25*(ylim[1]-ylim[0]),'Full sample-HiBAL :'+str(len(Fhzvi)))
    #ax.text(xlim[0]+0.5*(xlim[1]-xlim[0]),ylim[1]-0.2*(ylim[1]-ylim[0]),'Full sample-LoBAL :'+str(len(Flzvi)))
    #ax.text(xlim[0]+0.5*(xlim[1]-xlim[0]),ylim[1]-0.1*(ylim[1]-ylim[0]),'Initial sample-HiBAL :'+str(len(Ihzvi)))
    #ax.text(xlim[0]+0.5*(xlim[1]-xlim[0]),ylim[1]-0.15*(ylim[1]-ylim[0]),'Initial sample-LoBAL :'+str(len(Ilzvi)))
    #ax.grid()
    ax.legend(loc=1)
    ax.grid()
    fig.tight_layout()
    fig.savefig('SDSSIV_LoHiBALs_redshift.jpg')
    #plt.show()



def plot_spectra():
    text_font = {'fontname':'Times New Roman', 'size':'14'}
    pp = PdfPages('Fullanalysis_BAL_plot_spectra.pdf') 
    specdir = 'SDSSIV_BALdata'
    linelist = np.genfromtxt('/Users/vzm83/Proposals/linelist_speccy.txt',usecols=(0,1,2),dtype=('|S10',float,'|S5'),names=True)
    color=['black','red','blue','green','brown','cyan','magenta','gold']
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    #fsm = fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))
    for i in range(len(data['name'])):
    #for i in range(10):
        plates = data['plate'][i]
        mjds = data['mjd'][i]
        fibers = data['fiber'][i]
        print data['name'][i]
        xx=np.where(fsm['name'] == data['name'][i])[0]
        if len(xx)>0:
            zvi = fsm['Z_VI'][xx[0]]
            print xx,data['name'][i],fsm['name'][xx]
            minflux =[] ;maxflux=[]
            fig,ax=plt.subplots(figsize=(15,8))
            for j in range(len(plates)):
                if plates[j] >=10000:
                    PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
                else:
                    PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
                print PMF
                if os.path.isfile(os.path.join(specdir,PMF)):
                    data1 = fits.open(os.path.join(specdir,PMF))[1].data
                    minflux.append(np.median((data1['flux']*(data1['and_mask'] == 0)).copy()))
                    maxflux.append(np.std((data1['flux']*(data1['and_mask'] == 0)).copy()))
                    if j < 8:
                        ax.plot(10**data1['loglam']/(1.0+zvi),gaussian_filter1d(data1['flux'],2),color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                        ax.plot(10**data1['loglam']/(1.0+zvi),1.0/np.sqrt(data1['ivar']),color=color[j],alpha=0.1)
                    else:
                        ax.plot(10**data1['loglam']/(1.0+zvi),gaussian_filter1d(data1['flux'],2),color=plt.cm.RdYlBu(j*300),alpha=0.5,label=PMF.split('.')[0][5:])
                        ax.plot(10**data1['loglam']/(1.0+zvi),1.0/np.sqrt(data1['ivar']),color=plt.cm.RdYlBu(j*300),alpha=0.1)
                
                
                if j == len(plates)-1:
                    if (len(maxflux) > 0):
                        ax.set(xlabel='Rest wavelength ($\AA$)',ylabel='Flux',ylim=(-2,max(minflux)+3.0*max(maxflux)), title=data['name'][i])
                    xlim,ylim=ax.get_xlim(),ax.get_ylim()
                    #string1 = 'SDSS J{0}\tZ\_VI: {1:4.4f}\tN$\_{{spec}}$: {2}'.format(fsm['SDSS_NAME'][xx[0]],zvi,len(plates))
                    string1 = 'RA: {0:5.4f}\t DEC: {1:5.4f}  \tZ\_VI: {2:4.4f}\tN$\_{{spec}}$: {3}'.format(fsm['ra'][xx[0]],fsm['dec'][xx[0]],zvi,len(plates))
                    print string1,xlim,ylim
                    ax.text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.05*(ylim[1] - ylim[0]), string1,fontsize=18)
                
                    obslambda = linelist['lambda']#*(1.+zvi)
                    x = np.where((obslambda > xlim[0]) & (obslambda < xlim[1]))[0]
	            plotlambda = obslambda[x]
	            plotname = linelist['Name'][x]
	            plota_e = linelist['a_e'][x]
	            #print plotlambda
	            for k in range(len(plotlambda)):
	                if plota_e[k].strip() == 'Abs.' : 
	            	    ax.axvline(x=plotlambda[k], color='lawngreen', linestyle=':')
	            	    ax.text(plotlambda[k],ylim[0]+0.75*(ylim[1]-ylim[0]),plotname[k],color='Orange',ha='center',rotation=90,**text_font)
	                else :
	            	    ax.axvline(x=plotlambda[k], color='lightblue', linestyle=':')
	            	    ax.text(plotlambda[k],ylim[0]+0.75*(ylim[1]-ylim[0]),plotname[k],color='Brown',ha='center',rotation=90,**text_font)
    
        ax.legend(loc=1)
        fig.tight_layout()
        #fig.savefig(pp,format='pdf')
        fig.savefig(os.path.join('BAL_Plots','spectra_overplot_'+data['name'][i]+'.jpg'),format='jpg')
        #sdlfj=raw_input()
    pp.close()
                

def createBigFits():
    specdir = 'SDSSIV_BALdata'
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))
    #fsm = fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    balflux = []; balivar=[] ; balmask =[] ; balmap = []; balwave = []
    
    nfibers = 10#np.sum([len(a) for a in data['plate']])

    llmin = 3600
    llmax = 10500
    dll = 0.001
    nbins = int( (llmax-llmin)/dll + 1)
    ll = llmin + np.arange(nbins)*dll
    llmax = ll.max()
    
    hdr = fits.Header()
    hdr['comment'] = 'This fits file contains the all the spectra  of the 3014 sources in  the SDSS BAL variability project'
    hdr['comment'] = 'and the redshift information updated to HW redshifts'
    hdr['Author'] = 'Vivek M.'
    hdr['Date'] = 'May 8 2019'
    primary_hdu = fits.PrimaryHDU(header=hdr)


    
    #print nbins
    #--
    hdulist = fits.HDUList(primary_hdu)#np.zeros((nfibers, nbins))
    #nivar = fits.HDUList()#np.zeros((nfibers, nbins))
    #nmask = fits.HDUList()#np.zeros((nfibers, nbins))
    #nloglam = fits.HDUList()#np.zeros((nfibers, nbins))
    fibermap = fits.HDUList(primary_hdu)

    for i in range(len(data['name'])):
    #for i in range(10):
        plates = data['plate'][i]
        mjds = data['mjd'][i]
        fibers = data['fiber'][i]
        print data['name'][i]
        xx=np.where(fsm['name'] == data['name'][i])[0]
        if len(xx)>0:
            zvi = fsm['Z_VI'][xx[0]]
        for j in range(len(plates)):
                if plates[j] >=10000:
                    PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
                else:
                    PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
                print PMF
                if os.path.isfile(os.path.join(specdir,PMF)):
                    data1 = fits.open(os.path.join(specdir,PMF))
                    #balflux.append(data1['flux'])
                    #balwave.append(10**data1['loglam'])
                    #balivar.append(data1['ivar'])
                    #balmask.append(data1['and_mask'])
                    if (data1[1].data['flux'].all() !=0):
                        hdulist.append(data1[1].copy())
                        tmpmap= fits.open(os.path.join(specdir,PMF))[2].data
                        tmpmap['z'] = zvi
                        fibermap.append(data1[2].copy())
    #balflux = np.array(balflux)
    #balwave = np.array(balwave)
    #balivar = np.array(balivar)
    #balmask = np.array(balmask)
    #balmap = np.array(balmap)
    #print balflux 
    #bhdu0 = fits.ImageHDU(np.array(balflux))
    #bhdu1 = fits.ImageHDU(balivar)
    #bhdu2 = fits.ImageHDU(balmask)
    #bhdu4 = fits.ImageHDU(balwave)
    #bhdu3 = fits.BinTableHDU(data=balmap)
    #bhdulist = fits.HDUList([primary_hdu,bhdu0,bhdu1,bhdu2,bhdu4,bhdu3])

    hdulist.writeto('SDSSIV_BALQSO_BigFits.fits')
    fibermap.writeto('SDSSIV_BALQSO_BigFits_fibermap.fits')

def plot_BALfootprint():
    print 'Begin'
    sp = fits.open('spAll-v5_13_0.fits')[1].data
    print 'Reading spAll-new'
    #data = fits.open('spAll-v5_10_10.fits')[1].data
    print 'Reading spAll-old'
    #yy=np.where(data['PROGRAMNAME'] == 'sequels')[0]
    #sequels = data[yy]
    elgngc = np.where(sp['PROGRAMNAME'] == 'ELG_NGC')[0]
    elgsgc = np.where(sp['PROGRAMNAME'] == 'ELG_SGC')[0]

    nn=pd.read_csv('NGC_bounds',names=['ra','dec'])
    dd=pd.read_csv('SGC_bounds',names=['ra','dec'])

    npolygon = [[ngcp[0],ngcp[1]] for ngcp in zip(nn['ra'],nn['dec'])]
    dpolygon = [[sgcp[0],sgcp[1]] for sgcp in zip(dd['ra'],dd['dec'])]
    
    
    


    radec = np.genfromtxt('Master_initial_sample.txt',usecols=(1,2),names=['ra','dec'],skip_header=1)
    #radec = np.genfromtxt('Master_gibson_2005_targets_cor.txt',usecols=(1,2),names=['ra','dec'],skip_header=1)
    ra=radec['ra'] ; dec = radec['dec']
    
    mcount = np.genfromtxt('Fullsample_3epoch_counts.txt',names=['name','ra','dec','sdss1','sdss2','sdss3'],dtype=('|S15',float,float,int,int,int))
    #mcount = np.genfromtxt('Fullsample_3epoch_counts_2005.txt',names=['name','ra','dec','sdss1','sdss2','sdss3'],dtype=('|S15',float,float,int,int,int))
    xx=np.where(mcount['sdss3']>0)[0]
    
    sqbe = fits.open('sequelsVARBALS_before_eBOSS56777_fullsample.fits')[1].data
    #sqbe = fits.open('sequelsVARBALS_before_eBOSS56777_initialsample.fits')[1].data
    mcountra = [ s for s in mcount['ra'][xx] ]
    mcountdec = [ s for s in mcount['dec'][xx] ]
    sqbera = [s for s in sqbe['RA']]
    sqbedec = [s for s in sqbe['dec']]
    
    ra1,dec1=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_NGC.txt').T
    ra2,dec2=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_SGC.txt').T
    eboss = np.where(sp['PROGRAMNAME'] == 'eboss')[0]
    speboss = sp[eboss]
    q = np.where(speboss['CLASS'] == 'QSO')[0]
    #q1 = np.where(sequels['CLASS'] == 'QSO')[0]
    print 'starting to plot'
    fig,ax1 = plt.subplots(figsize=(15,8))
    ax1.plot(ra1,dec1,'-',color='black',alpha=0.5)
    ax1.plot(ra2,dec2,'-',color='black',alpha=0.5)
    ax1.plot(nn['ra'],nn['dec'],'-',color='green',alpha=0.8,lw=3,label='eBOSS final Footprint')
    ax1.plot(dd['ra'],dd['dec'],'-',color='green',alpha=0.8,lw=3,label='')
    
 
    #ax1.plot(negativeRAs(speboss['RA'][q]),speboss['DEC'][q],'.',markersize=3,color='red',alpha=0.1)
    #ax1.plot(negativeRAs(sequels['RA'][q1]),sequels['DEC'][q1],'x',markersize=3,color='red',alpha=0.1)
    #ax1.plot(negativeRAs(sp['RA'][elgngc]),sp['DEC'][elgngc],'.',markersize=3,color='red',alpha=0.1)
    #ax1.plot(negativeRAs(sp['RA'][elgsgc]),sp['DEC'][elgsgc],'.',markersize=3,color='red',alpha=0.1)

    ax1.plot(negativeRAs(ra),dec,'x',markersize=2,color='blue',label='Targets')
    ax1.plot(negativeRAs(mcount['ra'][xx]),mcount['dec'][xx],'o',markersize=3,color='black',fillstyle='none',label='Observed')
    ax1.plot(negativeRAs(sqbera),sqbedec,'o',markersize=3,color='black',fillstyle='none',label='')
    ax1.set(xlabel='RA',ylabel='DEC',title='eBOSS QSO Full sample footprint')
    #ax1.set(xlabel='RA',ylabel='DEC',title='eBOSS QSO Initial sample footprint')
    ax1.set_xlim(-55,300)
    ax1.set_ylim(-15,100)
    ax1.legend(loc=1)
    ax1.grid()
    xlim,ylim=ax1.get_xlim(),ax1.get_ylim()
    ax1.text(xlim[0]+0.1*(xlim[1]-xlim[0]),ylim[1]-0.1*(ylim[1]-ylim[0]),'1895 targets inside final footprint',fontsize=18)
    ax1.text(xlim[0]+0.1*(xlim[1]-xlim[0]),ylim[1]-0.15*(ylim[1]-ylim[0]),'1744 observed inside final footprint',fontsize=18)
    #ax1.text(xlim[0]+0.1*(xlim[1]-xlim[0]),ylim[1]-0.1*(ylim[1]-ylim[0]),'1138 targets inside final footprint',fontsize=18)
    #ax1.text(xlim[0]+0.1*(xlim[1]-xlim[0]),ylim[1]-0.15*(ylim[1]-ylim[0]),'1039 observed inside final footprint',fontsize=18)

    fig.tight_layout()
    #print 'Interactive window should open now'
    coords = []
    NGC_targets = [ray_tracing_method(point[0], point[1], npolygon) for point in zip(negativeRAs(ra),dec)]
    SGC_targets = [ray_tracing_method(point[0], point[1], dpolygon) for point in zip(negativeRAs(ra),dec)]

    NGC_observed = [ray_tracing_method(point[0], point[1], npolygon) for point in zip(negativeRAs(mcountra),mcountdec)]
    NSQBE_observed = [ray_tracing_method(point[0], point[1], npolygon) for point in zip(negativeRAs(sqbera),sqbedec)]
    SGC_observed = [ray_tracing_method(point[0], point[1], dpolygon) for point in zip(negativeRAs(mcountra),mcountdec)]
    SSQBE_observed = [ray_tracing_method(point[0], point[1], dpolygon) for point in zip(negativeRAs(sqbera),sqbedec)]
    NGC_targets = np.array(NGC_targets)
    SGC_targets = np.array(SGC_targets)
    NGC_observed = np.array(NGC_observed)
    SGC_observed = np.array(SGC_observed)
    NSQBE_observed = np.array(NSQBE_observed)
    SSQBE_observed = np.array(SSQBE_observed)

    print NGC_targets
    print SGC_targets
    print NGC_observed
    print SGC_observed
    print len(np.where(NGC_targets == True)[0]) + len(np.where(SGC_targets == True)[0]), len(np.where(NGC_targets == True)[0]), len(np.where(SGC_targets == True)[0])
    print len(np.where(NGC_observed == True)[0]) + len(np.where(SGC_observed == True)[0])+len(np.where(NSQBE_observed == True)[0]) + len(np.where(SSQBE_observed == True)[0]), len(np.where(NGC_observed == True)[0])+len(np.where(NSQBE_observed == True)[0]) , len(np.where(SGC_observed == True)[0]) + +len(np.where(SSQBE_observed == True)[0]) 
    # Call click func
    #cid = fig.canvas.mpl_connect('button_press_event', onclick)

    #plt.show(1)
    
    fig.savefig('SDSSIV_eBOSS_VARBAL_footprint_fullsample.jpg')
    #fig.savefig('SDSSIV_eBOSS_VARBAL_footprint_initialsample.jpg')
    print len(ra)



    #plt.show()

def checkifSpectra():
    specdir = 'SDSSIV_BALdata'
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    count = 0
    for i in range(len(data['name'])):
        plates = data['plate'][i]
        mjds = data['mjd'][i]
        fibers = data['fiber'][i]
        for j in range(len(plates)):
            if plates[j] >=10000:
                PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
            else:
                PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
            
            filename = os.path.join(specdir,PMF)
            if not os.path.isfile(filename):
                print 'File not found {0:10.5f}\t{1:10.5f}\t{2}\t{3}'.format(data['ra'][i],data['dec'][i], PMF,i)
                count +=1
            else:
                cpcmd = 'cp {0} BALQSOs_spectra/.'.format(os.path.join(specdir,PMF))
                os.system(cpcmd)
    print '--'*51
    print 'Total :', count

def makefits():
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    c1 = fits.Column(name='name', array=np.array(data['name']), format='10A')
    c2 = fits.Column(name='ra', array=np.array(data['ra']), format='E')
    c3 = fits.Column(name='dec', array=np.array(data['dec']), format='E')
    c4 = fits.Column(name='plate', array=np.array(data['plate'],dtype=np.object), format='PJ()')
    c5 = fits.Column(name='mjd', array=np.array(data['mjd'],dtype=np.object), format='PJ()')
    c6 = fits.Column(name='fiber', array=np.array(data['fiber'],dtype=np.object), format='PJ()')
    hdr = fits.Header()
    hdr['comment'] = 'This fits file contains the plate-mjd-fiber info for the 3014 sources in  the SDSS BAL variability project'
    hdr['Author'] = 'Vivek M.'
    hdr['Date'] = 'May 8 2019'
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdu = fits.BinTableHDU.from_columns([c1, c2, c3,c4, c5, c6])
    hdul = fits.HDUList([primary_hdu, hdu])
    hdul.writeto('SDSSIV_BALQSO_masterList.fits')
    #for i in range(10):
    #    plates = data['plate'][i]
    #    mjds = data['mjd'][i]
    #    fibers = data['fiber'][i]
    #    ra = data['ra'][i]
    #    dec = data['dec'][i]
    #    name = data['name'][i]

def SN1700(wave,flux,err,zvi):
    yy=np.where(err >0)[0]
    wave = wave[yy]; flux=flux[yy] ; err = err[yy]
    if zvi < 1.2 :
        ww=np.where((wave >=2950) & (wave <= 3050))[0]
    else:
        ww=np.where((wave >=1650) & (wave <= 1750))[0]
    if np.isfinite(np.median(flux[ww])/np.median(err[ww])):
        return np.median(flux[ww])/np.median(err[ww])
    else :
        return np.median(flux)/np.median(err)

def computeSN1700():
    specdir = 'SDSSIV_BALdata'
    jxvf=open('SN1700_estimates_Master_full_sample_sources.txt','w')
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))
    for i in range(len(data['name'])):
    #for i in range(10):
        plates = data['plate'][i]
        mjds = data['mjd'][i]
        fibers = data['fiber'][i]
        print data['name'][i]
        xx=np.where(fsm['name'] == data['name'][i])[0]
        if len(xx)>0:
            zvi = fsm['Z_VI'][xx[0]]
        for j in range(len(plates)):
                if plates[j] >=10000:
                    PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
                else:
                    PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(plates[j],mjds[j],fibers[j])
                print PMF
                if os.path.isfile(os.path.join(specdir,PMF)):
                    data1 = fits.open(os.path.join(specdir,PMF))[1].data
                    wave = (10**(data1.loglam))/(1.0+zvi)
                    flux = data1.flux ; error = 1.0/np.sqrt(data1.ivar)
                    wave1 = wave;flux1=flux;error1=error
                    print>>jxvf,'{0}\t{1}\t{2}\t{3}\t{4:5.4f}\t{5:5.3f}'.format(data['name'][i],plates[j],mjds[j],fibers[j],zvi,SN1700(wave1,flux1,error1,zvi))

    jxvf.close()


def linefreeRegions(wave,flux,err):
    xx= np.where( ( (wave >= 1270) & ( wave <= 1320) ) | ( (wave >= 1700) & ( wave <= 1800) ) | ( (wave >= 2000) & ( wave <= 2200) ) |( (wave >= 2650) & ( wave <= 2700) ) |( (wave >= 2950) & ( wave <= 3700) ) | ( (wave >= 3950) & ( wave <= 4050) )  )[0]
    nw = wave[xx] ; nf=flux[xx];nerr=err[xx]

    return nw,nf,nerr

def powerlawFunc(xdata,  amp,index):
    return amp*np.power(xdata,index)


def fitPowerlaw(wave,flux,weight, amp=1,index=1): 
    from lmfit import minimize, Parameters, fit_report, Minimizer
    import numpy as np
    import scipy.optimize as optimization
    #wav_range= [(1250,1350),(1700,1800),(1950,2500),(2650,2710),(2950,3700),(3950,4050)]
    #xx = np.where( ((wave > 1250 ) & (wave < 1300))  | ((wave > 1590 ) & (wave < 1750))  |((wave > 2000 ) & (wave < 2100)) | ((wave > 2350 ) & (wave < 2500)) | ((wave > 2650 ) & (wave < 2700)) | ((wave > 2950 ) & (wave < 3700)) | ((wave > 3950 ) & (wave < 4050))   )
    x0= [amp,index]
    xdata=np.asarray(wave)  
    ydata=np.asarray(flux)
    sigma=np.asarray(weight)
    #print len(xdata),len(ydata),len(sigma) 
    try:    
        popt, pcov = optimization.curve_fit(powerlawFunc, xdata, ydata, x0, sigma)
    except (RuntimeError, TypeError):
        popt,pcov = (1,1),1
    #print popt
    #popt, pcov = optimization.curve_fit(func, xdata, ydata, x0)
    model = powerlawFunc(wave,popt[0],popt[1])
    chi2 = ((flux - model)*np.sqrt(weight))**2
    rchi2 = np.sum(chi2)/(len(xdata) - 2)
    print 'Reduced Chi Square : {0}  Number of points: {1}'.format(rchi2,len(xdata))
    print popt
    return popt,pcov   

def maskOutliers(wave,flux,error,amp,index):
    model = powerlawFunc(wave,amp,index)
    std =np.std(flux)
    fluxdiff = flux - model
    ww = np.where (np.abs(fluxdiff) > 3*std)
    nwave = np.delete(wave,ww)
    nflux = np.delete(flux,ww)
    nerror = np.delete(error,ww)
    
    return nwave,nflux,nerror

def makeOverplots():
    text_font = {'fontname':'Times New Roman', 'size':'14'}
    outtxt=open('PowerlawFits_Master_Fullsample_output.txt','w')
    #pp = PdfPages('Fullanalysis_BAL_overplot_Plaw_spectra.pdf') 
    specdir = 'SDSSIV_BALdata'
    linelist = np.genfromtxt('/Users/vzm83/Proposals/linelist_speccy.txt',usecols=(0,1,2),dtype=('|S10',float,'|S5'),names=True)
    color=['black','red','blue','green','brown','cyan','magenta','gold','purple','black','red','blue','green','brown','cyan']
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    #fsm = fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))
    snfile = np.genfromtxt('SN1700_estimates_Master_full_sample_sources.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))
    #for i in range(len(data['name'])):
    for i in np.arange(1700,len(data['name'])-1):
    #for i in range(10):
        fig,ax=plt.subplots(figsize=(15,8))
        nplates=[];nmjds=[];nfibers=[];nsn=[]
        plates = data['plate'][i]
        mjds = data['mjd'][i]
        fibers = data['fiber'][i]
        print data['name'][i]
        xx=np.where(fsm['name'] == data['name'][i])[0]
        yy=np.where(snfile['name'] == data['name'][i])[0]
        
        idx = np.argsort(snfile['sn'][yy])[::-1]
        for ii in range(len(idx)):
            print '{0}\t{1}\t{2}\t{3:5.2f}'.format(snfile['plate'][yy[idx[ii]]],snfile['mjd'][yy[idx[ii]]],snfile['fiber'][yy[idx[ii]]],snfile['sn'][yy[idx[ii]]])
            nplates.append(snfile['plate'][yy[idx[ii]]]) ; nmjds.append(snfile['mjd'][yy[idx[ii]]]) ; nfibers.append(snfile['fiber'][yy[idx[ii]]]) ; nsn.append(snfile['sn'][yy[idx[ii]]])
        
        print nplates,nmjds,nfibers,nsn
        if len(xx)>0:
            zvi = fsm['Z_VI'][xx[0]]
            print xx,data['name'][i],fsm['name'][xx]
            minflux =[] ;maxflux=[]
            for j in range(len(nplates)):
                if nplates[j] >=10000:
                    PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(nplates[j],nmjds[j],nfibers[j])
                else:
                    PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(nplates[j],nmjds[j],nfibers[j])
                print PMF
                if os.path.isfile(os.path.join(specdir,PMF)):
                    data1 = fits.open(os.path.join(specdir,PMF))[1].data
                    minflux.append(np.median((data1['flux']*(data1['and_mask'] == 0)).copy()))
                    maxflux.append(np.std((data1['flux']*(data1['and_mask'] == 0)).copy()))

                    wave1 = (10**(data1.loglam))/(1.0+zvi)
                    flux1 = data1.flux ; error1 = 1.0/np.sqrt(data1.ivar)
                    flux2= savitzky_golay(flux1,15,2)
                    mask1 = data1.and_mask
                    wave1 = wave1[np.where(mask1==0)[0]]
                    flux1 = flux1[np.where(mask1==0)[0]]
                    error1 = error1[np.where(mask1==0)[0]]
                    
                    nw,nf,nerr = linefreeRegions(wave1,flux1,error1)
                    plaw,pcoeff = fitPowerlaw(nw,nf,nerr)
                    nnw,nnf,nnerr = maskOutliers(nw,nf,nerr,plaw[0],plaw[1])
                    for jj in range(10):
                        plaw,pcoeff = fitPowerlaw(nnw,nnf,nnerr)
                        print plaw
                        if jj !=max(range(10)):
                            nnw,nnf,nnerr = maskOutliers(nnw,nnf,nnerr,plaw[0],plaw[1])
                    plchi2 = np.sum((nnf - powerlawFunc(nnw,plaw[0],plaw[1]))/nnerr)**2 
                    print len(nnw)
                    if j==0:
                        print '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:5.5f}\t{6:10.5f}\t{7:10.5f}\t{8:10.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),1)    
                        print>>outtxt, '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:5.5f}\t{6:10.5f}\t{7:10.5f}\t{8:10.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),1)  
                    if j!=0:
                        print '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:5.5f}\t{6:10.5f}\t{7:10.5f}\t{8:10.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),0)    
                        print>>outtxt, '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:10.5f}\t{6:10.5f}\t{7:10.5f}\t{8:5.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),0)
                    if j ==0:
                        refamp=plaw[0];refindex=plaw[1]
              

                    if j < 8:
                        if j ==0:
                            ax.plot(wave1,gaussian_filter1d(flux1,2),color=color[j],alpha=0.5,label=PMF.split('.')[0][5:]+' SN: '+str(nsn[j]))
                            ax.plot(wave1,error1,color=color[j],alpha=0.1)
                        if j !=0:
                            ax.plot(wave1,gaussian_filter1d(flux1,2)*(refamp/plaw[0])*(wave1)**(refindex-plaw[1]),color=color[j],alpha=0.5,label=PMF.split('.')[0][5:]+' SN: '+str(nsn[j]))
                            ax.plot(wave1,error1,color=color[j],alpha=0.1)
                            #ax.plot(wave1,powerlawFunc(wave1,plaw[0],plaw[1]),lw=3,ls='--',color=color[j],alpha=0.9)
                            #ax.plot(nnw,nnf,'.',color=color[j],alpha=0.9)
                    else:
                        if j ==0:
                            ax.plot(wave1,gaussian_filter1d(flux1,2),color=plt.cm.RdYlBu(j*300),alpha=0.5,label=PMF.split('.')[0][5:])
                            ax.plot(wave1,error1,color=color[j],alpha=0.1)
                        if j !=0:
                            ax.plot(wave1,gaussian_filter1d(flux1,2)*(refamp/plaw[0])*(wave1)**(refindex-plaw[1]),color=plt.cm.RdYlBu(j*300),alpha=0.5,label=PMF.split('.')[0][5:])
                            ax.plot(wave1,error1,color=plt.cm.RdYlBu(j*300),alpha=0.1)
                            #ax.plot(wave1,powerlawFunc(wave1,plaw[0],plaw[1]),lw=3,ls='--',color=plt.cm.RdYlBu(j*300),alpha=0.9)
                    if j == len(plates)-1:
                        if (len(maxflux) > 0):
                            ax.set(xlabel='Rest wavelength ($\AA$)',ylabel='Flux',ylim=(-2,max(minflux)+3.0*max(maxflux)), title=data['name'][i])
                        xlim,ylim=ax.get_xlim(),ax.get_ylim()
                        #string1 = 'SDSS J{0}\tZ\_VI: {1:4.4f}\tN$\_{{spec}}$: {2}'.format(fsm['SDSS_NAME'][xx[0]],zvi,len(plates))
                        string1 = 'RA: {0:5.4f}\t DEC: {1:5.4f}  \tZ\_VI: {2:4.4f}\tN$\_{{spec}}$: {3}'.format(fsm['ra'][xx[0]],fsm['dec'][xx[0]],zvi,len(plates))
                        print string1,xlim,ylim
                        ax.text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.05*(ylim[1] - ylim[0]), string1,fontsize=18)
                
                        obslambda = linelist['lambda']#*(1.+zvi)
                        x = np.where((obslambda > xlim[0]) & (obslambda < xlim[1]))[0]
	                plotlambda = obslambda[x]
	                plotname = linelist['Name'][x]
	                plota_e = linelist['a_e'][x]
	                #print plotlambda
	                for k in range(len(plotlambda)):
	                    if plota_e[k].strip() == 'Abs.' : 
	            	        ax.axvline(x=plotlambda[k], color='lawngreen', linestyle=':')
	            	        ax.text(plotlambda[k],ylim[0]+0.75*(ylim[1]-ylim[0]),plotname[k],color='Orange',ha='center',rotation=90,**text_font)
	                    else :
	            	        ax.axvline(x=plotlambda[k], color='lightblue', linestyle=':')
	            	        ax.text(plotlambda[k],ylim[0]+0.75*(ylim[1]-ylim[0]),plotname[k],color='Brown',ha='center',rotation=90,**text_font)
    
        ax.legend(loc=1)
        fig.tight_layout()
        #fig.savefig(pp,format='pdf')
        fig.savefig(os.path.join('BAL_Plots','spectra_overplot_Plaw_'+data['name'][i]+'.png'),format='png')
        #sdlfj=raw_input()
    pp.close()
    outtxt.close()


def highestSNspectraList():
    outfile = open('highestSNspectra_PMF_master_sample.txt','w')

    snfile = np.genfromtxt('/Users/vzm83/SDSSIV_BALQSO/SN1700_estimates_Master_full_sample_sources.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))
    
    unames=np.unique(snfile['name']) 
    count=0
    for i in range(len(unames)):
        xx=np.where(snfile['name']==unames[i])[0]
        print unames[i],len(xx)
        nfile=snfile[xx]
        yy=np.where(nfile['sn'] == max(nfile['sn']))[0][0]
        print>>outfile, '{0}\t{1}\t{2}\t{3}\t{4:10.5f}\t{5:10.5f}'.format(nfile['name'][yy],nfile['plate'][yy],nfile['mjd'][yy],nfile['fiber'][yy],nfile['Z_VI'][yy],nfile['sn'][yy])#,nfile['sn']
        count += 1
    print count
    outfile.close()


def CrossCorr_template():
    snfile = np.genfromtxt('/Users/vzm83/SDSSIV_BALQSO/SN1700_estimates_Master_full_sample_sources.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))
    #usnfile = np.genfromtxt('/Users/vzm83/SDSSIV_BALQSO/highestSNspectra_PMF_master_sample.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))
    usnfile = fits.open('../DR14QSO/DR12Q_BAL.fits')[1].data
    specdir = 'BOSS_BALDATA_allDR12'
    #create a template
    
    templateName= 'Target2577' ; templatePlate = 367 ; templateMJD = 51997 ; templateFiber = 506
    #templateName= 'Target0468' ; templatePlate = 410 ; templateMJD = 51877 ; templateFiber = 623
    m=np.where((snfile['plate'] == templatePlate) & (snfile['mjd'] == templateMJD) & (snfile['fiber'] == templateFiber))[0]
    if templatePlate >=10000:
        tPMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(templatePlate,templateMJD,templateFiber)
    else:
        tPMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(templatePlate,templateMJD,templateFiber)
    print 'Template File: ', tPMF
    tdata = fits.open(os.path.join('/Users/vzm83/SDSSIV_BALQSO/BALQSOs_spectra',tPMF))[1].data
    twave = 10**tdata.loglam/(1.0+snfile['Z_VI'][m])
    tflux = tdata.flux
    terr = 1.0/np.sqrt(tdata.ivar)
    tandmask= tdata.and_mask
    # Trim the wavelengths to > 1215 and < 3500
    xx=np.where((twave > 1250) & (twave < 3500) & (tandmask==0))[0]
    if len(xx) >10:
        twave = twave[xx] ; tflux = tflux[xx]; terr = terr[xx]
    outfile =open('AllDR12_Chisquare_test_for_template'+templateName+'.txt','w') 
    
    for i in range(len(usnfile)):
    #for i in range(10):
        if usnfile['plate'][i] >=10000:
            PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(usnfile['plate'][i],usnfile['mjd'][i],usnfile['fiber'][i])
        else:
            #PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(usnfile['plate'][i],usnfile['mjd'][i],usnfile['fiber'][i])
            PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(usnfile['PLATE'][i],usnfile['MJD'][i],usnfile['FIBERID'][i])
        print i, PMF
    
        data = fits.open(os.path.join(specdir,PMF))[1].data
        swave = 10**data.loglam/(1.0+usnfile['Z_VI'][i])
        sflux = data.flux
        serr = 1.0/np.sqrt(data.ivar)
        and_mask = data.and_mask
        xx1=np.where((swave > 1250) & (swave < 3500) & (and_mask==0))[0]
        print xx1,len(xx1),len(swave)
        if len(xx1)>10:
            wave = swave[xx1] ; flux=sflux[xx1]; err =serr[xx1]
        else:
            print>>outfile, '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(usnfile['SDSS_NAME'][i],usnfile['PLATE'][i],usnfile['MJD'][i],usnfile['FIBERID'][i],0,0,0)
            continue

        #Interpolate to template wavelength grid
        f = interp1d(wave,flux,bounds_error = False, fill_value = 0)
        iflux = f(twave)
    
        #print>>outfile, '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(usnfile['name'][i],usnfile['plate'][i],usnfile['mjd'][i],usnfile['fiber'][i],sp.stats.chisquare(iflux,f_exp=tflux)[0],sp.stats.chisquare(iflux,f_exp=tflux)[1],np.corrcoef(tflux,iflux)[0,1])
        print>>outfile, '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}'.format(usnfile['SDSS_NAME'][i],usnfile['PLATE'][i],usnfile['MJD'][i],usnfile['FIBERID'][i],sp.stats.chisquare(iflux,f_exp=tflux)[0],sp.stats.chisquare(iflux,f_exp=tflux)[1],np.corrcoef(tflux,iflux)[0,1])

    outfile.close()


def numerousBALSystems():
    save=open('NumerousBALsystems_subsample.txt','w')
    dt = np.load('MasterList_Plate-MJD-Fiber.npz')

    list =['Target2577','Target0536','Target0700','Target2206','Target2986','Target0966','Target1108','Target1392','Target2431','Target2570','Target2827','Target3002','Target0521','Target1074','Target2577','Target0269','Target1368','Target2992','Target0934','Target1104']
    allPMFlist = np.genfromtxt('allDR12_numerousBALsystems_list.txt',names=['plate','mjd','fiber'])
    snfile = np.genfromtxt('/Users/vzm83/SDSSIV_BALQSO/SN1700_estimates_Master_full_sample_sources.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))
    dr12q =fits.open('../DR14QSO/DR12Q_BAL.fits')[1].data
    count =0
    tplates = []; tmjds=[];tfibers =[]
    for ll in list:
        #print ll
        xx=np.where(snfile['name']==ll)[0]
        isnfile = snfile[xx]
        found =False
        for j in range(len(isnfile)):
            tplates.append(isnfile['plate'][j])
            tmjds.append(isnfile['mjd'][j])
            tfibers.append(isnfile['fiber'][j])
            mm=np.where(isnfile['sn'] == max(isnfile['sn'] ))[0]
            yz = np.where(dt['name'] == isnfile['name'][mm[0]])[0]
         #   print mm
            zz=np.where((allPMFlist['plate']== isnfile['plate'][j]) & (allPMFlist['mjd'] == isnfile['mjd'][j]  ) & (  allPMFlist['fiber']==isnfile['fiber'][j])  )[0]
            if len(zz) >0:
                found=True
        if found:
            count +=1
            print>>save, '{0}\t{1:30}\t{2:10.5f}\t{3:10.5f}\t{4}\t{5}\t{6}\t{7:10.5f}\t{8}'.format(count,isnfile['name'][mm[0]],dt['ra'][yz[0]],dt['dec'][yz[0]],isnfile['plate'][mm[0]],isnfile['mjd'][mm[0]],isnfile['fiber'][mm[0]],isnfile['Z_VI'][mm[0]],'BothMES-DR12Q')
        else:
            count +=1
            print>>save, '{0}\t{1:30}\t{2:10.5f}\t{3:10.5f}\t{4}\t{5}\t{6}\t{7:10.5f}\t{8}'.format(count,isnfile['name'][mm[0]],dt['ra'][yz[0]],dt['dec'][yz[0]],isnfile['plate'][mm[0]],isnfile['mjd'][mm[0]],isnfile['fiber'][mm[0]],isnfile['Z_VI'][mm[0]],'Multi-epochSubSample')
    print '--'*21
    for k in range(len(allPMFlist)):
        if ((allPMFlist['plate'][k] not in tplates) & (allPMFlist['mjd'][k] not in tmjds) & (allPMFlist['fiber'][k] not in tfibers) ):
            mmm = np.where((dr12q['PLATE'] == allPMFlist['plate'][k])& (dr12q['MJD'] == allPMFlist['mjd'][k]) & (dr12q['FIBERID'] == allPMFlist['fiber'][k]))[0][0]
            #print mmm
            count +=1
            print>>save, '{0}\t{1:30}\t{2:10.5f}\t{3:10.5f}\t{4}\t{5}\t{6}\t{7:10.5f}\t{8}'.format(count,dr12q['SDSS_NAME'][mmm],dr12q['RA'][mmm],dr12q['DEC'][mmm],dr12q['PLATE'][mmm],dr12q['MJD'][mmm],dr12q['FIBERID'][mmm],dr12q['Z_VI'][mmm],'DR12Q')
            
    

def plotnumerousBALSystems():
    text_font = {'fontname':'Times New Roman', 'size':'14'}
    pp = PdfPages('NumerousBALsystems_BAL_plot_spectra.pdf') 
    linelist = np.genfromtxt('/Users/vzm83/Proposals/linelist_speccy.txt',usecols=(0,1,2),dtype=('|S10',float,'|S5'),names=True)
    color=['black','red','blue','green','brown','cyan','magenta','gold']
    df = np.genfromtxt('NumerousBALsystems_subsample.txt',names=['idx','name','plate','mjd','fiber','Z_VI','tag'],dtype=(int,'|S30',int,int,int,float,'|S30'))
    for i in range(len(df)):
        if df['name'][i][0:3] == 'Tar':
            specdir = 'BALQSOs_spectra'
        else:
            specdir = 'BOSS_BALDATA_allDR12'
        if df['plate'][i] >=10000:
            PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(df['plate'][i],df['mjd'][i],df['fiber'][i])
        else:
            PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(df['plate'][i],df['mjd'][i],df['fiber'][i])
        print i, PMF
        zvi = df['Z_VI'][i]
        fig,ax=plt.subplots(figsize=(15,8))
        data1 = fits.open(os.path.join(specdir,PMF))[1].data
        data2 = fits.open(os.path.join(specdir,PMF))[2].data
        minflux=np.median((data1['flux']*(data1['and_mask'] == 0)).copy())
        maxflux =np.std((data1['flux']*(data1['and_mask'] == 0)).copy())
        ax.plot(10**data1['loglam']/(1.0+zvi),gaussian_filter1d(data1['flux'],2),color='black',alpha=0.5,label=PMF.split('.')[0][5:])
        ax.plot(10**data1['loglam']/(1.0+zvi),1.0/np.sqrt(data1['ivar']),color='black',alpha=0.1)
        ax.set(xlabel='Rest wavelength ($\AA$)',ylabel='Flux',ylim=(-2,minflux+3.0*maxflux), title=df['name'][i])
        xlim,ylim=ax.get_xlim(),ax.get_ylim()
        #string1 = 'SDSS J{0}\tZ\_VI: {1:4.4f}\tN$\_{{spec}}$: {2}'.format(fsm['SDSS_NAME'][xx[0]],zvi,len(plates))
        string1 = 'RA: {0:5.4f}\t DEC: {1:5.4f}  \tZ\_VI: {2:4.4f}\tN$\_{{spec}}$: {3}'.format(data2['PLUG_RA'][0],data2['PLUG_DEC'][0],zvi,1)
        print string1,xlim,ylim
        ax.text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.05*(ylim[1] - ylim[0]), string1,color='red',fontsize=18)
                
        obslambda = linelist['lambda']#*(1.+zvi)
        x = np.where((obslambda > xlim[0]) & (obslambda < xlim[1]))[0]
	plotlambda = obslambda[x]
	plotname = linelist['Name'][x]
	plota_e = linelist['a_e'][x]
	#print plotlambda
        for k in range(len(plotlambda)):
	    if plota_e[k].strip() == 'Abs.' : 
	        ax.axvline(x=plotlambda[k], color='lawngreen', linestyle=':')
	        ax.text(plotlambda[k],ylim[0]+0.75*(ylim[1]-ylim[0]),plotname[k],color='Orange',ha='center',rotation=90,**text_font)
	    else :
	        ax.axvline(x=plotlambda[k], color='lightblue', linestyle=':')
	        ax.text(plotlambda[k],ylim[0]+0.75*(ylim[1]-ylim[0]),plotname[k],color='Brown',ha='center',rotation=90,**text_font)
    
        ax.legend(loc=1)
        fig.tight_layout()
        fig.savefig(pp,format='pdf')
        #fig.savefig(os.path.join('BAL_Plots','Numsspectra_overplot_'+df['name'][i]+'.jpg'),format='jpg')
        #sdlfj=raw_input()
    pp.close()
 
def makeNormBALQSOStack():
    df = np.genfromtxt('highestSNspectra_PMF_master_sample.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S20',int,int,int,float,float))
    specdir='BALQSOs_spectra'
    pl = np.genfromtxt('PowerlawFits_Master_Fullsample_output.txt',names=['name','plate','mjd','fiber','Z_VI','sn','plaw0','plaw1','chi2','dof','chi2pdof','mfl'],dtype=('|S20',int,int,int,float,float,float,float,float,int,float,int))
    tmp = np.genfromtxt('tmpfilemedian.txt',names=['name','pmf','med'],dtype=('|S20','|S25',float))
    ux=np.where(tmp['med'] == np.inf)[0]  
    nspec = len(df)
    #outfile=open('tmpfilemedian.txt','w')
    zsort = np.argsort(df['Z_VI'])
    ldf =df[zsort]
    
    olam0 = 3800 ; olam1 = 9000 ; odisp = 0.5
    rlam0 = 1215 ; rlam1 = 3200 ; rdisp = 0.25
    
    owave = olam0+np.arange((olam1-olam0)/odisp)*odisp
    rwave = rlam0+np.arange((rlam1-rlam0)/rdisp)*rdisp
    
    oimage = np.zeros((nspec,len(owave)))
    rimage = np.zeros((nspec,len(rwave)))
    #for i in range(len(ldf)):
    for i in range(nspec):
        zz=np.where((pl['plate']==ldf['plate'][i]) &   (pl['mjd']==ldf['mjd'][i]) & (pl['fiber']==ldf['fiber'][i]))[0][0]
        print zz
        if ldf['plate'][i] >=10000:
            PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(ldf['plate'][i],ldf['mjd'][i],ldf['fiber'][i])
        else:
            PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(ldf['plate'][i],ldf['mjd'][i],ldf['fiber'][i])
        print i, PMF
        zvi = ldf['Z_VI'][i]
    
        data1 = fits.open(os.path.join(specdir,PMF))[1].data
        towave = 10**data1.loglam
        tflux = data1.flux
        and_mask = data1.and_mask
        
        xx=np.where(and_mask==0)[0]
        towave = towave[xx]; tflux=tflux[xx]
        trwave = towave/(1.0+zvi)
        rpow =  powerlawFunc(trwave,pl['plaw0'][zz],pl['plaw1'][zz])


        flux = tflux/rpow

        #print>>outfile, df['name'][i],PMF,np.median(flux)
        if ((tmp['med'][i] == np.inf) | (tmp['med'][i] < 0.1) ):
            flux = tflux/np.median(tflux[towave>3000])
        #if tmp['med'][i] < 0.02:
        #    fig,ax=plt.subplots(2,1,figsize=(15,8))
        #    ax[0].plot(towave,flux,label=df['name'][i])
        #    ax[0].legend('best')
        #    ax[1].plot(towave,tflux)
        #    ax[1].plot(towave,rpow)
        #    plt.show()
        of = interp1d(towave,flux,bounds_error = False, fill_value = 0)
        rf = interp1d(trwave,flux,bounds_error = False, fill_value = 0)
        ioflux = of(owave)
        irflux = rf(rwave)
        oimage[i] = ioflux
        rimage[i] = irflux

    fig,ax=plt.subplots(1,2,figsize=(20,8))
    ax[0].imshow(oimage,origin='lower',aspect='auto',vmax=2, vmin=0,extent=[min(owave), max(owave), min(ldf['Z_VI']), max(ldf['Z_VI'])])
    ax[1].imshow(rimage,origin='lower',aspect='auto',vmax=2, vmin=0,extent=[min(rwave), max(rwave), min(ldf['Z_VI']), max(ldf['Z_VI'])])
    ax[0].set(xlabel='Observed Wavelength (\AA)',ylabel='Redshift',title='Observer frame')
    ax[1].set(xlabel='Rest Wavelength (\AA)',ylabel='Redshift',title='QSO rest frame')
    fig.suptitle('SDSS BAL Factory',fontsize=24)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('SDSS-BAL-Factory-poster.jpg')
    fig.savefig('SDSS-BAL-Factory-poster.pdf')
    plt.show()
    #outfile.close()
    

def VI_stats():
    files = glob.glob('/Users/vzm83/SDSSIV_BALQSO/VI_out/*.npz')
    total = len(files)
    keys = ['_hibal_','_lobal_','_felobal_','_manynarrowsys_','_j0300analog_','_reddened_','_ironemitter_','_redshifted_','_accn_','_stable_','_emergence_','_disappear_','_lo2hibal_','_hi2lobal_','_lo2febal_','_fe2lobal_','_coordvar_','_xtreme_','_CIV_','_SiiV_','_AlIII_','_MgII_','_FeII_','_FeIII_','_AlII_']
    Llist =['Must get a follow up','May be interesting','Typical Source']
    
    print '--'*51
    for item in keys:
        count =0    
        for f in files:
            data = np.load(f)
            if data[item] == True:
                count +=1
        print '{0:30}\t\t{1}\t\t{2:3.4f}'.format(item,count,float(count)/total*100.)
    print '--'*51
    for ll in Llist:
        count =0    
        for f in files:
            data = np.load(f)
            if data['_LIST_'] == ll:
                count +=1
        print '{0:30}\t\t{1}\t\t{2:3.4f}'.format(ll,count,float(count)/total*100.)
    print '--'*51


def ChangeBALClass():
    files = glob.glob('/Users/vzm83/SDSSIV_BALQSO/VI_out/*.npz')
    total = len(files)
    keys = ['_lo2hibal_','_hi2lobal_','_lo2febal_','_fe2lobal_']
    Llist =['Must get a follow up','May be interesting','Typical Source']
    
    
    print '--'*51
    for item in keys:
        count =0
        appendlist=[]
        #print item
        print '--'*51
        for f in files:
            data = np.load(f)
            if data[item] == True:
                count +=1
                appendlist.append(f.split('.')[0].split('_')[-1])
        print item, appendlist
        print '{0:30}\t\t{1}\t\t{2:3.4f}'.format(item,count,float(count)/total*100.)

    #lo2hibal = ['Target0147', 'Target1171', 'Target1796', 'Target2040', 'Target2131', 'Target2216', 'Target2349', 'Target2953']
    #hi2lobal = ['Target0028', 'Target0055', 'Target0404', 'Target0735', 'Target1619', 'Target1693', 'Target1869', 'Target2239', 'Target2357', 'Target2384', 'Target2800']
    #lo2febal = ['Target2989']
    #fe2lobal = ['Target0035', 'Target0704', 'Target1622', 'Target2989']


def HETProposal_2019_3():
    targetsname =['Target2989','Target0704','Target2800','Target2357','Target2953','Target0659','Target2477']
    targettype=['FeLoBAL to LoBAL','FeLoBAL to LoBAL','HiBAL to LoBAL','HiBAL to LoBAL','LoBAL to HiBAL',' Variability in Redshifted BAL','Variability in Redshifted BAL']
    text_font = {'fontname':'Times New Roman', 'size':'14'}
    #pp = PdfPages('Fullanalysis_BAL_overplot_Plaw_spectra.pdf') 
    specdir = 'SDSSIV_BALdata'
    linelist = np.genfromtxt('/Users/vzm83/Proposals/linelist_speccy.txt',usecols=(0,1,2),dtype=('|S10',float,'|S5'),names=True)
    color=['black','red','blue','green','brown','cyan','magenta','gold','purple','black','red','blue','green','brown','cyan']
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    #fsm = fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))
    snfile = np.genfromtxt('SN1700_estimates_Master_full_sample_sources.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))
    fig,ax=plt.subplots(7,1,figsize=(15,15))
    for i in range(len(targetsname)):
        if 1==1:
            nplates=[];nmjds=[];nfibers=[];nsn=[]
            zx = np.where(data['name'] == targetsname[i])[0][0]
            plates = data['plate'][zx]
            mjds = data['mjd'][zx]
            fibers = data['fiber'][zx]
            print data['name'][zx]
            xx=np.where(fsm['name'] == data['name'][zx])[0]
            yy=np.where(snfile['name'] == data['name'][zx])[0]
            
            #idx = np.argsort(snfile['sn'][yy])[::-1]
            idx = np.argsort(snfile['mjd'][yy])[::-1]
            for ii in range(len(idx)):
                print '{0}\t{1}\t{2}\t{3:5.2f}'.format(snfile['plate'][yy[idx[ii]]],snfile['mjd'][yy[idx[ii]]],snfile['fiber'][yy[idx[ii]]],snfile['sn'][yy[idx[ii]]])
                nplates.append(snfile['plate'][yy[idx[ii]]]) ; nmjds.append(snfile['mjd'][yy[idx[ii]]]) ; nfibers.append(snfile['fiber'][yy[idx[ii]]]) ; nsn.append(snfile['sn'][yy[idx[ii]]])
            
            print nplates,nmjds,nfibers,nsn
            if len(xx)>0:
                zvi = fsm['Z_VI'][xx[0]]
                print xx,data['name'][i],fsm['name'][xx]
                minflux =[] ;maxflux=[]
                for j in range(len(nplates)):
                    if nplates[j] >=10000:
                        PMF = 'spec-{0:05d}-{1:05d}-{2:04d}.fits'.format(nplates[j],nmjds[j],nfibers[j])
                    else:
                        PMF = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(nplates[j],nmjds[j],nfibers[j])
                    print PMF
                    if os.path.isfile(os.path.join(specdir,PMF)):
                        data1 = fits.open(os.path.join(specdir,PMF))[1].data
                        minflux.append(np.median((data1['flux']*(data1['and_mask'] == 0)).copy()))
                        maxflux.append(np.std((data1['flux']*(data1['and_mask'] == 0)).copy()))

                        wave1 = (10**(data1.loglam))#/(1.0+zvi)
                        flux1 = data1.flux ; error1 = 1.0/np.sqrt(data1.ivar)
                        flux2= savitzky_golay(flux1,15,2)
                        mask1 = data1.and_mask
                        wave1 = wave1[np.where(mask1==0)[0]]
                        flux1 = flux1[np.where(mask1==0)[0]]
                        error1 = error1[np.where(mask1==0)[0]]
                        
                        nw,nf,nerr = linefreeRegions(wave1,flux1,error1)
                        plaw,pcoeff = fitPowerlaw(nw,nf,nerr)
                        nnw,nnf,nnerr = maskOutliers(nw,nf,nerr,plaw[0],plaw[1])
                        for jj in range(10):
                            plaw,pcoeff = fitPowerlaw(nnw,nnf,nnerr)
                            print plaw
                            if jj !=max(range(10)):
                                nnw,nnf,nnerr = maskOutliers(nnw,nnf,nnerr,plaw[0],plaw[1])
                        plchi2 = np.sum((nnf - powerlawFunc(nnw,plaw[0],plaw[1]))/nnerr)**2 
                        print len(nnw)
                        #if j==0:
                        #    print '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:5.5f}\t{6:10.5f}\t{7:10.5f}\t{8:10.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),1)    
                        #    print>>outtxt, '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:5.5f}\t{6:10.5f}\t{7:10.5f}\t{8:10.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),1)  
                        #if j!=0:
                        #    print '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:5.5f}\t{6:10.5f}\t{7:10.5f}\t{8:10.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),0)    
                        #    print>>outtxt, '{0}\t{1}\t{2}\t{3}\t{4:5.5f}\t{5:10.5f}\t{6:10.5f}\t{7:10.5f}\t{8:5.5f}\t{9}\t{10:10.5f}\t{11}'.format(data['name'][i],nplates[j],nmjds[j],nfibers[j],zvi,nsn[j],plaw[0],plaw[1],plchi2,len(nnw)-2,plchi2/(len(nnw)-2),0)
                        if j ==0:
                            refamp=plaw[0];refindex=plaw[1]
                  

                        if j < 8:
                            if j ==0:
                                ax[i].plot(wave1,gaussian_filter1d(flux1,2),color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                ax[i].plot(wave1,error1,color=color[j],alpha=0.1)
                            if j !=0:
                                if i==2:
                                    if j==1:
                                        ax[i].plot(wave1,gaussian_filter1d(flux1,2)/1.2,color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                
                                elif i==1:
                                    if j==2:
                                        ax[i].plot(wave1,gaussian_filter1d(flux1,2)*2.3,color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                    else :
                                        ax[i].plot(wave1,gaussian_filter1d(flux1,2),color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                elif i==6:
                                    if j==1:
                                        ax[i].plot(wave1,gaussian_filter1d(flux1,2)*1.3,color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                    else:
                                        ax[i].plot(wave1,gaussian_filter1d(flux1,2),color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])

                                elif i==3:
                                    if j==1:
                                        ax[i].plot(wave1,gaussian_filter1d(flux1,2)/1.3,color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                    else:
                                        ax[i].plot(wave1,gaussian_filter1d(flux1,2)/1.8,color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                else:
                                    ax[i].plot(wave1,gaussian_filter1d(flux1,2),color=color[j],alpha=0.5,label=PMF.split('.')[0][5:])
                                ax[i].plot(wave1,error1,color=color[j],alpha=0.1)
                                #ax[i].plot(wave1,powerlawFunc(wave1,plaw[0],plaw[1]),lw=3,ls='--',color=color[j],alpha=0.9)
                                #ax[i].plot(nnw,nnf,'.',color=color[j],alpha=0.9)
                        else:
                            if j ==0:
                                ax[i].plot(wave1,gaussian_filter1d(flux1,2),color=plt.cm.RdYlBu(j*300),alpha=0.5,label=PMF.split('.')[0][5:])
                                ax[i].plot(wave1,error1,color=color[j],alpha=0.1)
                            if j !=0:
                                ax[i].plot(wave1,gaussian_filter1d(flux1,2)*(refamp/plaw[0])*(wave1)**(refindex-plaw[1]),color=plt.cm.RdYlBu(j*300),alpha=0.5,label=PMF.split('.')[0][5:])
                                ax[i].plot(wave1,error1,color=plt.cm.RdYlBu(j*300),alpha=0.1)
                                #ax[i].plot(wave1,powerlawFunc(wave1,plaw[0],plaw[1]),lw=3,ls='--',color=plt.cm.RdYlBu(j*300),alpha=0.9)
                        if j == len(plates)-1:
                            if (len(maxflux) > 0):
                                if i == 6:
                                    ax[i].set(xlabel='Observed wavelength ($\AA$)',ylabel='Flux',ylim=(-2,max(minflux)+4.0*max(maxflux)))
                                else:
                                    ax[i].set(ylabel='Flux',ylim=(-2,max(minflux)+3.0*max(maxflux)))
                            xlim,ylim=ax[i].get_xlim(),ax[i].get_ylim()
                            #string1 = 'SDSS J{0}\tZ\_VI: {1:4.4f}\tN$\_{{spec}}$: {2}'.format(fsm['SDSS_NAME'][xx[0]],zvi,len(plates))
                            string1 = 'RA: {0:5.4f}\t DEC: {1:5.4f}  \tZ\_VI: {2:4.4f}\tN$\_{{spec}}$: {3}'.format(fsm['ra'][xx[0]],fsm['dec'][xx[0]],zvi,len(plates))
                            print string1,xlim,ylim
                            ax[i].text(xlim[0]+0.05*(xlim[1] - xlim[0]),ylim[1]-0.2*(ylim[1] - ylim[0]), string1,fontsize=18)
                            ax[i].text(xlim[0]+0.55*(xlim[1] - xlim[0]),ylim[1]-0.25*(ylim[1] - ylim[0]), targettype[i],fontsize=18)
                    
                            obslambda = linelist['lambda']*(1.+zvi)
                            x = np.where((obslambda > 3500) & (obslambda < 8500))[0]
	                    plotlambda = obslambda[x]
	                    plotname = linelist['Name'][x]
	                    plota_e = linelist['a_e'][x]
	                    #print plotlambda
	                    for k in range(len(plotlambda)):
	                        if plota_e[k].strip() == 'Abs.' : 
	                	        ax[i].axvline(x=plotlambda[k], color='lawngreen', linestyle=':')
	                	        ax[i].text(plotlambda[k],ylim[0]+0.65*(ylim[1]-ylim[0]),plotname[k],color='Orange',ha='center',rotation=90,**text_font)
	                        else :
	                	        ax[i].axvline(x=plotlambda[k], color='lightblue', linestyle=':')
	                	        ax[i].text(plotlambda[k],ylim[0]+0.65*(ylim[1]-ylim[0]),plotname[k],color='Brown',ha='center',rotation=90,**text_font)
                lrsb = [3700,7000]#*(1.+zvi)
                lrsr = [6500,10500]#*(1.+zvi)
                if i==0:
                    ax[i].plot(lrsb,[ylim[1]-0.85*(ylim[1] - ylim[0]),ylim[1]-0.85*(ylim[1] - ylim[0])],color='blue',ls='-',lw=5)
                    ax[i].plot(lrsr,[ylim[1]-0.9*(ylim[1] - ylim[0]),ylim[1]-0.9*(ylim[1] - ylim[0])],color='red',ls='-',lw=5)
                    ax[i].text(5000,ylim[1]-0.8*(ylim[1] - ylim[0]),'LRS2-B Coverage',fontsize=18)
                    ax[i].text(7500,ylim[1]-0.85*(ylim[1] - ylim[0]),'LRS2-R Coverage',fontsize=18)
 
                ax[i].legend(loc=1)
        fig.tight_layout()
        fig.savefig('HET_Proposal_3_BALcahngingclasses.jpg')
    plt.show()

def BCC_plotContinuumFits():
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    #fsm = fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))
    snfile = np.genfromtxt('SN1700_estimates_Master_full_sample_sources.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))


    targets = np.genfromtxt('BALchangeClass/target_list_4LC_changingclass.txt',names=['name','ra','dec'],dtype=('|S20',float,float))
    for i in range(len(targets['name'])):
    #for i in range(len(targets['name'][0:3])):
        target = targets['name'][i]
        z = fsm['Z_VI'][np.where(fsm['name'] == target)[0][0]]
        print target,z
        xx = np.where(data['name'] == target)[0][0]
        nplates=[];nmjds=[];nfibers=[];nsn=[]
        plates = data['plate'][xx]
        mjds = data['mjd'][xx]
        fibers = data['fiber'][xx]
        print data['name'][xx] ,plates
        fig,ax=plt.subplots(4,2,figsize=(25,15))
        medianflux=[];stdflux=[]
        coords = SkyCoord(targets['ra'][i],targets['dec'][i],unit='degree',frame='icrs')
        print coords
        #sfd = SFDQuery()
        #eb_v = sfd(coords)
        #dered_flux1 = dered_flux(3.1*eb_v,wave1,flux1)
        for j in range(len(plates)):
            if plates[j] >=10000:
                qsoNormfile = 'pyqsofit/Cluster_Result_Run1/norm-{0:05d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
                empcaNormfile = 'EMPCA_norm/normPCA-{0:05d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
                NempcaNormfile = 'EMPCA_norm_48365/normPCA-{0:05d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
            else:
                qsoNormfile = 'pyqsofit/Cluster_Result_Run1/norm-{0:04d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
                empcaNormfile = 'EMPCA_norm/normPCA-{0:04d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
                NempcaNormfile = 'EMPCA_norm_48365/normPCA-{0:04d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
            print target, qsoNormfile,empcaNormfile
            if os.path.isfile(qsoNormfile):
                qsonorm = np.genfromtxt(qsoNormfile,names=['wave','norm','normerr','flux','fluxerr','cont','conterr'],dtype=(float,float,float,float,float,float,float))
                #
                medianflux.append(np.median(qsonorm['flux']))
                stdflux.append(np.std(qsonorm['flux']))
                ax[j][0].plot(qsonorm['wave'],qsonorm['flux'],color='black',alpha=0.5,label=os.path.basename(qsoNormfile).split('.')[0].split('orm-')[1])
                #ax[j][0].plot(qsonorm['wave'],qsonorm['fluxerr'],color='black',alpha=0.5)
                ax[j][0].plot(qsonorm['wave'],qsonorm['cont'],color='red',lw=3,ls='--',alpha=1,label='PowerLaw')
                #ax[j][0].plot(qsonorm['wave'],qsonorm['conterr'],color='red',lw=2,ls='--',alpha=1)
                ax[j][0].set_ylim(np.median(qsonorm['flux'])-2.0*np.std(qsonorm['flux']),np.median(qsonorm['flux'])+4.0*np.std(qsonorm['flux']))
                ax[j][0].legend(loc=1)
                ax[j][0].set(ylabel='Flux')
                #
                #
                #medianflux.append(np.median(qsonorm['norm']))
                #stdflux.append(np.std(qsonorm['norm']))
                #ax[j][0].plot(qsonorm['wave'],qsonorm['norm'],color='black',alpha=0.5,label=os.path.basename(qsoNormfile).split('.')[0].split('orm-')[1])
                #ax[j][0].plot(qsonorm['wave'],qsonorm['normerr'],color='black',ls='--',alpha=0.5)
                #ax[j][0].set_ylim(np.median(qsonorm['norm'])-2.0*np.std(qsonorm['norm']),np.median(qsonorm['norm'])+4.0*np.std(qsonorm['norm']))
                #ax[j][0].legend(loc=1)
                #ax[j][0].set(ylabel='Norm Flux')


            if j==0:
                ax[j][0].set(title='Powerlaw+emission line model:{}'.format(target))
            if j==3:
                ax[j][0].set(xlabel='Rest wavelength ($\AA$)')
            #
            if os.path.isfile(empcaNormfile):
                pcanorm = np.genfromtxt(empcaNormfile,names=['wave','pnorm','pnormerr','nnorm','nnormerr','pcont','ncont','flux','fluxerr'],dtype=(float,float,float,float,float,float,float,float,float))
                npcanorm = np.genfromtxt(NempcaNormfile,names=['wave','pnorm','pnormerr','nnorm','nnormerr','pcont','ncont','flux','fluxerr'],dtype=(float,float,float,float,float,float,float,float,float))
                ax[j][1].plot(qsonorm['wave'],qsonorm['flux'],color='black',alpha=0.5,label=os.path.basename(empcaNormfile).split('.')[0].split('PCA-')[1])
                #ax[j][1].plot(qsonorm['wave'],qsonorm['fluxerr'],color='black',alpha=0.5)
                ax[j][1].plot(pcanorm['wave'],_DeRedden(pcanorm['wave']*(1.0+z),pcanorm['pcont'],targets['ra'][i],targets['dec'][i]),color='magenta',lw=3,ls='--',alpha=1,label='EMPCA')
                ax[j][1].plot(pcanorm['wave'],_DeRedden(pcanorm['wave']*(1.0+z),pcanorm['ncont'],targets['ra'][i],targets['dec'][i]),color='green',lw=3,ls='--',alpha=1,label='H-NMF')
                #ax[j][1].plot(pcanorm['wave'],pcanorm['ncont'],color='blue',lw=3,ls='--',alpha=1)
                ax[j][1].legend(loc=1)
                ax[j][1].set(ylabel='Flux')
                ax[j][0].plot(pcanorm['wave'],_DeRedden(pcanorm['wave']*(1.0+z),pcanorm['pcont'],targets['ra'][i],targets['dec'][i]),color='blue',lw=3,ls='--',alpha=1,label='EMPCA')
                ax[j][0].plot(npcanorm['wave'],_DeRedden(npcanorm['wave']*(1.0+z),npcanorm['pcont'],targets['ra'][i],targets['dec'][i]),color='cyan',lw=3,ls='--',alpha=1,label='N_EMPCA')
                ax[j][1].plot(npcanorm['wave'],_DeRedden(npcanorm['wave']*(1.0+z),npcanorm['pcont'],targets['ra'][i],targets['dec'][i]),color='cyan',lw=3,ls='--',alpha=1,label='N_EMPCA')
                ax[j][1].set_ylim(np.median(qsonorm['flux'])-2.0*np.std(qsonorm['flux']),np.median(qsonorm['flux'])+4.0*np.std(qsonorm['flux']))
                #
                #
                #ax[j][1].plot(pcanorm['wave'],pcanorm['pnorm'],color='black',alpha=0.5,label=os.path.basename(empcaNormfile).split('.')[0].split('PCA-')[1])
                #ax[j][1].plot(pcanorm['wave'],pcanorm['pnormerr'],color='black',alpha=0.5)
                #ax[j][1].legend(loc=1)
                #ax[j][1].set(ylabel='NormFlux')
                #ax[j][1].set_ylim(np.median(qsonorm['norm'])-2.0*np.std(qsonorm['norm']),np.median(qsonorm['norm'])+4.0*np.std(qsonorm['norm']))
            if j==0:
                ax[j][1].set(title='EMPCA model:{}'.format(target))
            if j==3:
                ax[j][1].set(xlabel='Rest wavelength ($\AA$)')
        fig.tight_layout()
        #fig.savefig('BALChangeClass/NormPlots_check_{}.jpg'.format(target))
        fig.savefig('BALChangeClass/NContinuumPlots_check_{}.jpg'.format(target))
        #plt.show()



def resolve_redden():
    pp = PdfPages('BCC_Check_redden_spectra.pdf') 
    data = np.load('MasterList_Plate-MJD-Fiber.npz')
    #fsm = fits.open('Fullsample_3028_BAL_sources_crossmatch-DR12Q.fits')[1].data
    fsm = np.genfromtxt('Master_full_sample_redshift.txt',names=['name','ra','dec','Z_VI','tag'],dtype=('|S25',float,float,float,'|S25'))
    snfile = np.genfromtxt('SN1700_estimates_Master_full_sample_sources.txt',names=['name','plate','mjd','fiber','Z_VI','sn'],dtype=('|S25',int,int,int,float,float))


    targets = np.genfromtxt('BALchangeClass/target_list_4LC_changingclass.txt',names=['name','ra','dec'],dtype=('|S20',float,float))
    for i in range(len(targets['name'])):
    #for i in range(len(targets['name'][0:3])):
        target = targets['name'][i]
        z = fsm['Z_VI'][np.where(fsm['name'] == target)[0][0]]
        print target,z
        xx = np.where(data['name'] == target)[0][0]
        nplates=[];nmjds=[];nfibers=[];nsn=[]
        plates = data['plate'][xx]
        mjds = data['mjd'][xx]
        fibers = data['fiber'][xx]
        print data['name'][xx] ,plates
        fig,ax=plt.subplots(4,1,figsize=(25,15))
        medianflux=[];stdflux=[]
        coords = SkyCoord(targets['ra'][i],targets['dec'][i],unit='degree',frame='icrs')
        print coords
        #sfd = SFDQuery()
        #eb_v = sfd(coords)
        #dered_flux1 = dered_flux(3.1*eb_v,wave1,flux1)
        for j in range(len(plates)):
            if plates[j] >=10000:
                qsoNormfile = 'pyqsofit/Cluster_Result_Run1/norm-{0:05d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
                empcaNormfile = 'EMPCA_norm/normPCA-{0:05d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
            else:
                qsoNormfile = 'pyqsofit/Cluster_Result_Run1/norm-{0:04d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
                empcaNormfile = 'EMPCA_norm/normPCA-{0:04d}-{1:05d}-{2:04d}.txt'.format(plates[j],mjds[j],fibers[j])
            print target, qsoNormfile,empcaNormfile
            if os.path.isfile(qsoNormfile):
                qsonorm = np.genfromtxt(qsoNormfile,names=['wave','norm','normerr','flux','fluxerr','cont','conterr'],dtype=(float,float,float,float,float,float,float))
                medianflux.append(np.median(qsonorm['flux']))
                stdflux.append(np.std(qsonorm['flux']))
                ax[j].plot(qsonorm['wave'],qsonorm['flux'],color='black',alpha=0.5,label=os.path.basename(qsoNormfile).split('.')[0].split('orm-')[1])

            if os.path.isfile(empcaNormfile):
                pcanorm = np.genfromtxt(empcaNormfile,names=['wave','pnorm','pnormerr','nnorm','nnormerr','pcont','ncont','flux','fluxerr'],dtype=(float,float,float,float,float,float,float,float,float))
                dustmap_path='/Users/vzm83/Softwares/sfddata-master'
                m = sfdmap.SFDMap(dustmap_path) 
                flux_unred = pyasl.unred(pcanorm['wave']*(1.0+z),pcanorm['flux'],m.ebv(targets['ra'][i],targets['dec'][i]))

                ax[j].plot(pcanorm['wave'],flux_unred,color='red',alpha=0.5,label=os.path.basename(empcaNormfile).split('.')[0].split('PCA-')[1])
                #
            #ax[j].set_ylim(np.median(qsonorm['flux'])-2.0*np.std(qsonorm['flux']),np.median(qsonorm['flux'])+4.0*np.std(qsonorm['flux']))
            ax[j].legend(loc=1)
            ax[j].set(ylabel='Flux')
            if j==3:
                ax[j].set(xlabel='Rest wavelength ($\AA$)')
        fig.tight_layout()
        #fig.savefig('BALChangeClass/CheckReddenPlots_{}.jpg'.format(target))
        fig.savefig(pp,format='pdf')
        #plt.show()
    pp.close()
                #

def check_Continuum_Fits():
    print "Nothing"


def main():         

    print '-------'
    #merge2initialSamples()
    #getMulti_epochInfo()
    #plot_nepochs()
    #download_data(print_cmd=False)
    #hist_multiepochs()
    #cumulative_MJD()
    #plot_spPlate()
    #three_epochs_count()
    #restFrame_timescales()
    #plot_spectra()
    #createBigFits()
    #plot_BALfootprint()
    #LoHIBAL_plot_redshift()
    #checkifSpectra()
    #computeSN1700()
    #makeOverplots()
    #highestSNspectraList()
    #CrossCorr_template()
    #numerousBALSystems()
    #plotnumerousBALSystems()
    #makeNormBALQSOStack()
    #VI_stats()
    #ChangeBALClass()
    #HETProposal_2019_3()
    BCC_plotContinuumFits()
    #resolve_redden()
if __name__== "__main__":
    main()
