import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import os
import glob
from pydl.pydl.pydlutils.spheregroup import *
import pydl.pyspherematch as pshm
from pydl.pydlutils import yanny
tolerance_arcsec=1.5
tolerance_deg = tolerance_arcsec/3600.

#Readin Hewett & Wild 2010 Redshifts 
hra,hdec,hz,hzerr,first,ahz,zmethod,hplate,hmjd,hfiber = np.loadtxt('/Users/vzm83/Downloads/1045471_Supplementary_Data/mnras0408-2302-SD1.txt',usecols=(1,2,3,4,5,6,7,8,9,10)).T
hplate=[int(x) for x in hplate] ; hmjd=[int(x) for x in hmjd] ; hfiber = [int(x) for x in hfiber]


data=fits.open('Skyserver_CrossID4_17_2018_3_04_34PM.fits')[1].data
unames,ucounts = np.unique(data['name'],return_counts=True)   
spAllfile = 'spAll-v5_10_10.fits'
spAll = fits.open(spAllfile)[1].data
xx=np.where(spAll['mjd'] > 57520)
nspAll=spAll[xx]
out = open('Skyserver_analysis_output_survey_epochcounts_v5_10_10.txt','w')
out1=open('Skyserver_query_spAll_merged_3epoch_info_v5_10_10.dat','w')
all_sdsspmf=[];all_bosspmf=[];all_ebosspmf=[]
all_sdssmjd=[];all_bossmjd=[];all_ebossmjd=[]
all_ra=[];all_dec=[];all_redshift=[];all_redshifttag=[]
sdssboss = 0
sdssbosseboss = 0
for i in range(len(unames)):
    xx=np.where(data['name'] == 'Target'+str(i+1))[0]
    if len(xx)>0:
        sdsspmf=[];bosspmf=[];ebosspmf=[]
        sdssmjd=[];bossmjd=[];ebossmjd=[]
        ndata=data[xx]
        piperedshift=ndata['z'][0]
        mra=ndata['ra'][0];mdec=ndata['dec'][0]
        a,b,ds= pshm.spherematch(mra,mdec,hra,hdec,tol=tolerance_deg)
        if len(a) > 0:
            print 'Hewett & Wild',hra[b],hdec[b], hz[b]
            redshift = hz[b][0]
            redshift_tag = "HW10"
        else:
            redshift = piperedshift 
            redshift_tag = "PIPE"

        sdsscount = 0 ; bosscount=0;ebosscount=0
        for j in range(len(ndata)):
            if ndata['mjd'][j] < 55050:
                sdsscount +=1
                sdssstring = '{0:04d}-{1:05d}-{2:04d}'.format(ndata['plate'][j],ndata['mjd'][j],ndata['fiberID'][j])
                sdsspmf.append(sdssstring)
                sdssmjd.append(ndata['mjd'][j])
            elif np.logical_and(ndata['mjd'][j] >= 55050., ndata['mjd'][j] <= 56660.):
                bosscount +=1
                bossstring = '{0:04d}-{1:05d}-{2:04d}'.format(ndata['plate'][j],ndata['mjd'][j],ndata['fiberID'][j])
                bosspmf.append(bossstring)
                bossmjd.append(ndata['mjd'][j])
            else:
                ebosscount +=1
                ebossstring = '{0:04d}-{1:05d}-{2:04d}'.format(ndata['plate'][j],ndata['mjd'][j],ndata['fiberID'][j])
                ebosspmf.append(ebossstring)
                ebossmjd.append(ndata['mjd'][j])
            yy=np.where(nspAll['OBJID'] == str(ndata['objID'][0]))[0]
            if len(yy) > 0:
                mspAll = nspAll[yy]
                for jj in range(len(mspAll)):
                    if mspAll['mjd'][jj] > 56660:
                        ebosscount +=1
                        ebossstring = '{0:04d}-{1:05d}-{2:04d}'.format(mspAll['PLATE'][jj],mspAll['MJD'][jj],mspAll['FIBERID'][jj])
                        if ebossstring not in ebosspmf:
                            ebosspmf.append(ebossstring)
                            ebossmjd.append(mspAll['MJD'][jj])
        
        sdsspmf=np.array(sdsspmf);sdssmjd=np.array(sdssmjd)
        bosspmf=np.array(bosspmf);bossmjd=np.array(bossmjd)
        ebosspmf=np.array(ebosspmf);ebossmjd=np.array(ebossmjd)
        sdssinds = np.array(sdssmjd).argsort()
        sortedsdsspmf = sdsspmf[sdssinds];sortedsdssmjd = sdssmjd[sdssinds]
        bossinds = np.array(bossmjd).argsort()
        sortedbosspmf = bosspmf[bossinds];sortedbossmjd = bossmjd[bossinds]
        ebossinds = np.array(ebossmjd).argsort()
        sortedebosspmf = ebosspmf[ebossinds];sortedebossmjd = ebossmjd[ebossinds]
        
        print 'Target'+str(i+1), mra,mdec,redshift,redshift_tag
        print 'SDSS: ',sortedsdsspmf,sortedsdssmjd
        print 'BOSS: ',sortedbosspmf,sortedbossmjd
        print 'eBOSS: ',sortedebosspmf,sortedebossmjd
        #jhfksjh=raw_input()
        
        if ((sdsscount > 0) & (bosscount > 0)):
            sdssboss +=1
        if ((sdsscount > 0) & (bosscount > 0)& (ebosscount > 0)):
            sdssbosseboss +=1
            pickedsdss= sdsspmf[-1]
            pickedboss= bosspmf[-1]
            pickedeboss= ebosspmf[-1]
            print>>out1, '{0:10.5f}\t{1:10.5f}\t{2:10.5f}\t{3}\t{4}\t{5}'.format(mra,mdec,redshift,pickedsdss,pickedboss,pickedeboss)

        all_ra.append(mra);all_dec.append(mdec);all_redshift.append(redshift);all_redshifttag.append(redshift_tag)
        all_sdsspmf.append(sortedsdsspmf);all_sdssmjd.append(sortedsdssmjd)
        all_bosspmf.append(sortedbosspmf);all_bossmjd.append(sortedbossmjd)
        all_ebosspmf.append(sortedebosspmf);all_ebossmjd.append(sortedebossmjd)
        print>>out, '{0:10s}\t{1:10.5f}\t{2:10.5f}\t{3:10.5f}\t{4}\t{5}\t{6}'.format('Target'+str(i+1),ndata['ra'][0],ndata['dec'][0],ndata['z'][0],sdsscount,bosscount,ebosscount)
out.close()
out1.close()
print 'No of quasars having SDSS and BOSS:',sdssboss
print 'No of quasars having SDSS and BOSS and eBOSS:',sdssbosseboss
#np.savez('Correct_Compilation_PMF_info_skyserver_query.npz',
#        ra = all_ra ,
#        dec = all_dec ,
#        redshift = all_redshift ,
#        redshift_tag = all_redshifttag ,
#        sdsspmf = all_sdsspmf ,
#        bosspmf = all_bosspmf ,
#        ebosspmf = all_ebosspmf ,
#        sdssmjd = all_sdssmjd ,
#        bossmjd = all_bossmjd ,
#        ebossmjd = all_ebossmjd)


#skydata=np.genfromtxt('Skyserver_analysis_output_survey_epochcounts.txt',names=('name','ra','dec','z','sdsscount','bosscount','ebosscount'),dtype=('|S15',float,float,float,int,int,int))
#
#index1,index2,dist = spherematch(skydata['ra'],skydata['dec'],nspAll['RA'],nspAll['DEC'],tolerance_deg,maxmatch=0)
#cccount=0
#uindex1 = np.unique(index1)
#nsdssboss = 0
#nsdssbosseboss = 0
#
#for k in range(len(uindex1)):
#    print skydata[uindex1[k]]
#    if skydata['ebosscount'][uindex1[k]] == 0:
#        cccount+=1
#    pp=np.where(index1 == uindex1[k])[0]
#    for kk in range(len(pp)):
#        print  skydata['name'][uindex1[k]],skydata['ra'][uindex1[k]], skydata['dec'][uindex1[k]],nspAll['ra'][index2[pp[kk]]],nspAll['dec'][index2[pp[kk]]],nspAll['plate'][index2[pp[kk]]],nspAll['mjd'][index2[pp[kk]]],nspAll['fiberid'][index2[pp[kk]]]
#        skydata['ebosscount'][uindex1[k]] +=1
#    
#    print '--'*102
#    #jdldf=raw_input()
#
#
#for l in range(len(skydata)):
#    if ((skydata['sdsscount'][l] > 0) & (skydata['bosscount'][l] > 0)):
#            nsdssboss +=1
#    if ((skydata['sdsscount'][l]>0) & (skydata['bosscount'][l]> 0)& (skydata['ebosscount'][l] > 0)):
#            nsdssbosseboss +=1
#
#print cccount
#print 'No of quasars having SDSS and BOSS:',nsdssboss
#print 'No of quasars having SDSS and BOSS and eBOSS:',nsdssbosseboss
def init_plotting():
    plt.rcParams['font.size'] = 15
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.25*plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.25*plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.25*plt.rcParams['font.size']
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = 2

    #plt.gca().spines['right'].set_color('none')
    #plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')

def negativeRAs(ralist):
    newralist=[]
    for ra in ralist:
        if ra >= 300 :
            t=ra - 360.0
            ra = t
        newralist.append(ra)
    return newralist

ra1,dec1=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_NGC.txt').T
ra2,dec2=np.loadtxt('boss_survey_outline_points_sorted_ra-60_300_SGC.txt').T

baltargets = yanny.read_table_yanny(filename='master-BAL-targets-yanny-format1.dat.txt',tablename='TARGET')

#baltargets = yanny.read_table_yanny(filename='green01-TDSS_FES_VARBALmaster1.par.txt',tablename='TARGET')
newtargets=yanny.read_table_yanny('targeting13-explained_more_TDSS_FES_VARBAL_201605.dat',tablename='TARGET')
print len(baltargets)
print baltargets['ra'],baltargets['dec']
baltargetsra = baltargets['ra']#np.concatenate((baltargets['ra'],newtargets['ra']))
baltargetsdec = baltargets['dec']#np.concatenate((baltargets['dec'],newtargets['dec']))

#baltargetsra = np.concatenate((baltargets['ra'],newtargets['ra']))
#baltargetsdec = np.concatenate((baltargets['dec'],newtargets['dec']))

data = np.genfromtxt('Skyserver_analysis_output_survey_epochcounts_v5_10_10.txt',names=['name','ra','dec','z','sdss','boss','eboss'],dtype=('|S10',float,float,float,int,int,int))
triple_epochs = 0

sdss_boss=0

sdss_eboss=0

boss_eboss = 0

for i in range(len(data)):
    if ((data['sdss'][i] >0) & (data['boss'][i] > 0) & (data['eboss'][i]> 0)): 
        triple_epochs+=1
    if ((data['sdss'][i] >0) & (data['boss'][i] > 0)):
        sdss_boss+=1
    if ((data['sdss'][i] >0) & (data['eboss'][i] > 0)):
        sdss_eboss+=1
    if ((data['boss'][i] > 0) & (data['eboss'][i]> 0)):
        boss_eboss+=1
print '--'*51
print 'Summary of BAL QSO observations'
print '--'*51
print 'No of BAL QSOs with SDSS epochs  ',len(data[np.where(data['sdss']>0)])
print 'No of BAL QSOs with BOSS epochs  ',len(data[np.where(data['boss']>0)])
print 'No of BAL QSOs with eBOSS epochs  ',len(data[np.where(data['eboss']>0)])
print '--'*51
print 'No of BAL QSOs with SDSS and BOSS epochs  ',sdss_boss
print 'No of BAL QSOs with SDSS and eBOSS epochs  ',sdss_eboss
print 'No of BAL QSOs with BOSS and eBOSS epochs  ',boss_eboss
print '--'*51
print 'No of BAL QSOs with SDSS, BOSS and eBOSS epochs  ',triple_epochs
print '--'*51

x1 = np.where(data['sdss'] > 0)[0]
x2 = np.where(data['boss'] > 0)[0]
x3 = np.where(data['eboss'] > 0)[0]

x12 = np.where((data['sdss'] > 0) & (data['boss'] > 0))[0]
x13 = np.where((data['sdss'] > 0) & (data['eboss'] > 0))[0]
x23 = np.where((data['boss'] > 0) & (data['eboss'] > 0))[0]

x123 = np.where((data['sdss'] > 0) & (data['boss'] > 0)&  (data['eboss'] > 0))[0]
fig,(ax,ax1)=plt.subplots(1,2,figsize=(25,10))
init_plotting()
ax.plot(ra1,dec1,'-',color='black',alpha=0.5)
ax.plot(ra2,dec2,'-',color='black',alpha=0.5)
ax1.plot(ra1,dec1,'-',color='black',alpha=0.5)
ax1.plot(ra2,dec2,'-',color='black',alpha=0.5)


ax.plot(negativeRAs(baltargetsra),baltargetsdec,'.',markersize=2,color='black',label='Parent Sample'+'(#'+str(len(baltargetsra))+')')
ax.plot(negativeRAs(data['ra'][x1]),data['dec'][x1],'+',markersize=2,color='red',label='SDSS Sample'+'(#'+str(len(x1))+')')
ax.plot(negativeRAs(data['ra'][x2]),data['dec'][x2],'o',markerfacecolor='none',markersize=3,color='blue',label='BOSS Sample'+'(#'+str(len(x2))+')')
ax.plot(negativeRAs(data['ra'][x3]),data['dec'][x3],'s',markerfacecolor='none',markersize=4,color='magenta',label='eBOSS Sample'+'(#'+str(len(x3))+')')


ax1.plot(negativeRAs(baltargetsra),baltargetsdec,'.',markersize=2,color='black',label='Parent Sample'+'(#'+str(len(baltargetsra))+')')
ax1.plot(negativeRAs(data['ra'][x12]),data['dec'][x12],'x',markersize=2,color='red',label='SDSS + BOSS Sample'+'(#'+str(len(x12))+')')
ax1.plot(negativeRAs(data['ra'][x13]),data['dec'][x13],'o',markerfacecolor='none',markersize=3,lw=2,color='blue',label='SDSS + eBOSS Sample'+'(#'+str(len(x13))+')')
ax1.plot(negativeRAs(data['ra'][x23]),data['dec'][x23],'s',markerfacecolor='none',markersize=4,color='magenta',label='BOSS + eBOSS Sample'+'(#'+str(len(x23))+')')
#ax1.plot(negativeRAs(baltargetsra[indexs5]),baltargetsdec[indexs5],'d',markersize=2,color='green',label='SEQUELS :'+ str(len(indexs5)))
ax1.plot(negativeRAs(data['ra'][x123]),data['dec'][x123],'v',markerfacecolor='none',markersize=5,color='cyan',label='SDSS + BOSS + eBOSS Sample'+'(#'+str(len(x123))+')')

ax.set_xlabel(r'RA',fontsize=20)
ax1.set_xlabel(r'RA',fontsize=20)
ax.set_ylabel(r'DEC',fontsize=20)
ax1.set_ylabel(r'DEC',fontsize=20)
ax1.legend(loc=2)
ax.legend(loc=2)
fig.suptitle('BAL Targets on Sky')

fig.tight_layout()
fig.savefig('SDSSIV_BALQSO_Observation_status_multiple_epochs_v5_10_10.jpeg')
fig.savefig('SDSSIV_BALQSO_Observation_status_multiple_epochs_v5_10_10.pdf')
plt.show()

