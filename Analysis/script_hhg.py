import numpy as np
import sys
from yambopy import *
from yambopy.plot  import *
import matplotlib.pyplot as plt

def plot_psd_new(f,psd,lfreq,tstring='',singlefig = True, lmax=41,ymin=10e-7, gap=None):
    l=np.arange(2,lmax,2)
    if (singlefig):
        plt.show()
        tlabel='Laser freq = '+str(lfreq)+'eV'
        ttitle = tstring
    else:
        tlabel = tstring
        ttitle = ''
    print(f)
    plt.semilogy(f/lfreq, psd,label=tlabel)
    plt.title(ttitle)
    plt.ylim([ymin,1])
    plt.xlim([0.1,lmax])
    plt.vlines(l,ymin,1,linestyles='dotted')
    plt.xticks(l)
    if gap is not None:
        plt.vlines(gap/lfreq,ymin,1,linestyles='dotted',colors='red')
    plt.xlabel('Harmonic number')
    plt.ylabel(r"PSD $\left[V^2\right]$")
    plt.legend()
    return


tb_current     =np.genfromtxt("current_test.csv",comments="#",delimiter=",")
polarization=np.genfromtxt("polarization.csv",comments="#",delimiter=",")

n_steps=np.shape(tb_current)[0]
print("Number of steps : "+str(n_steps))
current=np.zeros((3,n_steps),dtype=np.double)
time   =np.zeros((n_steps),dtype=np.double)

for it in range(n_steps):
    time[it]     =tb_current[it,0]*fs2aut
    current[0,it]=tb_current[it,1]+tb_current[it,3]  #x-dir inter + intra
    current[1,it]=tb_current[it,2]+tb_current[it,4]  #y-dir inter + intra


t_initial=0.02*fs2aut
e_versor =[1.0, 1.0, 0.0]
e_versor = e_versor/np.linalg.norm(e_versor)
lfreq=0.4132
idir=0
gap=3.625*2.0
#hdir=e_versor
print("E-versor ",e_versor)
hdir=e_versor

print(np.dot(hdir,e_versor))

plot_signal(data=current[0:3,:],time=time,padded=False,hdir=hdir,tstring='current',singlefig=True)
f,psd = get_psd(data=current[0:3,:],time=time,hdir=hdir)
plot_psd_new(f,psd,lfreq,tstring='KG from current',singlefig=True,lmax=30,gap=gap,ymin=10e-9)
plt.savefig('HHG_KG_1_par.pdf')
plt.show()

