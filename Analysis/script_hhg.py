import numpy as np
import sys
from units    import *
from hhg_tool import *
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


tb_current     =np.genfromtxt("current.csv",comments="#",delimiter=",")
polarization=np.genfromtxt("polarization.csv",comments="#",delimiter=",")

n_steps=np.shape(tb_current)[0]

# I add some initial steps for the padding
n_initial=7000

print("Number of steps : "+str(n_steps))
current=np.zeros((3,n_steps+n_initial),dtype=np.double)
time   =np.zeros((n_steps+n_initial),dtype=np.double)

dt=tb_current[1,0]*fs2aut-tb_current[0,0]*fs2aut

for it in range(n_initial):
    time[it]=it*dt

for it in range(n_steps):
    time[it+n_initial]     =tb_current[it,0]*fs2aut+dt*n_initial
    current[0,it+n_initial]=tb_current[it,1]+tb_current[it,3]  #x-dir inter + intra
    current[1,it+n_initial]=tb_current[it,2]+tb_current[it,4]  #y-dir inter + intra

n_steps=n_steps+n_initial

t_initial=0.02*fs2aut
e_versor =[1.0, 0.0, 0.0]
#e_versor =[0.0, 1.0, 0.0]
e_versor = e_versor/np.linalg.norm(e_versor)
lfreq=0.4132
idir=0
gap=3.625*2.0
#hdir=e_versor
print("E-versor ",e_versor)
hdir=e_versor

print(np.dot(hdir,e_versor))

plot_signal(data=current[0:3,:],time=time,padded=False,hdir=hdir,tstring='current',singlefig=True)
#f,psd = get_psd(data=current[0:3,:],time=time,hdir=hdir,padded=True,wind="boxcar",Npad=2000)
f,psd = get_psd(data=current[0:3,:],time=time,hdir=hdir,padded=True,wind="blackmanharris",Npad=600)
plot_psd_new(f,psd,lfreq,tstring='KG from current',singlefig=True,lmax=30,gap=gap,ymin=10e-9)
#plt.savefig('HHG_KG_1_par.pdf')
plt.savefig('HHG_KG_1_per.pdf')
plt.show()

