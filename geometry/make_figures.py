"""Generate the three paper figures; requires NumPy and Matplotlib."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import isqrt


def isprime(n):
    return n >= 2 and all(n % d for d in range(2, isqrt(n) + 1))

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.titlesize': 11, 'axes.labelsize': 10,
                     'pdf.fonttype': 42})


def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


fig, ax = plt.subplots(figsize=(8.2, 5.6), layout='constrained')
for m in range(1, 7):
    ax.axhspan(m*m, (m+1)**2, color='#e1ecd9' if m % 2 == 0 else '#fff1cb', alpha=.6)
for d in range(1, 26):
    k = np.arange(1, min(30, 60//d) + 1)
    color = plt.cm.tab20((d-1) % 20)
    ax.plot(np.r_[0,k], np.r_[0,k*d], color=color, alpha=.55, lw=.7)
    ax.scatter(k, k*d, color=color, s=6)
x = np.linspace(0, np.sqrt(60), 400)
ax.plot(x, x*x, color='#b52a36', lw=2, label=r'$\Pi: n=k^2$')
stars = np.arange(1,8)
ax.scatter(stars, stars*stars, marker='*', color='#b52a36', s=110, zorder=5)
for n in range(2,61):
    if isprime(n):
        ax.text(-.5, n, 'p', color='#91222b', ha='center', va='center', fontsize=7)
ax.set(xlim=(-1,30.5), ylim=(0,60.5), xlabel=r'$k$ (multiplier)', ylabel=r'$n$')
ax.legend(loc='upper right')
ax.grid(alpha=.18)
save(fig,'fig1_factor_rays')

fig, axes = plt.subplots(1,2,figsize=(9.5,3.8),layout='constrained')
for ax, low, high, title in zip(axes,[100,390],[121,410],
                              [r'(a) $W_{10}=[100,121]$',r'(b) $J_{10}=[390,410]$']):
    ax.axvspan(np.sqrt(low),21,color='#efaf52',alpha=.25)
    pts = [(k,n) for n in range(low,high+1) for k in range(1,22) if n%k==0]
    ax.scatter(*zip(*pts),s=12,color='#426c9c')
    x=np.linspace(np.sqrt(low),np.sqrt(high),200)
    ax.plot(x,x*x,color='#b52a36',lw=2)
    ax.set(xlim=(.5,21.5),ylim=(low-.5,high+.5),xlabel=r'$k$',ylabel=r'$n$',title=title)
    ax.grid(alpha=.2)
save(fig,'fig2_band_comparison')

fig, axes=plt.subplots(1,2,figsize=(9.5,4.1),layout='constrained')
for ax,low,high,xlo,xhi in [(axes[0],100,200,2,26),(axes[1],420,490,23,39)]:
    pts=[(k,n) for n in range(low,high+1) for k in range(xlo,xhi+1) if n%k==0 and n//k>=2]
    ax.scatter(*zip(*pts),s=10,color='#c8cdd2',zorder=1)
    ax.set(xlim=(xlo-.5,xhi+.5),ylim=(low,high),xlabel=r'$k$',ylabel=r'$n$')
    ax.grid(alpha=.15)
ax=axes[0]
colors=plt.cm.viridis(np.linspace(.05,.85,8))
for s,color in zip(range(21,29),colors):
    x=np.linspace(2,s-2,800)
    ax.plot(x,x*(s-x),color=color,lw=1)
    k=np.arange(2,s-1)
    ax.scatter(k,k*(s-k),s=15,color=color,zorder=3)
    h=s//2
    if s%2==0:
        ax.scatter([h],[h*h],marker='*',s=90,color=color,zorder=5)
    else:
        ax.scatter([h,h+1],[h*(h+1)]*2,s=35,facecolors='white',edgecolors=[color],zorder=5)
    ax.text(s/2+3.3,s*s/4-9+2,f'$s={s}$',fontsize=7,color=color)
x=np.linspace(10,np.sqrt(200),100)
ax.plot(x,x*x,color='#b52a36',lw=1.7,ls='--',label=r'$\Pi$')
ax.legend(loc='upper left',frameon=False)
ax.set_title(r'(a) $k+d=s$: square and pronic crests',pad=10)
ax=axes[1]
for S,color,ls in [(60,'#286eaa','-'),(61,'#bd641c','--'),(62,'#725398','-')]:
    x=np.linspace(23,39,400)
    ax.plot(x,x*(S-x)/2,color=color,lw=1.3,ls=ls,label=f'$S={S}$')
    k=np.arange(23,40)
    k=k[k%2==S%2]
    ax.scatter(k,k*(S-k)/2,color=color,s=24,zorder=4)
ax.set_title(r'(b) $k+2d=S$: residue-class sampling',pad=10)
ax.legend(loc='lower center',ncol=3,fontsize=8)
save(fig,'fig3_composite_waves')
print('Wrote three PDF and PNG figures')
