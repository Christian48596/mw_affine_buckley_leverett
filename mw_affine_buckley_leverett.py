#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mw_buckley_leverett.py

Public, fully commented driver for the affine-constrained conservative
multiwavelet/DG solver used in the Buckley--Leverett Berea-core manuscript.

What the code solves
--------------------
The code solves the one-dimensional hyperbolic Buckley--Leverett saturation
transport equation

    dS/dt + dF(S)/dx = 0,        F(S) = (v/phi) f_w(S),

where f_w is a Corey fractional-flow function.  The numerical unknown is not a
finite-volume cell value alone.  The evolved state is a cell-local modal
coefficient array S[c,k]:

    k = 0      cell mean mode,
    k >= 1    intra-cell detail/multiwavelet modes.

The residual is assembled in conservative weak form with numerical interface
fluxes.  The left/inflow boundary condition is imposed as a linear trace
constraint on the coefficient vector and enforced by an affine projection.

Default physical case
---------------------
The parser defaults reproduce the Berea-core waterflood benchmark used in the
manuscript:

    L = 6 in, D = 1.5 in, phi = 0.20,
    Swc = 0.10, Sor = 0.20,
    mu_w = 1 cP, mu_o = 4 cP,
    Corey exponents nw = no = 2,
    injection rate q = 1 mL/min.

These are only defaults.  To use another core/stone, override any physical
quantity directly from the command line, for example

    --L 0.30 --D 0.025 --phi 0.18 --Swc 0.15 --Sor 0.25     --mu-w 1.0e-3 --mu-o 8.0e-3 --nw 2.5 --no 2.0 --q-mL-min 0.5

Recommended manuscript commands
--------------------------------
1) Generate Figure 1 and Figure 2 for the main validation case.
   This uses p=2, Rusanov flux, and a fixed midpoint probe at x=L/2 for Berea.
   The probe is explicitly set to 0.0762 m = 7.62 cm for reproducibility.

    python mw_buckley_leverett.py       --ncells 256       --p 2       --flux rusanov       --limiter tvb       --cfl 0.20       --t-end-pvi 1.50       --probe-x 0.0762       --plot       --outdir JCP_RESULTS/final_figures_rusanov_Nc256_p2

   Output:
      Figure1_Sw_vs_t_probe.pdf/png
      Figure2_profiles.pdf/png

2) Generate the manuscript resolution figures plus flux/p-sensitivity tables.
   This is an MPI parameter sweep over Nc, p, and flux.  The aggregate plots
   use p=2 as the reference order and write the flux and p-dependence results
   as tables rather than as potentially misleading two-point/nonmonotone plots.

    mpirun -np 8 mw_buckley_leverett.py       --mpi-sweep       --ncells-list 64 128 256 512       --p-list 1 2 3 4       --flux-list rusanov godunov       --limiter tvb       --cfl 0.20       --t-end-pvi 1.50       --plot-sweep       --outdir JCP_RESULTS/full_sweep_p2_tables

   Output:
      sweep_summary.csv/json
      manuscript_plots/Figure4_resolution_study_errors.pdf/png
      manuscript_plots/Figure4_resolution_study_runtime.pdf/png
      manuscript_plots/Figure6_constraint_diagnostics.pdf/png
      manuscript_plots/Table_flux_comparison.csv/tex
      manuscript_plots/Table_modal_order_sensitivity.csv/tex

Probe modes for Figure 1
------------------------
The breakthrough curve can be measured in two ways:

  * Fixed probe: use --probe-x VALUE_IN_METERS.  This is recommended for the
    final manuscript because it is completely reproducible.

  * Automatic shock probe: use --probe-mode auto-shock.  The code scans the
    independent reference solution at --probe-auto-pvi, finds the largest
    absolute saturation gradient, and uses that location as the probe.  This is
    useful when changing stone/core parameters and the shock location is not
    known in advance.

MPI note
--------
MPI parallelism is used for independent parameter sweeps only.  A single
Buckley--Leverett solve is not domain-decomposed.  This is intentional because
for the manuscript the natural parallel tasks are the independent Nc/p/flux
cases.
"""
from __future__ import annotations

import argparse, csv, json, math, os, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss, Legendre

# mpi4py is optional. Without it, the script still runs all cases serially.
try:
    from mpi4py import MPI
    HAVE_MPI4PY = True
except Exception:
    MPI = None
    HAVE_MPI4PY = False

# pywaterflood is optional. If unavailable, the script falls back to a manual
# tangent-construction Buckley--Leverett reference for the Corey curve.
try:
    from pywaterflood.buckleyleverett import Reservoir, breakthrough_sw
    HAVE_PYWATERFLOOD = True
except Exception:
    HAVE_PYWATERFLOOD = False

# Physical/core presets. Berea is the default manuscript case. The design is
# deliberately parser-driven: for another stone/core, keep --stone-preset berea
# as a template and override the physical parameters from the command line.
# Keep all dimensional quantities explicit so that JSON summaries are
# self-contained and reproducible.
STONE_PRESETS: Dict[str, Dict[str, float]] = {
    "berea": dict(phi=0.20, Swc=0.10, Sor=0.20, mu_w=1.0e-3, mu_o=4.0e-3,
                  nw=2.0, no=2.0, krw0=1.0, kro0=1.0,
                  D=1.5*0.0254, L=6.0*0.0254, q_mL_min=1.0)
}

def set_publication_style():
    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 350, "font.size": 13,
                         "axes.labelsize": 17, "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 11, "axes.linewidth": 1.1, "lines.linewidth": 2.0,
                         "font.family": "serif", "mathtext.fontset": "dejavuserif"})

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)
def savefig(fig, path_no_ext: str):
    fig.savefig(path_no_ext + ".png", bbox_inches="tight")
    fig.savefig(path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)

def clamp(a, lo, hi): return np.minimum(np.maximum(a, lo), hi)

def minmod3(a, b, c):
    if a > 0 and b > 0 and c > 0: return min(a,b,c)
    if a < 0 and b < 0 and c < 0: return max(a,b,c)
    return 0.0

@dataclass
class CoreyBL:
    """Corey fractional-flow model and scaled BL flux.

    The PDE solved by the coefficient-space method is

        dS/dt + dF(S)/dx = 0,   F(S) = (v/phi) f_w(S),

    where f_w is the Corey fractional-flow function. Saturations are clipped
    inside [Sw_lo, Sw_hi] during nonlinear flux evaluation to avoid unphysical
    mobility values after high-order reconstruction near shocks.
    """
    phi: float; Swc: float; Sor: float; Sw_init: float; Sw_inj: float
    mu_w: float; mu_o: float; nw: float; no: float; krw0: float; kro0: float
    q_m3_day: float; A: float
    @property
    def v_darcy(self): return self.q_m3_day / self.A
    @property
    def flux_prefactor(self): return self.v_darcy / self.phi
    @property
    def Sw_lo(self): return min(self.Swc, self.Sw_init, self.Sw_inj)
    @property
    def Sw_hi(self): return max(self.Swc, self.Sw_init, self.Sw_inj)
    def Se(self, Sw):
        return clamp((np.asarray(Sw, dtype=float)-self.Swc)/(self.Sw_inj-self.Swc), 0.0, 1.0)
    def fw(self, Sw):
        Sw = clamp(np.asarray(Sw, dtype=float), self.Sw_lo, self.Sw_hi)
        Se = self.Se(Sw)
        lw = self.krw0*Se**self.nw/self.mu_w
        lo = self.kro0*(1-Se)**self.no/self.mu_o
        return lw/(lw+lo+1e-300)
    def F(self, Sw): return self.flux_prefactor*self.fw(Sw)
    def dF(self, Sw):
        Sw = np.asarray(Sw, dtype=float)
        h = 1e-7
        Sp = clamp(Sw+h, self.Sw_lo, self.Sw_hi); Sm = clamp(Sw-h, self.Sw_lo, self.Sw_hi)
        return (self.F(Sp)-self.F(Sm))/np.maximum(Sp-Sm, 1e-15)
    def max_wave_speed(self, n=20000):
        ss = np.linspace(self.Sw_lo, self.Sw_hi, n)
        return float(np.max(np.abs(self.dF(ss))))

def make_model(params):
    """Create the Corey model and compute the pore-volume time scale.

    q is entered in mL/min because this is the natural laboratory unit for the
    Berea-core benchmark. It is converted to m^3/day so the solver time unit is
    days. Output plots also report minutes for readability.
    """
    D=float(params['D']); L=float(params['L']); A=math.pi*D**2/4
    q_m3_day=float(params['q_mL_min'])*1e-6*60*24
    m=CoreyBL(phi=float(params['phi']), Swc=float(params['Swc']), Sor=float(params['Sor']),
              Sw_init=float(params.get('sw_init', params['Swc'])),
              Sw_inj=float(params.get('sw_inj', 1-params['Sor'])),
              mu_w=float(params['mu_w']), mu_o=float(params['mu_o']), nw=float(params['nw']), no=float(params['no']),
              krw0=float(params['krw0']), kro0=float(params['kro0']), q_m3_day=q_m3_day, A=A)
    t_pv = m.phi*L*A/q_m3_day
    return m,L,A,t_pv

class BLReference:
    """Fast Buckley--Leverett reference profile for the same Corey flux.

    The reference is not produced by the multiwavelet solver. It uses either
    pywaterflood's breakthrough saturation or, if pywaterflood is unavailable,
    a manual tangent construction for the entropy shock. This gives an
    independent profile against which the coefficient-space solver is compared.
    """
    def __init__(self, model: CoreyBL, L: float, A: float):
        self.m=model; self.L=L; self.A=A
        self.Swf=self._breakthrough_sw_py_or_manual()
        self.Sw_branch=np.linspace(model.Sw_inj, self.Swf, 4000)
        self.v_branch=np.asarray(model.dF(self.Sw_branch), dtype=float)
    def _breakthrough_sw_py_or_manual(self):
        if HAVE_PYWATERFLOOD:
            try:
                r=Reservoir(phi=self.m.phi, viscosity_oil=self.m.mu_o, viscosity_water=self.m.mu_w,
                            sat_oil_r=self.m.Sor, sat_water_c=self.m.Swc, sat_gas_c=0.0,
                            n_oil=self.m.no, n_water=self.m.nw, flow_cross_section=self.A)
                s=float(breakthrough_sw(r))
                if np.isfinite(s) and abs(s-self.m.Swc)>1e-6: return s
            except Exception:
                pass
        Sw=np.linspace(self.m.Swc+1e-8, self.m.Sw_inj, 80001)
        f=self.m.fw(Sw); df=np.gradient(f,Sw)
        slope=(f-self.m.fw(self.m.Swc))/(Sw-self.m.Swc+1e-30)
        return float(Sw[np.argmin(np.abs(df-slope))])
    def sw_profile_at_time(self, t_days, x):
        x=np.asarray(x,dtype=float); xf=self.v_branch*t_days
        shock=float(self.m.dF(self.Swf))*t_days
        order=np.argsort(xf); xs=xf[order]; ss=self.Sw_branch[order]
        xs,idx=np.unique(xs, return_index=True); ss=ss[idx]
        inj_edge=xs.min()
        out=np.empty_like(x)
        mask0=x>=shock; mask1=x<=inj_edge; mask=~(mask0|mask1)
        out[mask0]=self.m.Swc; out[mask1]=self.m.Sw_inj; out[mask]=np.interp(x[mask], xs, ss)
        return clamp(out, self.m.Sw_lo, self.m.Sw_hi)

class FastLegendreMW:
    """Cell-local orthonormal Legendre basis used as modal multiwavelet space.

    Each cell carries p modes. Mode k=0 is the mean mode; modes k>=1 are detail
    coefficients. The basis is discontinuous across cell interfaces, and the
    conservative coupling is therefore carried by numerical fluxes.
    """
    def __init__(self, L, ncells, p, qorder=None):
        self.L=float(L); self.ncells=int(ncells); self.p=int(p); self.h=self.L/self.ncells
        self.qorder=int(qorder or max(2*p+3,12))
        # Gauss--Legendre nodes on the reference interval and physical weights.
        # The same nodes are used for volume integration and bound checks.
        self.xi_q,self.w_q=leggauss(self.qorder); self.w_dx=self.w_q*(self.h/2)

        # Precompute basis values and derivatives on reference quadrature nodes.
        # This makes every RHS evaluation a small dense tensor contraction.
        self.P=np.zeros((p,self.qorder)); self.dP=np.zeros((p,self.qorder))
        for k in range(p):
            Pk=Legendre.basis(k); self.P[k]=Pk(self.xi_q); self.dP[k]=Pk.deriv()(self.xi_q)
        # Orthonormal scaling on the physical cell:
        #   psi_k(x) = sqrt((2k+1)/h) P_k(xi).
        self.norm=np.sqrt((2*np.arange(p)+1)/self.h)
        self.phi_q=self.norm[:,None]*self.P
        self.dphi_dx_q=self.norm[:,None]*self.dP*(2/self.h)
        # Trace vectors at xi=-1 and xi=+1. Only the left trace of the first
        # cell is used in the affine inflow constraint M s = S_inj.
        self.phi_left=self.norm*((-1.0)**np.arange(p)); self.phi_right=self.norm.copy()
        self.M_left=np.zeros((ncells,p)); self.M_left[0,:]=self.phi_left
        self.MM_left=float(np.sum(self.M_left*self.M_left))
    @property
    def ndof(self): return self.ncells*self.p
    def centers(self): return (np.arange(self.ncells)+0.5)*self.h
    def coeff_constant(self, val):
        # For a constant saturation, only the mean mode is nonzero because
        # psi_0 = 1/sqrt(h), hence s_0 = mean*sqrt(h).
        S=np.zeros((self.ncells,self.p)); S[:,0]=val*math.sqrt(self.h); return S
    def cell_means(self,S): return S[:,0]/math.sqrt(self.h)
    def eval_centers(self,S):
        # Legendre values at xi=0
        vals=np.array([Legendre.basis(k)(0.0) for k in range(self.p)])*self.norm
        return S@vals
    def eval_at(self,S,x):
        x=np.asarray(x); flat=x.ravel(); cells=np.clip(np.floor(flat/self.h).astype(int),0,self.ncells-1)
        out=np.empty_like(flat,dtype=float)
        for c in np.unique(cells):
            m=cells==c; xi=2*(flat[m]-c*self.h)/self.h-1
            V=np.vstack([Legendre.basis(k)(xi) for k in range(self.p)])
            out[m]=S[c]@(self.norm[:,None]*V)
        return out.reshape(x.shape)
    def impose_left_trace(self,S,g,mode='full'):
        """Project coefficients onto the imposed left trace.

        mode='full' applies the minimum-norm correction in all first-cell modes.
        mode='detail' applies the correction only in modes k>=1, preserving the
        first-cell average. For p=1, detail-only enforcement is impossible and
        the method falls back to full correction.
        """
        out=S.copy(); defect=g-float(np.sum(self.M_left*out))
        if abs(defect) <= 1e-15:
            return out
        if mode == 'detail' and self.p > 1:
            md = np.zeros_like(self.M_left); md[0,1:] = self.phi_left[1:]
            denom = float(np.sum(md*md))
            if denom > 0.0:
                out += (defect/denom)*md
                return out
        out += (defect/self.MM_left)*self.M_left
        return out
    def project_left_tangent(self,R,mode='full'):
        """Project residual onto the tangent space of the trace constraint."""
        out=R.copy(); defect=float(np.sum(self.M_left*out))
        if abs(defect) <= 1e-15:
            return out
        if mode == 'detail' and self.p > 1:
            md = np.zeros_like(self.M_left); md[0,1:] = self.phi_left[1:]
            denom = float(np.sum(md*md))
            if denom > 0.0:
                out -= (defect/denom)*md
                return out
        out -= (defect/self.MM_left)*self.M_left
        return out
    def traces_left(self,S): return S@self.phi_left
    def traces_right(self,S): return S@self.phi_right
    def values_q(self,S): return S@self.phi_q
    def bound_limiter(self,S,lo,hi):
        """Zhang--Shu style detail rescaling at quadrature/interface points.

        The cell mean is clipped only as a safety measure, then the detail modes
        are uniformly rescaled so reconstructed values at the monitored points
        stay in [lo, hi]. For p<=1 there are no detail modes to rescale.
        """
        out=S.copy(); means=np.clip(self.cell_means(out),lo,hi); out[:,0]=means*math.sqrt(self.h)
        if self.p<=1: return out
        vals=out@self.phi_q; vL=out@self.phi_left; vR=out@self.phi_right
        vmin=np.minimum(np.min(vals,axis=1), np.minimum(vL,vR)); vmax=np.maximum(np.max(vals,axis=1), np.maximum(vL,vR))
        theta=np.ones(self.ncells)
        mask=vmax>hi+1e-14; theta[mask]=np.minimum(theta[mask], (hi-means[mask])/(vmax[mask]-means[mask]+1e-300))
        mask=vmin<lo-1e-14; theta[mask]=np.minimum(theta[mask], (means[mask]-lo)/(means[mask]-vmin[mask]+1e-300))
        theta=np.clip(theta,0,1); out[:,1:]*=theta[:,None]
        return out
    def troubled_limiter(self,S,lo,hi,kind='tvb',beta=1.0,M=0.0):
        """Troubled-cell limiter for shock control.

        kind='tvb' keeps the mean and reconstructs only a limited linear detail
        from neighboring means. Higher modes are removed in troubled cells.
        kind='flatten' keeps only the mean in troubled cells. Smooth cells are
        left unchanged. This is deliberately conservative with respect to cell
        averages, which is essential for the manuscript diagnostics.
        """
        out=S.copy(); means=np.clip(self.cell_means(out),lo,hi); out[:,0]=means*math.sqrt(self.h)
        if self.p<=1 or kind in ('none','bounds'): return out
        vals=out@self.phi_q; vL=out@self.phi_left; vR=out@self.phi_right
        sqrt3h=math.sqrt(3/self.h); tol=M*self.h*self.h+1e-13
        for c in range(self.ncells):
            ml=means[c-1] if c>0 else lo; mr=means[c+1] if c<self.ncells-1 else means[c]
            local_min=max(lo,min(ml,means[c],mr)); local_max=min(hi,max(ml,means[c],mr))
            vmin=min(float(np.min(vals[c])), float(vL[c]), float(vR[c])); vmax=max(float(np.max(vals[c])),float(vL[c]),float(vR[c]))
            troubled=(vmin<local_min-tol) or (vmax>local_max+tol)
            if not troubled: continue
            new=np.zeros(self.p); new[0]=means[c]*math.sqrt(self.h)
            if kind=='tvb':
                raw=sqrt3h*out[c,1]
                dL=beta*(means[c]-ml); dR=beta*(mr-means[c])
                edge=minmod3(raw,dL,dR); edge=max(min(edge,hi-means[c]),lo-means[c])
                new[1]=edge/sqrt3h
            # kind=flatten keeps only mean
            out[c]=new
        return out

class FullyMWSolver:
    """Semidiscrete conservative residual plus SSP-RK3 time stepping."""
    def __init__(self,basis,model,flux='rusanov',limiter='tvb',tvb_beta=1.0,tvb_M=0.0,affine_mode='detail'):
        self.b=basis; self.m=model; self.flux=flux; self.limiter=limiter; self.tvb_beta=tvb_beta; self.tvb_M=tvb_M; self.affine_mode=affine_mode
        self.alpha=model.max_wave_speed()
    def numerical_flux_array(self,SL,SR):
        """Evaluate interface fluxes for arrays of left/right states.

        Rusanov is robust and cheap. Godunov is evaluated by sampling the scalar
        flux between the two states; this is sufficient for the manuscript
        comparison and avoids deriving a special-case closed form for each
        Corey parameter set.
        """
        m=self.m; SL=clamp(SL,m.Sw_lo,m.Sw_hi); SR=clamp(SR,m.Sw_lo,m.Sw_hi)
        FL=m.F(SL); FR=m.F(SR)
        if self.flux=='central': return 0.5*(FL+FR)
        if self.flux=='rusanov': return 0.5*(FL+FR)-0.5*self.alpha*(SR-SL)
        if self.flux in ('godunov-sampled','godunov'):
            out=np.empty_like(SL)
            for i,(a,b) in enumerate(zip(SL,SR)):
                if a<=b: ss=np.linspace(a,b,80); out[i]=np.min(m.F(ss))
                else: ss=np.linspace(b,a,80); out[i]=np.max(m.F(ss))
            return out
        raise ValueError(self.flux)
    def rhs(self,S):
        """Assemble the conservative weak residual in coefficient space.

        Volume term:  int F(S_h) d_x psi_k dx.
        Surface term: -Fhat_{c+1/2} psi_k(right) + Fhat_{c-1/2} psi_k(left).

        For the mean mode this reduces exactly to the conservative flux
        difference update. The final tangent projection removes any residual
        component that would move the state out of the inflow trace constraint.
        """
        b=self.b; m=self.m
        Sq=clamp(b.values_q(S),m.Sw_lo,m.Sw_hi); Fq=m.F(Sq)
        R=(Fq*b.w_dx[None,:])@b.dphi_dx_q.T
        left=b.traces_left(S); right=b.traces_right(S)
        # Build interface states. For v>0 the left boundary is physical inflow,
        # while the right boundary is outflow and therefore uses the interior
        # state on both sides of the numerical flux.
        SL=np.empty(b.ncells+1); SR=np.empty(b.ncells+1)
        SL[0]=m.Sw_inj; SR[0]=left[0]
        SL[1:b.ncells]=right[:-1]; SR[1:b.ncells]=left[1:]
        SL[b.ncells]=right[-1]; SR[b.ncells]=right[-1]
        Fhat=self.numerical_flux_array(SL,SR)
        R -= Fhat[1:,None]*b.phi_right[None,:]
        R += Fhat[:-1,None]*b.phi_left[None,:]
        return b.project_left_tangent(R, self.affine_mode)
    def clean(self,S):
        """Apply stage cleanup: trace projection, bounds, troubled-cell limiting.

        The trace projection is applied before and after limiting because the
        limiter can slightly perturb the boundary trace through detail changes.
        """
        b=self.b; m=self.m
        S=b.impose_left_trace(S,m.Sw_inj,self.affine_mode)
        if self.limiter!='none': S=b.bound_limiter(S,m.Sw_lo,m.Sw_hi)
        S=b.troubled_limiter(S,m.Sw_lo,m.Sw_hi,kind=self.limiter,beta=self.tvb_beta,M=self.tvb_M)
        S=b.impose_left_trace(S,m.Sw_inj,self.affine_mode)
        return S
    def step_ssprk3(self,S,dt):
        S0=self.clean(S); S1=self.clean(S0+dt*self.rhs(S0)); S2=self.clean(0.75*S0+0.25*(S1+dt*self.rhs(S1)))
        return self.clean((1/3)*S0+(2/3)*(S2+dt*self.rhs(S2)))

def parse_args():
    # Command-line interface. The parser help is intentionally explicit because
    # this script is used to produce reproducible JCP tables and figures.
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--stone-preset', choices=['berea'], default='berea',
                   help='Physical preset used as default values; override individual parameters for another stone/core')
    for name in ['D','L','phi','Swc','Sor','mu_w','mu_o','nw','no','krw0','kro0','q_mL_min']:
        p.add_argument('--'+name.replace('_','-'),type=float,default=None)
    p.add_argument('--sw-init',type=float,default=None); p.add_argument('--sw-inj',type=float,default=None)
    p.add_argument('--level',type=int,default=8); p.add_argument('--ncells',type=int,default=None,help='Override --level and use an explicit number of cells')
    p.add_argument('--ncells-list',type=int,nargs='*',default=None,help='Resolution sweep, e.g. --ncells-list 64 128 256 512')
    p.add_argument('--p',type=int,default=3); p.add_argument('--p-list',type=int,nargs='*',default=None,help='Polynomial-order sweep, e.g. --p-list 1 2 3 4')
    p.add_argument('--qorder', type=int, default=None,
                   help='Gauss-Legendre quadrature order. Default is max(2*p+3,12).')
    p.add_argument('--flux',choices=['rusanov','godunov','godunov-sampled','central'],default='rusanov')
    p.add_argument('--flux-list',choices=['rusanov','godunov','godunov-sampled','central'],nargs='*',default=None,help='Flux sweep')
    p.add_argument('--affine-mode',choices=['detail','full'],default='detail',help='detail preserves first-cell mean when p>1')
    p.add_argument('--limiter',choices=['none','bounds','tvb','flatten'],default='tvb')
    p.add_argument('--tvb-beta',type=float,default=1.0); p.add_argument('--tvb-M',type=float,default=0.0)
    p.add_argument('--cfl',type=float,default=0.20); p.add_argument('--t-end-pvi',type=float,default=1.50)
    p.add_argument('--snapshot-pvis',type=float,nargs='*',default=[0.05,0.10,0.20,0.35,0.50,0.80,1.20,1.50])
    # Breakthrough-probe control for Figure 1.
    #
    # fixed/default behavior:
    #   If --probe-x is given, the code uses that physical position in meters.
    #   If --probe-x is omitted and --probe-mode midpoint, the code uses L/2.
    #
    # automatic behavior:
    #   With --probe-mode auto-shock, the code scans the independent reference
    #   solution, finds the steepest saturation transition at --probe-auto-pvi,
    #   and uses that x-location as the breakthrough probe. This is useful when
    #   changing rock/core parameters and you want the probe to sit where the
    #   shock/front is actually present.
    p.add_argument('--probe-x',type=float,default=None,help='Fixed breakthrough probe location in meters. Overrides --probe-mode.')
    p.add_argument('--probe-mode',choices=['midpoint','auto-shock'],default='midpoint',help='How to choose the Figure 1 probe if --probe-x is not supplied')
    p.add_argument('--probe-auto-pvi',type=float,default=0.20,help='PVI at which auto-shock scans the reference profile for the steepest front')
    p.add_argument('--probe-scan-points',type=int,default=4000,help='Number of x-points used by --probe-mode auto-shock')
    p.add_argument('--probe-samples',type=int,default=300)
    p.add_argument('--plot',action='store_true',help='Plot per-case breakthrough/profile figures')
    p.add_argument('--plot-sweep',action='store_true',help='After a list/sweep run, automatically plot manuscript-ready sweep figures from sweep_summary.csv')
    p.add_argument('--outdir',type=str,default='OUT_BEREA_FULLY_MW_FAST_LIMITED')
    p.add_argument('--mpi-sweep',action='store_true',help='Distribute sweep cases over mpi4py ranks')
    return p.parse_args()

def merged_params(args):
    params=dict(STONE_PRESETS[args.stone_preset]); mp={'D':args.D,'L':args.L,'phi':args.phi,'Swc':args.Swc,'Sor':args.Sor,'sw_init':args.sw_init,'sw_inj':args.sw_inj,'mu_w':args.mu_w,'mu_o':args.mu_o,'nw':args.nw,'no':args.no,'krw0':args.krw0,'kro0':args.kro0,'q_mL_min':args.q_mL_min}
    for k,v in mp.items():
        if v is not None: params[k]=v
    params.setdefault('sw_init',params['Swc']); params.setdefault('sw_inj',1-params['Sor']); return params

def front_position(x,Sw,lo,hi):
    thr=lo+0.5*(hi-lo); idx=np.where(Sw>thr)[0]
    return float(x[idx[-1]]) if len(idx) else 0.0

def choose_probe_x(args, ref: BLReference, model: CoreyBL, L: float, t_pv: float):
    """Choose the breakthrough probe location used for Figure 1.

    The standard manuscript choice is the midpoint, x=L/2. If the user selects
    --probe-mode auto-shock, the code uses the independent reference profile to
    scan the domain at a selected PVI and places the probe at the largest
    absolute saturation gradient. This identifies the main displacement front
    without using the multiwavelet solution itself.

    Returns
    -------
    probe_x : float
        Probe location in meters.
    probe_source : str
        Human-readable description stored in JSON and printed to screen.
    """
    # Explicit user input always has priority. This is the most reproducible
    # option for a final paper figure, e.g. --probe-x 0.0762 for x=L/2 in Berea.
    if args.probe_x is not None:
        px = float(args.probe_x)
        return float(np.clip(px, 0.0, L)), 'user-fixed'

    if args.probe_mode == 'midpoint':
        return 0.5*L, 'midpoint'

    # Automatic shock/front detection from the independent reference solution.
    # The scan avoids the endpoints because boundary layers or the imposed
    # injection state can otherwise dominate the numerical gradient.
    nscan = max(100, int(args.probe_scan_points))
    xscan = np.linspace(0.0, L, nscan)
    pvi = float(np.clip(args.probe_auto_pvi, 1.0e-8, max(args.t_end_pvi, 1.0e-8)))
    Sw_ref = ref.sw_profile_at_time(pvi*t_pv, xscan)
    grad = np.abs(np.gradient(Sw_ref, xscan))

    margin = max(2, int(0.02*nscan))
    interior = np.arange(margin, nscan-margin)
    if interior.size == 0 or not np.any(np.isfinite(grad[interior])):
        return 0.5*L, 'auto-shock-fallback-midpoint'

    imax = int(interior[np.nanargmax(grad[interior])])
    px = float(xscan[imax])

    # If the selected PVI has already pushed the shock out of the domain or has
    # not yet formed a resolvable internal transition, fall back to the midpoint.
    # This prevents pathological choices very close to x=0 or x=L.
    if px <= 0.02*L or px >= 0.98*L or grad[imax] <= 1.0e-12:
        return 0.5*L, 'auto-shock-fallback-midpoint'
    return px, f'auto-shock-reference-gradient-pvi-{pvi:.6g}'

def run_case(args, case: Optional[Dict[str, Any]]=None, rank: int=0, size: int=1):
    """Run one simulation case and return a compact metrics dictionary.

    A "case" is one choice of Nc, p, and flux. The function writes all raw data
    needed for postprocessing, then returns a summary row used by the global
    sweep_summary.csv file.
    """
    if case:
        for key, value in case.items():
            setattr(args, key, value)
    t_wall0 = time.perf_counter()
    set_publication_style(); ensure_dir(args.outdir)
    params=merged_params(args); model,L,A,t_pv=make_model(params); n=int(args.ncells if args.ncells is not None else 2**args.level)
    basis=FastLegendreMW(L,n,args.p,args.qorder); solver=FullyMWSolver(basis,model,args.flux,args.limiter,args.tvb_beta,args.tvb_M,args.affine_mode); ref=BLReference(model,L,A)
    S=basis.coeff_constant(model.Sw_init); S=solver.clean(S)
    # Explicit CFL restriction. The factor (2p+1) is a standard practical DG
    # stability scaling for modal polynomial bases. dt is then adjusted so the
    # final time lands exactly on t_end.
    dt=args.cfl*basis.h/((2*args.p+1)*solver.alpha+1e-30); t_end=args.t_end_pvi*t_pv; nsteps=int(math.ceil(t_end/dt)); dt=t_end/max(1,nsteps)
    snaps=sorted(set([0.0]+[float(x) for x in args.snapshot_pvis if 0<=x<=args.t_end_pvi]+[args.t_end_pvi])); snap_times=[x*t_pv for x in snaps]; isnap=0
    probe_x, probe_source = choose_probe_x(args, ref, model, L, t_pv)
    probe_times=np.linspace(0,t_end,args.probe_samples); iprobe=0
    x=basis.centers(); dx=basis.h; metrics=[]; probe_rows=[]
    # Global mass diagnostic for the conservative law:
    #   M(t) - M(0) + int_0^t [F_out - F_in] dt = 0.
    # We integrate the boundary fluxes by the trapezoidal rule.
    initial_mass=float(np.sum(basis.cell_means(S))*dx); flux_integral=0.0
    def record_snapshot(pvi,time,Svec):
        # Store both field data and manuscript-ready diagnostics at selected PVI.
        Sw=clamp(basis.eval_centers(Svec),model.Sw_lo,model.Sw_hi); Sr=ref.sw_profile_at_time(time,x); diff=Sw-Sr
        np.savetxt(os.path.join(args.outdir,f'Sw_profile_fully_mw_pvi{pvi:.6f}.txt'),np.c_[x,Sw,Sr,diff],header='x[m]  Sw_affine_constrained_MW  Sw_pywaterflood  diff')
        metrics.append(dict(PVI=pvi,time_days=time,time_min=time*24*60,RMSE=float(np.sqrt(np.mean(diff**2))),L1_rel=float(np.sum(np.abs(diff))*dx/(np.sum(np.abs(Sr))*dx+1e-30)),Linf=float(np.max(np.abs(diff))),front_mw_m=front_position(x,Sw,model.Sw_lo,model.Sw_hi),front_ref_m=front_position(x,Sr,model.Sw_lo,model.Sw_hi),left_trace=float(np.sum(basis.M_left*Svec)), trace_error=abs(float(np.sum(basis.M_left*Svec))-model.Sw_inj), min_Sw=float(np.min(Sw)), max_Sw=float(np.max(Sw)), mass=float(np.sum(basis.cell_means(Svec))*dx), mass_defect=float(abs(np.sum(basis.cell_means(Svec))*dx-initial_mass+flux_integral))))
    def record_probe(time,Svec):
        # Breakthrough curve at the probe location, usually x=L/2.
        mw=float(clamp(basis.eval_at(Svec,np.array([probe_x]))[0],model.Sw_lo,model.Sw_hi)); rr=float(ref.sw_profile_at_time(time,np.array([probe_x]))[0]); probe_rows.append((time,mw,rr,mw-rr))
    while isnap<len(snap_times) and snap_times[isnap]<=1e-15: record_snapshot(snaps[isnap],0,S); isnap+=1
    while iprobe<len(probe_times) and probe_times[iprobe]<=1e-15: record_probe(0,S); iprobe+=1
    t=0.0
    for step in range(1,nsteps+1):
        # Main time loop. Sold is retained so snapshots/probes can be linearly
        # interpolated to requested output times without forcing tiny final steps.
        Sold=S.copy(); told=t
        Fleft_old = model.F(model.Sw_inj)
        Fright_old = model.F(clamp(basis.traces_right(Sold)[-1],model.Sw_lo,model.Sw_hi))
        t=step*dt; S=solver.step_ssprk3(S,dt)
        Fleft_new = model.F(model.Sw_inj)
        Fright_new = model.F(clamp(basis.traces_right(S)[-1],model.Sw_lo,model.Sw_hi))
        flux_integral += 0.5*dt*((Fright_old-Fleft_old)+(Fright_new-Fleft_new))
        while iprobe<len(probe_times) and probe_times[iprobe]<=t+1e-15:
            th=(probe_times[iprobe]-told)/(t-told) if t>told else 0; record_probe(probe_times[iprobe],(1-th)*Sold+th*S); iprobe+=1
        while isnap<len(snap_times) and snap_times[isnap]<=t+1e-15:
            th=(snap_times[isnap]-told)/(t-told) if t>told else 0; Sq=solver.clean((1-th)*Sold+th*S); record_snapshot(snaps[isnap],snap_times[isnap],Sq); isnap+=1
    probe_arr=np.array(probe_rows,float); np.savetxt(os.path.join(args.outdir,'Sw_probe_time_fully_mw.txt'),probe_arr,header=f'time[days] Sw_affine_constrained_MW_at_x={probe_x}[m] Sw_pywaterflood diff')
    if metrics:
        with open(os.path.join(args.outdir,'validation_metrics_fully_mw.csv'),'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(metrics[0].keys())); w.writeheader(); w.writerows(metrics)
    wall_seconds=time.perf_counter()-t_wall0
    summary=dict(method='AFFINE_CONSTRAINED_MW_DG',not_fv_backbone=True,level=args.level,ncells=n,p=args.p,ndof=n*args.p,flux=args.flux,limiter=args.limiter,affine_mode=args.affine_mode,cfl=args.cfl,dt_days=dt,nsteps=nsteps,t_end_pvi=args.t_end_pvi,PV_time_days=t_pv,probe_x_m=probe_x,probe_source=probe_source,pywaterflood_available=HAVE_PYWATERFLOOD,mpi_rank=rank,mpi_size=size,wall_seconds=wall_seconds,cpu_count=os.cpu_count(),params=params,model=asdict(model))
    with open(os.path.join(args.outdir,'run_summary_fully_mw.json'),'w') as f: json.dump(summary,f,indent=2)
    if args.plot:
        fig,ax=plt.subplots(figsize=(7.4,4.6)); tt=probe_arr[:,0]*24*60; ax.plot(tt,probe_arr[:,2],label='pywaterflood reference'); ax.plot(tt,probe_arr[:,1],'--',label='affine-constrained MW'); ax.set_xlabel('time [min]'); ax.set_ylabel(rf'$S_w$ at x = {probe_x*100:.2f} cm'); ax.grid(True,alpha=.28); ax.legend(frameon=False); savefig(fig,os.path.join(args.outdir,'Figure1_Sw_vs_t_probe'))
        fig,ax=plt.subplots(figsize=(8.6,5.3)); colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i,pvi in enumerate([p for p in snaps if p>0]):
            path=os.path.join(args.outdir,f'Sw_profile_fully_mw_pvi{pvi:.6f}.txt')
            if not os.path.exists(path): continue
            d=np.loadtxt(path); color=colors[i%len(colors)]; ax.plot(d[:,0]*100,d[:,2],color=color,lw=2.1,label=f'PVI={pvi:.2f}, t={pvi*t_pv*24*60:.1f} min'); ax.plot(d[:,0]*100,d[:,1],'--',color=color,lw=2.1)
        ax.plot([],[],'k-',label='pywaterflood reference'); ax.plot([],[],'k--',label='affine-constrained MW'); ax.set_xlabel('x [cm]'); ax.set_ylabel(r'$S_w$'); ax.set_xlim(0,L*100); ax.set_ylim(model.Sw_lo-.03,model.Sw_hi+.03); ax.grid(True,alpha=.28); ax.legend(frameon=False,ncol=2,fontsize=10); savefig(fig,os.path.join(args.outdir,'Figure2_profiles'))
    print('DONE: affine-constrained MW/DG Buckley--Leverett solver'); print(f'  outdir={args.outdir}'); print(f'  ncells,p,DOF={n},{args.p},{n*args.p}'); print(f'  probe_x={probe_x:.8e} m ({probe_x*100:.3f} cm), source={probe_source}'); print(f'  dt,nsteps={dt:.6e} days,{nsteps}'); print(f'  PV time={t_pv:.6e} days = {t_pv*24*60:.3f} min')
    if metrics: print(f"  final RMSE={metrics[-1]['RMSE']:.6e}, Linf={metrics[-1]['Linf']:.6e}, left trace={metrics[-1]['left_trace']:.12f}, wall={wall_seconds:.3f}s")
    final = dict(summary)
    if metrics:
        final.update({f'final_{k}': v for k,v in metrics[-1].items()})
    return final

def build_cases(args):
    """Expand CLI lists into independent cases for serial or MPI sweeps."""
    ncells_values = args.ncells_list if args.ncells_list else [args.ncells if args.ncells is not None else 2**args.level]
    p_values = args.p_list if args.p_list else [args.p]
    flux_values = args.flux_list if args.flux_list else [args.flux]
    cases=[]
    for nc in ncells_values:
        for pp in p_values:
            for fl in flux_values:
                case_outdir=os.path.join(args.outdir, f'Nc{int(nc):04d}_p{int(pp)}_{fl.replace("-","_")}')
                cases.append(dict(ncells=int(nc), p=int(pp), flux=fl, outdir=case_outdir))
    return cases


def _to_float_array(values):
    """Convert a pandas-like/CSV column list to a clean numpy float array."""
    return np.asarray([float(v) for v in values], dtype=float)


def plot_sweep_summary(summary_csv: str, outdir: str):
    """Create manuscript-oriented aggregate plots from sweep_summary.csv.

    These figures are generated only after a sweep/list run, i.e. when the
    summary table contains multiple choices of resolution, polynomial order, or
    flux. The goal is to avoid manual postprocessing after each MPI sweep.

    Generated outputs:
      * Table_flux_comparison: Rusanov/Godunov error/runtime comparison.
      * Figure4_resolution_study: error/runtime versus number of cells.
      * Table_modal_order_sensitivity: p-sensitivity as a table.
      * Figure6_constraint_diagnostics: trace and mass diagnostics.

    The figures have no titles inside the graphics because captions in the
    manuscript provide the context.

    The function intentionally uses only Python's csv module plus matplotlib, so
    it does not require pandas on clusters.
    """
    if not os.path.exists(summary_csv):
        print(f'WARNING: cannot plot sweep; missing {summary_csv}')
        return
    with open(summary_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print('WARNING: sweep_summary.csv is empty; no aggregate plots produced')
        return

    # Keep only rows with final diagnostics. Failed/incomplete cases would lack
    # these fields and should not enter manuscript plots.
    rows = [r for r in rows if r.get('final_RMSE','') not in ('', None)]
    if not rows:
        print('WARNING: no valid final_RMSE rows found; no aggregate plots produced')
        return

    def val(r, key, default=np.nan):
        try:
            return float(r.get(key, default))
        except Exception:
            return default

    def unique_num(key):
        return sorted({val(r,key) for r in rows if not math.isnan(val(r,key))})

    def unique_str(key):
        return sorted({str(r.get(key,'')) for r in rows if str(r.get(key,''))})

    ncells_values = unique_num('ncells')
    p_values = unique_num('p')
    flux_values = unique_str('flux')

    # Choose central slices for manuscript diagnostics. The recommended
    # production case for the shock-dominated Berea benchmark is p=2, because
    # it gives the cleanest accuracy--cost compromise in the modal-order study.
    p_ref = 2.0 if 2.0 in p_values else (p_values[len(p_values)//2] if p_values else np.nan)
    nc_ref = 256.0 if 256.0 in ncells_values else (ncells_values[len(ncells_values)//2] if ncells_values else np.nan)
    flux_ref = 'rusanov' if 'rusanov' in flux_values else (flux_values[0] if flux_values else '')

    def select(**conds):
        out=[]
        for r in rows:
            ok=True
            for k,v in conds.items():
                if isinstance(v, (int,float)):
                    ok = ok and (abs(val(r,k)-float(v)) < 1e-12)
                else:
                    ok = ok and (str(r.get(k,'')) == str(v))
            if ok:
                out.append(r)
        return out

    figdir=os.path.join(outdir,'manuscript_plots')
    ensure_dir(figdir)
    set_publication_style()

    def write_table_csv_tex(name, table_rows, headers, tex_caption, tex_label):
        """Write compact manuscript tables as CSV and LaTeX.

        Flux comparison and modal-order sensitivity are clearer as tables than
        as two-point/strongly nonmonotone plots. The LaTeX file is intentionally
        plain so it can be copied directly into the manuscript and then edited.
        """
        csv_path=os.path.join(figdir, name + '.csv')
        tex_path=os.path.join(figdir, name + '.tex')
        with open(csv_path, 'w', newline='') as f:
            w=csv.writer(f); w.writerow(headers); w.writerows(table_rows)
        with open(tex_path, 'w') as f:
            f.write('\\begin{table}[t]\n')
            f.write('\\caption{' + tex_caption + '}\n')
            f.write('\\label{' + tex_label + '}\n')
            f.write('\\begin{ruledtabular}\n')
            f.write('\\begin{tabular}{' + 'c'*len(headers) + '}\n')
            f.write(' & '.join(headers) + ' \\\\n')
            f.write('\\hline\n')
            for row in table_rows:
                f.write(' & '.join(str(x) for x in row) + ' \\\\n')
            f.write('\\end{tabular}\n')
            f.write('\\end{ruledtabular}\n')
            f.write('\\end{table}\n')

    # Flux comparison at the reference resolution/order. This is written as a
    # table, not a plot, because there are only two flux choices and the
    # accuracy/runtime tradeoff is clearer numerically.
    flux_rows=select(ncells=nc_ref, p=p_ref)
    if len(flux_rows) >= 2:
        flux_rows=sorted(flux_rows, key=lambda r: str(r.get('flux','')))
        table_rows=[]
        for r in flux_rows:
            table_rows.append([
                r.get('flux',''),
                f"{val(r,'final_RMSE'):.6e}",
                f"{val(r,'final_Linf'):.6e}",
                f"{val(r,'final_trace_error'):.3e}",
                f"{val(r,'final_mass_defect'):.3e}",
                f"{val(r,'wall_seconds'):.3f}",
            ])
        write_table_csv_tex(
            'Table_flux_comparison',
            table_rows,
            ['Flux', r'$E_{\rm RMSE}$', r'$E_\infty$', 'trace error', 'mass defect', 'wall time [s]'],
            rf'Flux comparison at $N_c={int(nc_ref)}$ and $p={int(p_ref)}$.',
            'tab:flux-comparison'
        )

    # Resolution study at fixed p and flux.
    res_rows=sorted(select(p=p_ref, flux=flux_ref), key=lambda r: val(r,'ncells'))
    if len(res_rows) >= 2:
        ncs=np.array([val(r,'ncells') for r in res_rows])
        fig,ax=plt.subplots(figsize=(6.8,4.4))
        ax.loglog(ncs, [val(r,'final_RMSE') for r in res_rows], 'o-', label=r'$E_{\rm RMSE}$')
        ax.loglog(ncs, [val(r,'final_Linf') for r in res_rows], 's--', label=r'$E_\infty$')
        ax.set_xlabel(r'number of cells $N_c$')
        ax.set_ylabel('error')
        ax.grid(True,which='both',alpha=.28); ax.legend(frameon=False)
        savefig(fig, os.path.join(figdir,'Figure4_resolution_study_errors'))

        fig,ax=plt.subplots(figsize=(6.8,4.4))
        ax.loglog(ncs, [val(r,'wall_seconds') for r in res_rows], 'o-')
        ax.set_xlabel(r'number of cells $N_c$')
        ax.set_ylabel('wall time [s]')
        ax.grid(True,which='both',alpha=.28)
        savefig(fig, os.path.join(figdir,'Figure4_resolution_study_runtime'))

    # Modal-order sensitivity at fixed Nc and flux. This is written as a
    # table instead of a figure because the shock/limiter interaction can make
    # the dependence on p nonmonotone; a table is more honest and less visually
    # misleading.
    p_rows=sorted(select(ncells=nc_ref, flux=flux_ref), key=lambda r: val(r,'p'))
    if len(p_rows) >= 2:
        table_rows=[]
        for r in p_rows:
            pp=int(round(val(r,'p'))); degree=max(pp-1,0)
            table_rows.append([
                pp,
                degree,
                int(round(val(r,'ndof'))),
                int(round(val(r,'nsteps'))),
                f"{val(r,'final_RMSE'):.6e}",
                f"{val(r,'final_Linf'):.6e}",
                f"{val(r,'wall_seconds'):.3f}",
            ])
        write_table_csv_tex(
            'Table_modal_order_sensitivity',
            table_rows,
            [r'$p$', 'degree', r'$N_{\rm dof}$', r'$N_{\rm steps}$', r'$E_{\rm RMSE}$', r'$E_\infty$', 'wall time [s]'],
            rf'Modal-order sensitivity at fixed $N_c={int(nc_ref)}$ using the {flux_ref} flux. Here $p$ is the number of local modes per cell and the polynomial degree is $p-1$.',
            'tab:modal-order-sensitivity'
        )

    # Constraint/conservation diagnostics across the resolution slice.
    if len(res_rows) >= 2:
        ncs=np.array([val(r,'ncells') for r in res_rows])
        fig,ax=plt.subplots(figsize=(6.8,4.4))
        ax.semilogy(ncs, [max(val(r,'final_trace_error'),1e-18) for r in res_rows], 'o-', label='trace error')
        ax.semilogy(ncs, [max(val(r,'final_mass_defect'),1e-18) for r in res_rows], 's--', label='mass defect')
        ax.set_xlabel(r'number of cells $N_c$')
        ax.set_ylabel('diagnostic magnitude')
        ax.grid(True,which='both',alpha=.28); ax.legend(frameon=False)
        savefig(fig, os.path.join(figdir,'Figure6_constraint_diagnostics'))

    print(f'MANUSCRIPT SWEEP PLOTS written to {figdir}')

def main():
    args=parse_args()
    cases=build_cases(args)
    # MPI is used only to distribute independent sweep cases. This avoids the
    # complexity of domain decomposition and is ideal for resolution/order/flux
    # studies where each run is independent.
    if args.mpi_sweep and HAVE_MPI4PY:
        comm=MPI.COMM_WORLD; rank=comm.Get_rank(); size=comm.Get_size()
    else:
        comm=None; rank=0; size=1
        if args.mpi_sweep and not HAVE_MPI4PY:
            print('WARNING: --mpi-sweep requested, but mpi4py is unavailable. Running serial sweep.')
    local_results=[]
    for i,case in enumerate(cases):
        if i % size != rank:
            continue
        # copy args namespace so each case can safely customize outdir/ncells/p/flux
        case_args=argparse.Namespace(**vars(args))
        print(f'[rank {rank}/{size}] running case {i+1}/{len(cases)}: Nc={case["ncells"]}, p={case["p"]}, flux={case["flux"]}')
        local_results.append(run_case(case_args, case=case, rank=rank, size=size))
    if comm is not None:
        gathered=comm.gather(local_results, root=0)
    else:
        gathered=[local_results]
    if rank == 0:
        # Rank 0 collects all rows and writes a single summary table that can be
        # imported directly into LaTeX, pandas, or plotting scripts.
        results=[r for group in gathered for r in group]
        ensure_dir(args.outdir)
        if results:
            keys=sorted(set().union(*(r.keys() for r in results)))
            with open(os.path.join(args.outdir,'sweep_summary.csv'),'w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(results)
            with open(os.path.join(args.outdir,'sweep_summary.json'),'w') as f:
                json.dump(results,f,indent=2)
            print(f'SWEEP DONE: {len(results)} cases written to {args.outdir}/sweep_summary.csv')
            if args.plot_sweep:
                plot_sweep_summary(os.path.join(args.outdir,'sweep_summary.csv'), args.outdir)

if __name__=='__main__': main()
