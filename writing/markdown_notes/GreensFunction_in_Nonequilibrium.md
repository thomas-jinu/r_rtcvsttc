# Thinking about Green's Functions and Linear Response w.r.t to quantum dynamics in out-of-equilibrium

These are two sides of the same coin;

## 1. Starting from linear response, where the probe can be treated as a perturbative effect, one can derive the response functions of various experiments. 
For example the spectral function defined by the imaginary part of the single-particle Green's function is the response function for ARPES experiments.  For other probes such as INS we define correlation functions or transition amplitudes. 

There are also different types of Green's function that can be computed which are related to each other by various identities. Defining what Green's function is the response function of an experiment needs to be worked out using linear response theory (via Kubo formula). Alternatively, response functions can also be calculated directly in frequency space using the Lehmann representation. 

For probes which cannot be connected to a simple Green's functions (such as XAS and RIXS) the response functions are defined via Fermi's Golden rule with the appropriate transition amplitudes for relevant operators.

Similar arguments apply in non-equilibrium, however careful consideration is needed. The following are theory references that discuss the formalism for the two commonly used pump-probe spectroscopies. 

For tr-ARPES we have 
- [@freerickspruschkeTheoreticalDescriptionTimeResolved2009] as the original theory ;
- [@freericksWhatTwoTimes2021] for an easier read;
- [@fotsoCharacterizingNonequilibriumDynamics2020] for this using DMFT;
- [@weberfreericksElectronicMechanismThat2023] for steady state Green's functions;
- [@petrovicfreericksTheoreticalDescriptionPumpProbe2024] for tr-ARPES again;

For tr-XAS and tr-RIXS using the Keldysh formalism we have:
- [@chen2019theory] 

At a first glance, it is surprising to me that such response functions when derived takes the form of Green's function or correlation functions. However there is likely an underlying reason that I need to explore. Since response functions are response of a system to a probe integrated over time, the response theory actually might be borrowing the Green's function machine.

# 2. Green's function theory
Since the single particle Green's function is a general many-body object that can be computed for a given system, (quantum mechanical or otherwise but quantum mechanical is my knowledge base), one can calculate various Green's functions. 

In non-equilibrium particular, we use the non-equilibrium Green's function either in the Kadanoff-Baym formalism or the simpler Keldysh formalism. 

Whether the given NEGF is a response function for the experiment is for linear response theory to work out. But note the salient point here; the Green's function is a many-body object of the system. The response-function refers to how the system responds to the probe; So fundamentally the Green's function carries no information regarding the probe (such as its frequency in the case of ARPES). It so happens that the response-function has ways to connect back to the many-body Green's function)
  
# Definition of Single Particle Green's Function
From [@freericksWhatTwoTimes2021] we define the greater and lesser Green's functions for fermionic many body systems are:
$$
\begin{align}
G^>_{i,j,\sigma}(t,t^\prime) &= -i\braket{\hat{c}_{i,\sigma}(t)c^\dagger_{j,\sigma}(t^\prime)} \\
G^<_{i,j,\sigma}(t,t^\prime) &= i\braket{\hat{c}_{i,\sigma}^\dagger(t)c_{j,\sigma}(t^\prime)}
\end{align}
$$
One can define the retarded Green's function as:
$$
G^R_{i,j,\sigma} = \theta(t-t^\prime)(G^>_{i,j,\sigma}(t,t^\prime) - G^<_{i,j,\sigma}(t,t^\prime))
$$
From [@bruus_manybody_2004] we know that these are "single-particle" in the sense that we are measuring how creating(annihilating) a particle at a specific point in space and time propagates before being annihilated(created) at a later point in space and time.  As the propagation is governed by the full many-body Hamiltonian this hence is truly a many-body object.  These lesser and greater Green's functions are also referred to as propagators. 

# Real-Space to Momentum Space Fourier Transform of the Single Particle Green's function in Translational Invariant Systems.
It needs to be understood whether this translational invariance also applies in out-of-equilibrium contexts; (added as a task)
$$
\begin{align}
c^\dagger_{i,\sigma} &= \frac{1}{\sqrt{N}}\sum_{\mathbf{k}^\prime} e^{-i \mathbf{k}^\prime\cdot \mathbf{r}^\prime_i} c_{\mathbf{k^\prime,\sigma}}^\dagger \\
c_{i,\sigma} &= \frac{1}{\sqrt{N}}\sum_{\mathbf{k}} e^{i \mathbf{k}\cdot \mathbf{r}_i} c_{\mathbf{k},\sigma}
\end{align}
$$
The normalization factor is present so that the commutator relationship is the same in real-space and momentum-space
$$
\begin{align}
[c_{i,\sigma}^,c^\dagger_{j,\sigma}] &= \delta_{i,j} \\
[c_{\mathbf{k},\sigma}^,c_{\mathbf{k^\prime},\sigma}^\dagger ] &= \delta_{\mathbf{k},\mathbf{k}^\prime}
\end{align}
$$
Substituting this in to the greater Green's function we have
$$
\begin{align}
G^>_{i,j,\sigma}(t,t^\prime) &= -i\braket{\hat{c}_{i,\sigma}^(t)c^\dagger_{j,\sigma}(t^\prime)} \\
&= \frac{-i}{N} \braket{
\sum_{\mathbf{k},\mathbf{k}^\prime} e^{i \mathbf{k}\cdot \mathbf{r}_i} c_{\mathbf{k},\sigma} (t) e^{-i \mathbf{k}^\prime\cdot \mathbf{r}^\prime_j} c_{\mathbf{k^\prime,\sigma}}^\dagger(t^\prime)
} \\
&=\frac{-i}{N} \braket{
\sum_{\mathbf{k},\mathbf{k}^\prime} 
e^{i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} c_{\mathbf{k},\sigma} (t) e^{-i (\mathbf{k}^\prime - \mathbf{k})\cdot \mathbf{r}^\prime_j} c_{\mathbf{k^\prime,\sigma}}^\dagger(t^\prime)
}
\end{align}
$$
For the following expression
$$
\begin{align}
\sum_{\mathbf{k},\mathbf{k}^\prime} 
e^{-i (\mathbf{k}^\prime - \mathbf{k})\cdot \mathbf{r}^\prime_j} 
& = \sum_{\mathbf{k},\mathbf{k}^\prime} 
e^{-i (\mathbf{k}^\prime)\cdot \mathbf{r}^\prime_j} 
e^{+i (\mathbf{k}^\prime)\cdot \mathbf{r}^\prime_j}
\end{align}
$$
the momenta points in the Brilloun zone for a lattice with translational invariance is well-defined
$$
k= \frac{2\pi j}{N}, j\in[0,N)
$$
and for every $k$ there is a $k^\prime = -k$ except for $k=0$. So all the phases cancel each other except for the $k=0$ component giving. 
$$
\begin{align}
\sum_{\mathbf{k},\mathbf{k}^\prime} 
e^{-i (\mathbf{k}^\prime - \mathbf{k})\cdot \mathbf{r}^\prime_j} 
& = \sum_{\mathbf{k},\mathbf{k}^\prime} \delta_{\mathbf{k},\mathbf{k}^\prime}
\end{align}
$$
Substituting this back in:

$$
\begin{align}
G^>_{i,j,\sigma}(t,t^\prime) &= -i\braket{\hat{c}_{i,\sigma}(t)c^\dagger_{j,\sigma}(t^\prime)} \\
&= \frac{-i}{N} \braket{
\sum_{\mathbf{k},\mathbf{k}^\prime} e^{i \mathbf{k}\cdot \mathbf{r}_i} c_{\mathbf{k},\sigma} (t) e^{-i \mathbf{k}^\prime\cdot \mathbf{r}^\prime_j} c_{\mathbf{k^\prime,\sigma}}^\dagger(t^\prime)
} \\
&=\frac{-i}{N} \braket{
\sum_{\mathbf{k},\mathbf{k}^\prime} 
e^{i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} c_{\mathbf{k},\sigma} (t) \delta_{\mathbf{k},\mathbf{k}^\prime}c_{\mathbf{k^\prime,\sigma}}^\dagger(t^\prime)
} \\
&=\frac{-i}{N} \braket{
\sum_{\mathbf{k}} 
e^{i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} c_{\mathbf{k},\sigma} (t) c_{\mathbf{k,\sigma}}^\dagger(t^\prime)
}\\
&=\frac{-i}{N} \sum_{\mathbf{k}} 
e^{i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)}  \braket{
c_{\mathbf{k},\sigma} (t) c_{\mathbf{k,\sigma}}^\dagger(t^\prime)
} \\
&= \frac{1}{N} \sum_{\mathbf{k}} e^{i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} G^>_{\mathbf{k},\sigma}(t,t^\prime)
\end{align}
$$
Similarly we have
$$
G^<_{i,j,\sigma}(t,t^\prime) = \frac{1}{N} \sum_{\mathbf{k}} e^{i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} G^<_{\mathbf{k},\sigma}(t,t^\prime)
$$
The inverse Fourier transforms are, 
$$
\begin{align}
G^>_{\mathbf{k},\sigma}(t,t^\prime) &= \frac{1}{N} \sum_{i,j} e^{- i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} G^>_{i,j,\sigma}(t,t^\prime) \\
G^<_{\mathbf{k},\sigma}(t,t^\prime) &= \frac{1}{N} \sum_{i,j} e^{- i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} G^<_{i,j,\sigma}(t,t^\prime)
\end{align}
$$

# Quantities computed in DMRG++
In DMRG++ simulations I am computing without the imaginary pre-factors and fix one of the sites to the center-site. 
$$
G_{\mathrm{i,c,dmrg++}}^> = \braket{\hat{c}_{i,\sigma}(t)c^\dagger_{c,\sigma}(t^\prime)} \\
G_{\mathrm{i,c,dmrg++}}^< = \braket{\hat{c}_{i,\sigma}^\dagger(t)c_{c,\sigma}(t^\prime)}
$$
_I need to check that this is computed in the Keldysh space_

The Fourier transform is then:
$$
\begin{align}
G^>_{\mathbf{k},\sigma}(t,t^\prime) &= \frac{1}{N} \sum_{i,j} e^{- i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_j)} G^>_{i,j,\sigma}(t,t^\prime) \\
& \approx \frac{1}{N} \sum_{i} e^{- i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_c)} G^>_{i,c,\sigma}(t,t^\prime) \\
&= \frac{-i}{N} \sum_{i} e^{- i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_c)} G^>_{\mathrm{i,c,dmrg++}}(t,t^\prime) 
\end{align}
$$
Similarly
$$
G^<_{\mathbf{k},\sigma}(t,t^\prime) = \frac{i}{N} \sum_{i} e^{- i \mathbf{k}\cdot (\mathbf{r}_i - \mathbf{r}_c)} G^<_{\mathrm{dmrg++}}(t,t^\prime)
$$
We further simplify this to be local density of states for $i=c$ as
$$
\begin{align}
G^>_{\mathbf{k=0},\sigma}(t,t^\prime) &= -i G^>_{\mathrm{c,c,dmrg++}}(t,t^\prime) \\
G^<_{\mathbf{k=0},\sigma}(t,t^\prime) &= i G^<_{\mathrm{c,c,dmrg++}}(t,t^\prime)
\end{align}
$$
# Real-Time to Frequency Space Fourier Transform of the Single Particle Green's function.
Note here we have purposefully left out the defining time-translational invariance; This is so as the pump makes the Hamiltonian time dependent and hence time translational invariance is lost. 

The original full Fourier transform will be
$$
\begin{align} G^>_{\mathbf{k},\sigma}(\omega, \omega^\prime) &= -i \int_{-\infty}^{\infty} dt \int_{-\infty}^{\infty} dt^\prime \braket{e^{i \omega t}\hat{c}_{\mathbf{k},\sigma}(t)e^{-i \omega^\prime t^\prime}c^\dagger_{\mathbf{k},\sigma}(t^\prime)}  
\end{align}
$$
You can alternatively switch to Wigner coordinates (see [@fotsoCharacterizingNonequilibriumDynamics2020] or [@freericksWhatTwoTimes2021])
$$
\begin{align} 
t_{\mathrm{avg}} &= (t + t^\prime)/2 \\
t_{\mathrm{rel}} &= (t - t^\prime) 
\end{align}
$$
which gives
$$
\begin{align} 
t &= t_{\mathrm{avg}} + t_{\mathrm{rel}}/2 \\
t^\prime &= t_{\mathrm{avg}} - t_{\mathrm{rel}}/2 
\end{align}
$$
and hence
$$
G^>_{\mathbf{k=0},\sigma}(t,t^\prime) = G^>_{\mathbf{k=0},\sigma}(t_{\mathrm{avg}}+ t_{\mathrm{rel}}/2,t_{\mathrm{avg}} - t_{\mathrm{rel}}/2)
$$
The integration now becomes
$$
\begin{aligned} 
G^>_{\mathbf{k}=0,\sigma}(\nu,\Omega) &= \int_{-\infty}^{\infty} dt_{\mathrm{avg}} e^{i\Omega t_{\mathrm{avg}}} \int_{-\infty}^{\infty} dt_{\mathrm{rel}} e^{i\nu t_{\mathrm{rel}}} \\ &\quad\times G^>_{\mathbf{k}=0,\sigma} \left( t_{\mathrm{avg}}+\frac{t_{\mathrm{rel}}}{2}, t_{\mathrm{avg}}-\frac{t_{\mathrm{rel}}}{2} \right), 
\end{aligned}
$$
where
$$
\qquad \nu=\frac{\omega+\omega'}{2}, \quad \Omega=\omega-\omega
$$.

So here we have to pause; first we don't know what this double frequency Green's function is in relation to the experiment. It doesn't seem like it could be useful for any pump-probe experiment which is a frequency $\omega$ response defined at a $t_{\mathrm{delay}}$. The delay is w.r.t the pump, or the specific point in time that we refer to as time translational invariance is lost. 

One could imagine just doing a single-Fourier transform for in the $t_{\mathrm{rel}}$ axis which is the $t-t^\prime$ axis and define the following;
$$
\begin{aligned} 
G^>_{\mathbf{k}=0,\sigma}(\nu,t_{\mathrm{avg}}) &= \int_{-\infty}^{\infty} dt_{\mathrm{rel}} \ e^{i\nu t_{\mathrm{rel}}} \ G^>_{\mathbf{k}=0,\sigma} \left( t_{\mathrm{avg}}+\frac{t_{\mathrm{rel}}}{2}, t_{\mathrm{avg}}-\frac{t_{\mathrm{rel}}}{2} \right), 
\end{aligned}
$$

[@freericksWhatTwoTimes2021] discusses how there is dynamics along both $t_{\mathrm{rel}}$ and $t_{\mathrm{avg}}$ for out-of-equilibrium measurements. 

Only under specific circumstances can we choose to ignore one. Specifically we can take the Fourier transform along $t_{\mathrm{rel}}$
* if the measurements are made long after the pump (i.e H(t) = H(0)) and 
* the time-dependence along $t_{\mathrm{avg}}$ is slow compared to the $t_{\mathrm{rel}}$

"Typical parameters modeled to change as a function of $t_{\mathrm{avg}}$ are electronic/lattice temperature, scattering rate (in Drude formalism), order parameters and lattice (phonon) frequencies.

"This transformation is forcing non-equilibrium behavior into equilibrium form" and it is unclear when these approximations are valid and when they break down.

" The connection between out of equilibrium average time decay rate and equilibrium sel-fenergy breaks down (see reference 3 for experiment and reference 4 for theory)"

# Understanding the two-time Fourier transform in Wigner coordinates

Taken from [@najmiWIGNERDISTRIBUTIONTIMEFREQUENCY]. This is to connect what we know of the Green's functions to mathematics. I am unsure if it will help in my work.

The two-dimensional ambiguity function:
$$
\begin{aligned} G^>_{\mathbf{k}=0,\sigma}(\Omega,t_{\mathrm{rel}}) &= \int_{-\infty}^{\infty} dt_{\mathrm{avg}} \ e^{i\Omega t_{\mathrm{avg}}} \ G^>_{\mathbf{k}=0,\sigma} \left( t_{\mathrm{avg}}+\frac{t_{\mathrm{rel}}}{2}, t_{\mathrm{avg}}-\frac{t_{\mathrm{rel}}}{2} \right) 
\end{aligned}
$$
Consequently the Wigner distribution is:
$$
\begin{aligned} 
G^>_{\mathbf{k}=0,\sigma}(\nu,t_{\mathrm{avg}}) &= \int_{-\infty}^{\infty} dt_{\mathrm{rel}} \ e^{i\nu t_{\mathrm{rel}}} \ G^>_{\mathbf{k}=0,\sigma} \left( t_{\mathrm{avg}}+\frac{t_{\mathrm{rel}}}{2}, t_{\mathrm{avg}}-\frac{t_{\mathrm{rel}}}{2} \right)
\end{aligned}
$$
The second is the Green's function we have defined above. They are a Fourier transform pair as given below;
$$
\begin{aligned} 
G^>_{\mathbf{k}=0,\sigma}(\nu,t_{\mathrm{avg}}) &= \int \int G^>_{\mathbf{k}=0,\sigma}(\Omega,t_{\mathrm{rel}}) e^{-i\nu t_{\mathrm{rel}}}e^{- \Omega t_{\mathrm{avg}}}\ dt_{\mathrm{rel}} \ d{\Omega} 
\end{aligned}
$$
But this Fourier transform is general for the choice of an appropriate Kernel function.
$$\
\begin{aligned} G^>_{\mathbf{k}=0,\sigma}(\nu,t_{\mathrm{avg}}) &= \int \int G^>_{\mathbf{k}=0,\sigma}(\Omega,t_{\mathrm{rel}}) e^{-i\nu t_{\mathrm{rel}}}e^{- \Omega t_{\mathrm{avg}}} \ K(\Omega,t_{\mathrm{rel}}) \ dt_{\mathrm{rel}} \ d{\Omega} \end{aligned}\
$$

# Relating the Fourier Transform of the Non-equilibrium Green's Function to tr-ARPES

From [@freerickspruschkeTheoreticalDescriptionTimeResolved2009] we have

However relating this to the following is challenging;

$$
\begin{align} G^>_{\mathbf{k},\sigma}(t_{\mathrm{delay}},\omega_{\mathrm{photon}}) &= -i \int_{-\infty}^{\infty} dt \int_{-\infty}^{\infty} dt^\prime e^{i ,\omega_{\mathrm{photon}}(t - t^\prime)}\braket{\hat{c}_{\mathbf{k},\sigma}(t)c^\dagger_{\mathbf{k},\sigma}(t^\prime)} s(t-t_{\mathrm{delay}}) s(t^\prime-t_{\mathrm{delay}})  
\end{align}
$$
where 
$$
s(t-t_{\mathrm{delay}}) = \frac{1}{\sqrt{2\pi}\sigma}e^\frac{-(t-t_{\mathrm{delay}})^2}{2  \sigma^2}
$$ 
is the probe envelope function.  In equilibrium $s(t)=1$ (_but why?_). 



# Definition of the relative-time correlation function from [@Lai2019XASTr]
The single particle Green's function analogous to the definition in the paper:

$$
\begin{align}
G^>_{\mathbf{k},\sigma}(\omega,t_{\mathrm{delay}}) &= -i \int_0^\infty dt e^{i\omega t} \braket{\psi(t_{\mathrm{delay}})|e^{iHt}c_{\mathbf{k},\sigma}e^{-iHt}c_{\mathbf{k},\sigma}^\dagger | \psi(t_{\mathrm{delay}})} 
\end{align}
$$
We can take the correlation function and rewrite it as:
$$
\begin{align} &\braket{\psi(t_{\mathrm{delay}})|e^{iHt}c_{\mathbf{k},\sigma}e^{-iHt}c_{\mathbf{k},\sigma}^\dagger | \psi(t_{\mathrm{delay}})} \\ = & \braket{\psi_0|e^{iH(t+t_{\mathrm{delay}})}c_{\mathbf{k},\sigma}e^{-iH(t+t_{\mathrm{delay}})} e^{iHt_{\mathrm{delay}}} c_{\mathbf{k},\sigma}^\dagger e^{-iH(t_{delay})}| \psi_0} \\\ = & \braket{\psi_0|c_{\mathbf{k},\sigma}(t+t_{\mathrm{delay}}) c_{\mathbf{k},\sigma}^\dagger (t_{\mathrm{delay}})| \psi_0} \\\ = & \ G^>_{\mathbf{k},\sigma}(t+t_{\mathrm{delay}}, t_{\mathrm{delay}} ) \end{align}
$$
The point here is that this is now a relative-time Green's function; and it is neither equivalent to the full two-time nor the partial Fourier transform in the Wigner coordinates.



# Some identities for the non-equilibrium Green's functions;
$$
\begin{align}
[G_{\mathbf{k},\sigma}^>(t,t^\prime)]^\dagger &= -G_{\mathbf{k},\sigma}^>(t^\prime,t) \\
[G_{\mathbf{k},\sigma}^<(t,t^\prime)]^\dagger &= -G_{\mathbf{k},\sigma}^<(t^\prime,t)
\end{align}
$$

# Fourier Transform Script Density of States
## Stage 1 - Get the real-time Green's functions
1. Load all the data
2. Multiply by the appropriate imaginary prefactor
3. Get the center-center correlation function for the $t, t^\prime$ grid.