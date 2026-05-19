#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import quad
from scipy.special import polygamma  # For ψ'(z)
import mpmath  # Required for complex polygamma
import matplotlib.pyplot as plt


# In[2]:


#defining parameters from table 1
eta = 0.6               # Dimensionless coupling strength
wc = 20.5e12            # Bath cutoff frequency (Hz)
g = 2.46e9              # TLS-cavity coupling (Hz)
gamma_up = 0.1e9        # Pump rate (Hz) 
gamma_down = 0.25e9     # Decay rate (Hz)
Delta = 3487e12         # TLS transition frequency (Hz) - zero phonon line
hbar = 1.0545718e-34
beta = (1 / (300 * 1.38e-23)) * hbar  # So it has units 1/Hz and z becomes dimensionless


# In[3]:


#Compute ⟨D_j^-⟩_beta = ⟨D_j^+⟩_beta from equation (22a)
def D_pm(eta, wc, beta):
    return np.exp(2 * eta * (1 - (2 * polygamma(1, 1/(beta * wc))) / (beta * wc)**2))


# In[4]:


#Plotting D_pm
t_vals = np.linspace(0, 1e-12, 100)
y_vals = [D_pm(eta, wc, beta) for t in t_vals]

plt.plot(t_vals, y_vals)
plt.title("D_pm")
plt.show()


# In[5]:


#compute ⟨D_j^-(t)D_j^+⟩_beta from equation (22b)
def D_mp(t, eta, wc, beta):
    z = (1 - 1j * wc * t) / (beta * wc)
    term1 = 1 - 1 / (1 - 1j * wc * t)**2
    psi_z = mpmath.psi(1, z)
    psi_z_conj = mpmath.psi(1, mpmath.conj(z))
    psi_z0 = mpmath.psi(1, 1 / (beta * wc))
    term2 = (psi_z + psi_z_conj - 2 * psi_z0) / (beta * wc)**2

    return mpmath.exp(4 * eta * (term1 + term2))


# In[6]:


#Plotting D_mp
t_vals = np.linspace(0, 1e-13, 1000)
y_vals = [np.real(D_mp(t, eta, wc, beta)) for t in t_vals]

plt.plot(t_vals, y_vals)
plt.title("Real part of D_mp(t)")
plt.show()


# In[7]:


#compute c_beta from equation (12)
def c_beta(t,eta,wc,beta):
    return D_mp(t,eta,wc,beta) - (D_pm(eta,wc,beta))**2


# In[8]:


#Plotting c_beta
t_vals = np.linspace(0, 1e-13, 1000)
y_vals = [np.real(c_beta(t,eta, wc, beta)) for t in t_vals]

plt.plot(t_vals, y_vals)
plt.title("Real part of c_beta(t)")
plt.show()


# In[9]:


#defining the exponential decaying part
def decay(t, gamma_up, gamma_down):
    return np.exp(-0.5 * (gamma_up + gamma_down) * t) 


# In[10]:


#defining the fast oscillations
def osci(t, delta):
    return np.exp(1j * delta * t) 


# In[11]:


# Evaluate functions
t_vals = np.linspace(0, 1e-13, 1000)
test_delta = -187e12  # detuning value

c_beta1 = [np.real(c_beta(t, eta, wc, beta)) for t in t_vals]
decay1  = [np.real(decay(t, gamma_up, gamma_down)) for t in t_vals]
osci1   = [np.real(osci(t, test_delta)) for t in t_vals]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_vals, c_beta1, label='c_beta(t)', color='blue')
plt.plot(t_vals, decay1,  label='decay(t)', color='green')
plt.plot(t_vals, osci1,   label='osci(t)', color='red')

plt.xlabel('Time (s)')
plt.ylabel('Function value')
plt.title('Plot of c_beta, decay, and osci')
plt.legend()
plt.tight_layout()
plt.show()


# In[12]:


#defining the integrand of equation (11)
def integrand(t, delta, eta, wc, beta, gamma_up, gamma_down,g):
    return (2 * g**2) * decay(t, gamma_up, gamma_down) * c_beta(t, eta, wc, beta) * np.exp(1j * delta * t)


# In[13]:


test_delta = -187e12 
t_vals = np.linspace(0, 1e-13, 1000)
y_integrand = [np.real(integrand(t, test_delta, eta, wc, beta, gamma_up, gamma_down,g)) for t in t_vals]

plt.plot(t_vals, y_integrand)
plt.title("Integrand")
plt.xlabel('Time (s)')
plt.show()


# In[14]:


#Using Logarithmic Transformation
def transformed_integrand(u, delta, eta, wc, beta, gamma_up, gamma_down, g):
    #change of variables: u = ln(t), dt = e^u*du
    t = np.exp(u) 
    jacobian = np.exp(u)  # The dt/du factor
    return integrand(t, delta, eta, wc, beta, gamma_up, gamma_down, g) * jacobian


# In[15]:


#determining when the exponential becomes neglectible
def get_t_max(gamma_total):
    return -np.log(1e-16)*2/gamma_total  # Where term reaches ~1e-16


# In[16]:


tmax = get_t_max(gamma_up+gamma_down)
tmax


# In[17]:


# Using reasonable bounds
t_min = 1e-20  # Effectively zero
t_max = 1    # Where integrand becomes negligible
u_min = np.log(t_min)
u_max = np.log(t_max)


# In[18]:


#plotting the transformed integral
u_vals = np.linspace(u_min, u_max, 1000)
integrand_vals = [np.real(transformed_integrand(u, test_delta, eta, wc, beta, gamma_up, gamma_down, g)) for u in u_vals]
plt.plot(u_vals, integrand_vals)
plt.xlabel('u = log(t)')
plt.ylabel('Integrand value')
plt.title('Transformed integrand behavior')


# In[19]:


#Performing integration
#epsabs=1e-20, epsrel=1e-6, limit=5000
def compute_gamma(delta, eta, wc, beta, gamma_up, gamma_down, g):
    integral, error = quad(lambda u: np.real(transformed_integrand(u, delta, eta, wc, beta, gamma_up, gamma_down,g)), u_min, u_max)
    return integral


# In[20]:


'''# Plotting everything together
plt.figure(figsize=(10, 6))
plt.plot(t_vals, c_beta1, label='C_beta', color='blue')
plt.plot(t_vals, decay1,  label='Decay', color='green')
plt.plot(t_vals, osci1,   label='Oscillations', color='red')
plt.plot(t_vals, y_integrand,   label='Integrand', color='orange')

plt.xlabel('Time (s)')
plt.ylabel('Function value')
plt.title('Plot of the integrand, c_beta, decay, and oscillations')
plt.legend()
plt.tight_layout()
plt.show()'''


# In[21]:


'''#compute the integral from equation (11)
# ,weight='cos',wvar=delta,
upper_limit = 1
def compute_gamma(delta, eta, wc, beta, gamma_up, gamma_down, g):
    integral, error = quad(lambda t: np.real(integrand(t, delta, eta, wc, beta, gamma_up, gamma_down,g)),0, upper_limit, epsabs=1e-3, epsrel=1e-6, limit=2000)
    return integral'''


# In[22]:


#verifying experimental value B_12(3300 THz) = 1.3KHz
omega1 = 3300e12
delta1 = omega1 - Delta
B_12 = compute_gamma(delta1, eta, wc, beta, gamma_up, gamma_down, g)
print(f"B_12({omega1/1e12}THz) = {B_12/1e3} kHz")
B_21 = compute_gamma(-delta1, eta, wc, beta, gamma_up, gamma_down, g)
print(f"B_21({omega1/1e12}THz) = {B_21/1e3} kHz")


# In[23]:


#Defining the frequency range
omega_min = 3000e12  # Hz
omega_max = 4000e12  # Hz
num_points = 200     # Resolution of the plot
omega_range = np.linspace(omega_min, omega_max, num_points)


# In[24]:


omega_exp = np.load('omega_exp.npy') #Hz


# In[25]:


def compute_spectrum(omega_exp, Delta, eta, wc, beta, gamma_up, gamma_down, g):
    # Absorption rates γ(ω-Δ)
    absorption = np.array([compute_gamma(omega - Delta, eta, wc, beta, gamma_up, gamma_down, g) 
                         for omega in omega_exp])
    
    # Emission rates γ(-ω+Δ)
    emission = np.array([compute_gamma(-(omega - Delta), eta, wc, beta, gamma_up, gamma_down, g) 
                       for omega in omega_exp])

    return absorption, emission


# In[26]:


absorption_rates, emission_rates = compute_spectrum(omega_exp, Delta, eta, wc, beta, gamma_up, gamma_down, g)


# In[27]:


# Kennard-Stepanov relation: emission from absorption (equation 1.51 from Enrico's PhD thesis)
exponent = - hbar * (omega_exp - Delta) / (300 * 1.38e-23)
emission_ks = absorption_rates * np.exp(exponent)


# In[28]:


plt.figure(figsize=(10, 6))
plt.plot(omega_exp/1e12, absorption_rates/1e6, 'brown', linewidth=3, label='γ(ω-Δ) abs')
plt.plot(omega_exp/1e12, emission_rates/1e6, 'orange',linewidth=3, label='γ(-ω+Δ) emission')
plt.plot(omega_exp/1e12, emission_ks/1e6, 'blue',linewidth=3, label='γ(-ω+Δ) emission_KS')
plt.axvline(x=Delta/1e12, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero-phonon line (Δ)')

#plt.xlim(3000, 4000)
plt.ylim(0, 0.4)

plt.xlabel('Frequency (THz)')
plt.ylabel('Rate (MHz)')
plt.title('Absorption and Emission Rates')
plt.legend()
plt.grid(True)
plt.show()


# In[29]:


abs_num = absorption_rates.tolist()
np.save('abs_num.npy',abs_num)


# In[30]:


len(abs_num)


# In[ ]:




