import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numba

@numba.njit
def thomas_solve(d, u, l, b):
    """
    Solve a tridiagonal system Ax = b with diagonals stored as:
    d: main diagonal (length n)
    u: upper diagonal (length n-1)
    l: lower diagonal (length n-1)
    b: right-hand side (length n)
    using the Thomas algorithm.
    """
    n = len(d)
    dprime = np.zeros(n, dtype=d.dtype)
    bprime = np.zeros(n, dtype=d.dtype)

    dprime[0] = d[0]
    cprime = np.zeros(n-1, dtype=d.dtype)
    cprime[0] = u[0]/dprime[0]
    bprime[0] = b[0]/dprime[0]

    for i in range(1, n-1):
        dprime[i] = d[i] - l[i-1]*cprime[i-1]
        cprime[i] = u[i]/dprime[i]
        bprime[i] = (b[i] - l[i-1]*bprime[i-1])/dprime[i]

    dprime[n-1] = d[n-1] - l[n-2]*cprime[n-2]
    bprime[n-1] = (b[n-1] - l[n-2]*bprime[n-2])/dprime[n-1]

    x = np.zeros(n, dtype=d.dtype)
    x[n-1] = bprime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = bprime[i] - cprime[i]*x[i+1]

    return x


def solve_schrodinger_time_dependent(psi0, L, dt, t_steps, V_func):
    """
    Solve the time-dependent Schrödinger equation with a time-dependent potential using Crank-Nicolson.

    i d/dt psi = -1/2 d^2/dx^2 psi + V(x,t)*psi
    with psi=0 at x=0,L.

    Parameters
    ----------
    psi0 : array_like
        Initial wavefunction at interior points (complex).
    L : float
        Domain length (0 to L)
    dt : float
        Time step
    t_steps : int
        Number of time steps
    V_func : function
        A function V_func(x_array, t) that returns the potential array at time t.

    Returns
    -------
    x : ndarray
        Spatial grid including boundaries.
    psi_t : ndarray
        Wavefunction at all times. Shape (t_steps, N).
    times : ndarray
        Array of time values corresponding to each time step.
    """
    N = len(psi0)
    dx = L/(N+1)
    x = np.linspace(0, L, N+2)
    x_interior = x[1:-1]  # interior points

    # alpha = dt/(4 dx^2)
    alpha = dt/(4*dx**2)

    # Time array
    times = np.arange(t_steps)*dt

    psi = psi0.copy()
    psi_t = np.zeros((t_steps, N), dtype=complex)
    psi_t[0,:] = psi

    def laplacian(psi_arr):
        N=len(psi_arr)
        # lap = np.zeros_like(psi_arr, dtype=complex)
        lap = np.zeros(psi_arr.shape, dtype=np.complex128)
        lap[0] = (psi_arr[1] - 2*psi_arr[0] + 0)
        lap[-1] = (0 - 2*psi_arr[-1] + psi_arr[-2])
        # if N > 2:
        #     lap[1:-1] = psi_arr[2:] - 2*psi_arr[1:-1] + psi_arr[:-2]
        lap[1:-1] = psi_arr[2:] - 2 * psi_arr[1:-1] + psi_arr[:-2]
        return lap

    for n in range(1, t_steps):
        t_n = times[n-1]
        t_nplus = times[n]

        #approach using the average of V(t_{n+1}) and V(t_n)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # # Evaluate potential at t_n and t_{n+1}
        # V_n = V_func(x_interior, t_n)
        # V_nplus = V_func(x_interior, t_nplus)
        #
        # # Average potential
        # V_avg = 0.5*(V_n + V_nplus)
        #
        # # H_avg = -1/2 D + V_avg
        # # Create A and B:
        # # A = I + i(dt/2)*H_avg
        # # B = I - i(dt/2)*H_avg
        # #
        # # Kinetic part contributes:
        # # diag: 1 - 2i alpha and off: i alpha for A when time-independent
        # # but now we incorporate V_avg into the diagonal:
        # #
        # # Actually, to form A and B:
        # # i(dt/2)*H_avg = i(dt/2)*(-1/2 D + V_avg)
        # # = i alpha D + i(dt/2)*V_avg, where alpha = dt/(4 dx^2)
        #
        # # D matrix is the same as before:
        # # For the average Hamiltonian:
        # # diag_A = 1 - 2 i alpha + i(dt/2)*V_avg[j]
        # # diag_B = 1 + 2 i alpha - i(dt/2)*V_avg[j]
        # # off_A = i alpha, off_B = -i alpha
        #
        # dt_half = dt/2
        # diag_A = (1 - 2j*alpha) + 1j*(dt_half)*V_avg
        # diag_B = (1 + 2j*alpha) - 1j*(dt_half)*V_avg
        # off_A = np.full(N-1, 1j*alpha, dtype=complex)
        # off_B = np.full(N-1, -1j*alpha, dtype=complex)
        #
        # # Compute B psi^n
        # # B psi^n = psi^n - i(dt/2)*H_avg psi^n
        # # But we can directly do the operation as done previously:
        # # H_avg psi^n = -1/2 D psi^n + V_avg psi^n
        # Dpsi = laplacian(psi)
        # Hpsi = -0.5*Dpsi + V_avg*psi
        # RHS = psi - 1j*dt_half*Hpsi
        #
        # # Solve A psi^{n+1} = RHS
        # psi = thomas_solve(diag_A, off_A, off_A, RHS)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        #Approach without averages
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        V_n = V_func(x_interior, t_n)
        V_nplus = V_func(x_interior, t_nplus)

        # Construct H_n and H_nplus
        # Dpsi gives the second derivative
        # Dpsi_dummy = laplacian(np.zeros_like(psi))
        Dpsi_dummy = laplacian(np.zeros(psi.shape))

        # H psi = -0.5 * D psi + V(x)*psi
        # We'll just store the diagonal and off-diagonal modifications as before.

        # Form A and B
        diag_A = (1 - 2j * alpha) + 1j * (dt / 2) * V_nplus
        diag_B = (1 + 2j * alpha) - 1j * (dt / 2) * V_n
        # off_A, off_B remain the same for the kinetic terms.
        off_A = np.full(N - 1, 1j * alpha, dtype=complex)
        off_B = np.full(N - 1, -1j * alpha, dtype=complex)

        # Compute RHS = B psi^n
        Dpsi = laplacian(psi)
        Hpsi_n = -0.5 * Dpsi + V_n * psi
        RHS = psi - 1j * (dt / 2) * Hpsi_n

        # Solve A psi^{n+1} = RHS
        psi = thomas_solve(diag_A, off_A, off_A, RHS)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        psi_t[n,:] = psi

    return x, psi_t, times


def V_func(x_points, t):
    """
    Function that acts implements a time dependent potential. The default implementation is of an infinite square well
    :x_points: array of position points
    :t: scalar, represents current time value, intended to be taken from array of all time steps.
    :return: array representing values of the potential as a function of position at the given time step
    """

    barrier_height = 1e20
    L=1
    #
    V = np.full_like(x_points, barrier_height, dtype=float)
    #
    # # Now we use a boolean mask to find the points inside the well
    # Fixed infinite square well, reduces the situation to a time-independent potential case.
    mask = (x_points >= 0) & (x_points <= L)
    # Infinite square well with linearly moving wall
    # mask = (x_points >= 0) & (x_points <= L+1*t)
    # Infinite square well with oscillatory wall
    # mask = (x_points >= 0) & (x_points <= L + 0.05 * np.sin(30 * t))

    V[mask] = 0.0
    #
    # # Various ways to add a pertabation
    # V[mask] += 60000 * np.exp(-((x[mask] - 0.75)**2 / (2 * (0.01**2))))
    # V=3 * np.sin(100*t) * np.exp(-(x_points-0.75)**2/(2*0.02**2))
    # V=3*np.exp(-(x_points - 0.75) ** 2 / (2 * 0.02 ** 2))
    # V=1e9*(.5+0.5*np.tanh(100000*(x_points-(L+1*t))))

    return V

# Parameters:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
L = 1.0+0.25 #domain of position values to be solved over
N = 1000 # number of position steps to use
dt = 0.0001 # Smaller dt for better stability
t_steps = 1000 #total time interval
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dx = L/(N+1)
x_interior = np.linspace(dx, L-dx, N)
t_full=t_steps*dt
t_array=np.linspace(dt,t_full-dt,t_steps)

# Potentials
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
V_array=V_func(x_interior, t_array[0])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def psi0_gaussian_packet(x_points):
    """
    Creates an array that acts as the initial wave function for a gaussian wave packet
    """
    L = 1  # sets domain where psi doesnt vanish, in case whole domain is larger
    x0 = L/2
    sigma = 0.1
    k0 = 100.0
    psi0 = np.exp(-(x_points - x0)**2/(2*sigma**2))*np.exp(-1j*k0*x_points)
    # Normalize
    psi0 = psi0/np.sqrt(np.sum(np.abs(psi0)**2)*dx)
    return psi0

def psi0_inf_square_well_states(x_points):
    """
    Creates an array that acts as the initial wave function for the infinite square well
    """
    L = 1 # sets domain where psi doesnt vanish, in case whole domain is larger
    n = 3  # quantum number
    # psi0 = np.sqrt(2/L) * np.sin(n * np.pi * x_interior / L)
    psi0 = np.zeros_like(x_points, dtype=float)
    mask1 = (x_points > L)
    psi0[mask1]=0
    mask2 = (x_points >= 0) & (x_points <= L)
    psi0[mask2] += np.sqrt(2/L) * np.sin(n * np.pi * x_points[mask2] / L)
    return psi0

# Initial state
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# psi0=psi0_gaussian_packet(x_interior)
psi0=psi0_inf_square_well_states(x_interior)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

starttime=time.time() #specifically timing the solving

# Solve the darn thing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x, psi_t, times = solve_schrodinger_time_dependent(psi0, L, dt, t_steps, V_func)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


print('That took {} seconds to solve'.format(time.time() - starttime)) #specifically timing the solving


# Check normalization over time (should remain ~1).
# If it decreases significantly, make position and times steps smaller
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
norms = [np.sum(np.abs(psi_t[i,:])**2)*dx for i in range(t_steps)]
print("Mean norm:", np.mean(norms), "Std dev norm:", np.std(norms))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# plot the darn thing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots( figsize=(8,6))
line_real, = ax.plot(x[1:-1], psi_t[0,:].real, label='Real part')
line_imag, = ax.plot(x[1:-1], psi_t[0,:].imag, label='Imag part')
line_prob, = ax.plot(x[1:-1], np.abs(psi_t[0,:])**2, label='|psi|^2', color='red')
line_potential, = ax.plot(x[1:-1],V_array[0:], label='Potential', color='black')
# Adds vertical lines if they aren't created by the potential being used.
plt.vlines(x=0, ymin=0, ymax=10, color='black')
# plt.vlines(x=L, ymin=0, ymax=10, color='black')

#controls the range of x and y values that are shown in the plot.
ax.set_xlim(-.25, L+0.25)
ax.set_ylim(-2,3.5)

ax.set_xlabel('x')
ax.set_ylabel('Wavefunction')
title = ax.set_title('')
ax.legend()

def init():
    line_real.set_data([], [])
    line_imag.set_data([], [])
    line_prob.set_data([], [])
    line_potential.set_data([], [])
    return line_real, line_imag, line_prob, line_potential

def animate(i):
    psi = psi_t[i,:]
    V_current = V_func(x_interior, t_array[i])

    line_real.set_data(x[1:-1], psi.real)
    line_imag.set_data(x[1:-1], psi.imag)
    line_prob.set_data(x[1:-1], np.abs(psi)**2)
    line_potential.set_data(x[1:-1], V_current)
    title.set_text('Time = {0:1.3f} s, out of '.format(t_array[i])+f'{t_full} s')
    return line_real, line_imag, line_prob, line_potential

skip_steps=10 # controls the number of time steps to actually be included as frame in the animation. For example, if
                # skip_steps=10, every 10th time step is shown. A smaller value makes the animation slower, larger one
                # faster
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=range(0,t_steps,skip_steps), interval=1)
plt.show()

# Uncomment to save the animation to a specified file
# anim.save('file_path.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~