# Numerically-Solving-Time-Dependent-Schrodinger
A python implementation of methods to solve the Schrodinger equation, with the particular goal of handling cases of time-dependent potentials

## Overview
This Python script simulates the time evolution of a wavefunction using the time-dependent Schrödinger equation. It generates animations showing how the wavefunction changes over time under various potentials. The script is highly customizable, allowing users to:

- Change the initial wavefunction.
- Modify the potential function.
- Adjust simulation parameters (e.g., time steps, spatial resolution).
- Save the animation as a file.

No advanced coding knowledge is needed to use or modify the script. Follow the instructions below to get started.

---

## How to Run

1. **Ensure You Have the Required Libraries**:
   - Install Python (preferably version 3.9 or later).
   - Install the following Python libraries using pip:
     ```bash
     pip install numpy matplotlib numba
     ```

2. **Run the Script**:
   - Save the script to a file, e.g., `Solve_Schrodinger_Time_Dependent_Full.py`.
   - Run it using the command:
     ```bash
     python wave_simulation.py
     ```
   - An animation window will open, displaying the wavefunction evolution.

---

## Key Sections to Modify

### 1. **Simulation Parameters**

You can change the following variables to control the simulation:

```python
L = 1.0 + 0.25  # Domain length (default: 1.25)
N = 1000        # Number of spatial steps (default: 1000)
dt = 0.0001     # Time step size (default: 0.0001)
t_steps = 1000  # Number of time steps (default: 1000)
```

- **L**: Length of the spatial domain. Increase this to simulate a larger region.
- **N**: Number of spatial points. Higher values increase resolution but slow down the simulation.
- **dt**: Time step size. Smaller values increase accuracy but require more computational time.
- **t_steps**: Total number of time steps. Higher values simulate a longer time period.

---

In order for the numerical approach to be stable (i.e. physically realistic) small enough values of position and time steps (i.e. small `dt` and large `N`) must be used. One way to check if they are small enough is to check if probability is preserved. The following lines compute the probabilty over all the time steps and find the average and standard deviation. It should stay close to 1, if it deacreases signifigantly, smaller values steps are needed. 

```
norms = [np.sum(np.abs(psi_t[i,:])**2)*dx for i in range(t_steps)]
print("Mean norm:", np.mean(norms), "Std dev norm:", np.std(norms))
```

### 2. **Initial Wavefunction**

The initial wavefunction defines the starting state of the system. Uncomment one of the following options:

- **Gaussian Wave Packet** (smooth initial wave):
  ```python
  psi0 = psi0_gaussian_packet(x_interior)
  ```

- **Infinite Square Well State** (quantum energy levels):
  ```python
  psi0 = psi0_inf_square_well_states(x_interior)
  ```

### 3. **Potential Function**

The potential function determines how the wavefunction evolves over time. The script includes the following potential as a function:

- **Infinite Square Well** (with optional moving walls):
  ```python
  def V_func(x_points, t):
      mask = (x_points >= 0) & (x_points <= L + 0.05 * np.sin(30 * t))
      V = np.full_like(x_points, 1e20)  # Barrier height
      V[mask] = 0.0
      return V
  ```

To modify the potential, edit the `V_func` function and its definition inside the script. Several examples are included as lines that can be commented/uncommented. 

### 4. **Animation**

The animation will show the potential, the real and imaginary parts of the wave function, and the probability. You can adjust the appearance and speed of the animation:

- **Skip Steps**: Controls how many time steps to skip between frames (default: `10`):
  ```python
  skip_steps = 10
  ```
  A smaller value makes the animation slower and smoother; a larger value makes it faster.

- **Axis Ranges**: Modify the plot limits to focus on specific areas:
  ```python
  ax.set_xlim(-0.25, L + 0.25)
  ax.set_ylim(-2, 3.5)
  ```

### 5. **Saving the Animation**

Uncomment the following line to save the animation as an MP4 file:

```python
anim.save('wave_animation.mp4', fps=60, extra_args=['-vcodec', 'libx264'])
```

- Replace `'wave_animation.mp4'` with your desired file path.
- Adjust `fps` (frames per second) to control playback speed.

---

## Troubleshooting

- **Wavefunction Norm Not Stable**:
  - If the printed norm values deviate significantly from `1`, reduce `dt` and/or increase `N`.

- **Animation Too Slow**:
  - Increase `skip_steps`.

- **No Animation Display**:
  - Ensure your Python environment supports GUI windows.

---

## Additional Notes

- To explore different quantum systems, modify the initial wavefunction (`psi0`) and potential function (`V_func`).
- The script is intended to be flexible so make various modifications to examine different scenarious. 

