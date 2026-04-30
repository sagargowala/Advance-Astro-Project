import numpy as np
import matplotlib.pyplot as plt
import core_collapse as cc

t_arr, r_arr, v_arr, p_arr, rho_arr, e_arr, cs_arr = cc.collapse_simulation(lambda: cc.initialise_uniform_density(100), t_end=1.3)
shell_indices = np.arange(0, r_arr.shape[1], 10)

plt.figure(figsize=(9, 12))
plt.subplot(3,2, 1)
for i in shell_indices:
    plt.plot(t_arr, r_arr[:, i], label=f"shell {i}")
plt.xlabel("Time")
plt.ylabel("Radius")
plt.title("(a) r vs t for selected shells")
plt.legend(loc="lower left", fontsize=7)
plt.grid(True, alpha=0.3)

plt.subplot(3, 2, 2)
for i in shell_indices:
    plt.plot(t_arr, v_arr[:, i], label=f"shell {i}")
plt.xlabel("Time")
plt.ylabel("Velocity")
plt.title("(b) v vs t for selected shells")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower left", fontsize=7)


shock_radius = []
shock_velocity = []
t_valid = []

# --- Shock detection + physical filtering ---
for i in range(len(t_arr)):

    dv = np.diff(v_arr[i])
    dr = np.diff(r_arr[i])

    # Avoid division instability
    dv_dr = np.abs(dv / (dr + 1e-12))

    # Shock index (strongest gradient)
    idx = np.argmax(dv_dr)

    # Cell-centered values
    r_shock = 0.5 * (r_arr[i, idx] + r_arr[i, idx+1])
    v_shock = 0.5 * (v_arr[i, idx] + v_arr[i, idx+1])

    # Sound speed is cell-centered, so it uses the same index as dv/dr.
    cs_local = cs_arr[i, idx]

    # Physical condition: supersonic motion can be inward or outward.
    if abs(v_shock) > cs_local:
        shock_radius.append(r_shock)
        shock_velocity.append(v_shock)
        t_valid.append(t_arr[i])

# Convert to arrays
shock_radius = np.array(shock_radius)
shock_velocity = np.array(shock_velocity)
t_valid = np.array(t_valid)

# --- Plot ---
ax1 = plt.subplot(3, 2, 3)

ax1.plot(t_valid, shock_radius, color='tab:blue')
ax1.set_xlabel("Time")
ax1.set_ylabel("Shock Radius", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(t_valid, shock_velocity, color='tab:orange')
ax2.set_ylabel("Shock Velocity", color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.grid(True, alpha=0.3)
plt.title("(c) Shock Evolution ")




#4
# Assuming cs_arr has shape (Nt, N+1) like v_arr
shock_mach = []

for i in range(len(t_arr)):
    # Use the same idx you found for the shock
    dv_dr = np.abs(np.diff(v_arr[i]) / (np.diff(r_arr[i]) + 1e-12))
    idx = np.argmax(dv_dr)
    
    v_s = np.abs(v_arr[i, idx])
    c_s = cs_arr[i, idx] # Sound speed at the shock front
    
    shock_mach.append(v_s / c_s)

# Plotting the Mach Number
plt.subplot(3, 2, 4)
plt.plot(t_arr, shock_mach, color='red')
plt.axhline(1.0, color='black', linestyle='--',label='mach number = 1') # The supersonic threshold
plt.ylabel("Mach Number (M)")
plt.grid(True, alpha=0.3)
plt.xlabel("Time")
plt.title("(d) Mach No. vs t")
plt.legend()





# --- Figure 4: Radial Profiles at Snapshots ---
snap_times = [0.10, 0.30, 0.50, 0.55]
snap_colors = ['#2196F3','#4CAF50','#FF9800','#F44336']

for stime, scol in zip(snap_times, snap_colors):
    idx = np.argmin(np.abs(t_arr - stime))

    r_c = 0.5 * (r_arr[idx, 1:] + r_arr[idx, :-1])
    mask = r_c > 0

    # --- Density (subplot 325) ---
    plt.subplot(325)
    plt.semilogy(r_c[mask], rho_arr[idx][mask],
                 color=scol, lw=1.8, label=f't={stime}')

    # --- Homologous profile (subplot 326) ---
    plt.subplot(326)
    plt.plot(r_c[mask], np.abs(v_arr[idx,1:][mask]) / r_c[mask],
             color=scol, lw=1.8, label=f't={stime}')


# --- Formatting ---
plt.subplot(325)
plt.xlabel('Radius  r', fontsize=11)
plt.ylabel(r'Density $\rho$ (log)', fontsize=11)
plt.title('(e)Density profile', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.subplot(326)
plt.xlabel('Radius  r', fontsize=11)
plt.ylabel(r'$|v|/r$', fontsize=11)
plt.title('(f)Homologous collapse: $|v|/r$ vs $r$', fontsize=12)
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig("plot.png", dpi=500, bbox_inches='tight')


plt.show()
