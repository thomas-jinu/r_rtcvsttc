from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import sys

# Use the relative-time module beside the notebook's Gaussian-transform script.
sys.path.insert(0, str(script.resolve().parent))
from fourier_relative_time import transform_inputs

requested_times = np.array([10.0, 15.0, 20.0, 30.0, 40.0, 50.0])
target_probe_time = 15.0
offset_fraction = 0.22
background = '#fbfaf7'

# `times` and `z` come from the preceding transform/plot cell.
time_index = np.unique([
    np.abs(times - requested).argmin() for requested in requested_times
])
highlight = int(np.abs(times - target_probe_time).argmin())
n = len(time_index)

trace_colors = plt.get_cmap('plasma')(np.linspace(0, 1, n + 2))[:n]

in_range = (plot_frequencies >= display_min) & (plot_frequencies <= display_max)
x = plot_frequencies[in_range]
dos = z[in_range, :].real

# Calculate the unwindowed relative-time transform for the same inputs,
# selected probe times, and unshifted frequencies as the Gaussian transform.
relative = transform_inputs(inputs, times[time_index], frequencies, input_convention)
relative_dos = np.array([
    [relative[(*site_pair, float(times[j]), float(w))].real for j in time_index]
    for w in frequencies[in_range]
])

fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True, facecolor=background)
ax.set_facecolor(background)

peak = max(np.max(np.abs(dos[:, time_index])), np.max(np.abs(relative_dos)))
step = offset_fraction * peak
floor = -1.1 * peak

for k in reversed(range(n)):
    j = time_index[k]
    base = k * step
    y = dos[:, j] + base
    is_highlight = j == highlight
    z_order = 2 * (n - k)

    ax.fill_between(x, floor + base, y, color=background, lw=0, zorder=z_order)
    ax.plot(
        x, y, color=trace_colors[k],
        lw=2.4 if is_highlight else 1.2,
        zorder=z_order + 1, solid_capstyle='round'
    )
    ax.plot(
        x, relative_dos[:, k] + base, color=trace_colors[k], linestyle='--',
        lw=2.4 if is_highlight else 1.2, zorder=z_order + 2
    )

ax.set_yticks(np.arange(n) * step)
ax.set_yticklabels([f'{times[j]:g}' for j in time_index])
ax.set_ylabel('Probe time')
ax.set_ylim(floor, (n - 1) * step + 1.1 * peak)
ax.set_xlim(display_min, display_max)
ax.set_title(f'DOS, sites {site_pair}', fontsize=13, pad=10)
ax.legend(handles=[
    Line2D([0], [0], color='0.25', label='Gaussian probe'),
    Line2D([0], [0], color='0.25', linestyle='--', label='Relative time'),
])
ax.set_xlabel(f'Frequency ω − ({frequency_shift:g})')
ax.tick_params(length=3, labelsize=10)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('0.3')
    spine.set_linewidth(0.9)

discrete = BoundaryNorm(np.arange(n + 1) - 0.5, n)
cbar = fig.colorbar(
    ScalarMappable(norm=discrete, cmap=ListedColormap(trace_colors)),
    ax=ax, shrink=0.85, pad=0.02, ticks=np.arange(n)
)
cbar.set_ticklabels([f'{times[j]:g}' for j in time_index])
cbar.set_label('Probe time')
cbar.outline.set_visible(False)

plt.show()
