import numpy as np
import matplotlib.pyplot as plt
import py4DSTEM

def show_results(ptycho, power_fft : float = 0.5):
    if ptycho.object_cropped.ndim > 2:
        object_cropped = ptycho.object_cropped.mean(0)
    else:
        object_cropped = ptycho.object_cropped
    # Plots to show object phase in real and Fourier space
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    py4DSTEM.show(
        np.angle(object_cropped),
        title='Object phase (real space)',
        intensity_range='minmax',
        cmap='inferno',
        ticks=False,
        scalebar=True,
        pixelsize=ptycho.sampling[0],
        pixelunits=r'$Å$',
        show_cbar=True,
        figax=(fig, axs[0])
    )
    py4DSTEM.show(
        np.angle(object_cropped),
        title='Object phase (Fourier space)',
        intensity_range='minmax',
        cmap='inferno',
        ticks=False,
        scalebar=True,
        pixelsize=ptycho._reciprocal_sampling[0],
        pixelunits=r'$Å^{-1}$',
        show_fft=True,
        power=power_fft,
        figax=(fig, axs[1])
    )
    axs[1].set_aspect(object_cropped.shape[1]/object_cropped.shape[0])

    # Plots to show probes in real and Fourier space
    ptycho.show_probe(chroma_boost=2)
    ptycho.show_fourier_probe(chroma_boost=2)

def show_probe_positions(ptycho, initial_scan_positions, skip : int =5):
    fig, ax = plt.subplots(1,1, figsize=(4,6))
    ax.scatter(initial_scan_positions[:,1][::skip], initial_scan_positions[:,0][::skip], marker='s', s=2, label="Initial scan positions")
    ax.scatter(ptycho.positions[:,1][::skip], ptycho.positions[:,0][::skip], marker='s', s=2, label="Refined scan positions")
    ax.invert_yaxis()
    ax.set_aspect(1)
    ax.legend()

def plot_ptycho_error(ptycho, ylim : list = None, log_scale : bool =False):
    # Show error as a function of the number of iterations
    if ptycho.error > np.min(ptycho.error_iterations):
        print(f"Last iteration is not the minimum error! last = {ptycho.error:.3e} VS min = {np.min(ptycho.error_iterations):.3e}")

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.set_title('Ptychographic reconstruction error')
    ax.plot(ptycho.error_iterations)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('NMSE')
    ax.set_xlim([0, len(ptycho.error_iterations)])
    ax.grid()
    ax.grid(which="minor", color="0.9")

    if ylim is not None:
        ax.set_ylim(ylim)
        y_text = (ptycho.error + ylim[0]) / 2
    else:
        y_text = ptycho.error*0.95
        
    ax.hlines(y=ptycho.error, xmin=0, xmax=len(ptycho.error_iterations), linestyles='--', color='k', linewidth=0.75)
    ax.text(x=len(ptycho.error_iterations)*0.65, y=y_text, s="Last error = {:.2e}".format(ptycho.error))

    if log_scale:
        ax.set_yscale('log')
