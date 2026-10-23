# Imports and configuration
from pathlib import Path
import re
import batman  # from the pip/conda distribution named "batman-package"
import matplotlib.pyplot as plt
import numpy as np

# Reusable calculation for exoplanet lightcurves
def transit(params_dict=None):
    """
    A function allowing to compute and plot the transit lightcurve of exoplanets using the batman package.
    Keyword argouments:
    params_dict -- a python dictionary containing the name of the planet and all the important parameters.
    Return: a .png image with the transit lightcurve corresponding to the parameters given as input.
    """

    if params_dict is None:
        params_dict = {}

    # Initialize parameters
    # The code tries to import some values of the parameter from the dictionary provided as input,
    # if they are missing it uses as default values the ones associated to the exoplanet GJ 486 b.   
    params = batman.TransitParams()
    PLANET_NAME = params_dict.get('name', 'GJ 486 b')  # Name of the exoplanet
    params.t0 = params_dict.get('t0', 0.)  # Time of central transit (days)
    params.per = params_dict.get('per', 1.46712127)  # Orbital period (days
    params.rp = params_dict.get('rp', 0.037244)  # Planet radius (in units of stellar radius)
    params.a = params_dict.get('a', 11.380)  # Semi-major axis (in units of stellar radius)
    params.inc = params_dict.get('inc', 89.39)  # Orbital inclination (degrees)
    params.ecc = params_dict.get('ecc', 0.000860)  # Eccentricity
    params.w = params_dict.get('w', 14.0)  # Argument of periastron (degrees)
    params.limb_dark = params_dict.get('limb_dark', 'quadratic') # Limb darkening model
    params.u = params_dict.get('u', [0.0606, 0.1724])  # Limb darkening coefficients

    # Select observation times
    t = np.linspace(-0.047, 0.047, 1000)

    # Compute the model lightcurve
    model = batman.TransitModel(params, t)  # Initialise the transit model
    flux = model.light_curve(params)  # Calculate the normalised flux

    # Make the plot and same the output
    planet_slug = re.sub(r"[^A-Za-z0-9]+", "_", PLANET_NAME).strip("_")
    output_file = Path(f"{planet_slug}_assignment1_taskF.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(t, flux, lw=2, label="BATMAN transit model")
    ax.axhline(1.0, color="0.6", lw=1, ls="--", label="Out-of-transit flux")
    ax.set(xlabel="Time from transit centre [days]", ylabel="Relative stellar flux",
        title=f"Model transit light curve — {PLANET_NAME}")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.show()

# Call the function above
transit()