# Imports and configuration
from pathlib import Path
import re
import batman  # from the pip/conda distribution named "batman-package"
import matplotlib.pyplot as plt
import numpy as np

# Reusable calculation for exoplanet lightcurves
def transit(parameter_dicts=None, save=False):
    """
    A function allowing to compute and plot the transit lightcurve of exoplanets using the batman package.
    Keyword argouments:
    params_dict -- a list of python dictionaries containing the name of each planet and all the important parameters.
    Return: a .png image with all superimposed transit lightcurve corresponding to the parameters given as input.
    """

    # Prepare the plot
    t = np.linspace(-0.03, 0.03, 1000)
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    planet_slugs = []

    for params_dict in parameter_dicts:

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

        # Compute the model lightcurve
        model = batman.TransitModel(params, t)  # Initialise the transit model
        flux = model.light_curve(params)  # Calculate the normalised flux

        # Make the plot and same the output
        planet_slug = re.sub(r"[^A-Za-z0-9]+", "_", PLANET_NAME).strip("_")
        planet_slugs.append(planet_slug)
        ax.plot(t, flux, lw=2, label=f"BATMAN transit model, {planet_slug}")

    # Refine the plot
    combined_slug = ", ".join(planet_slugs)
    ax.axhline(1.0, color="0.6", lw=1, ls="--", label="Out-of-transit flux")
    ax.set(xlabel="Time from transit centre [days]", ylabel="Relative stellar flux",
        title=f"Model transit light curve — {combined_slug}")
    ax.grid(alpha=0.25)
    #ax.legend(loc="best")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    fig.tight_layout()
    if save:
        output_file = Path.cwd() / f"{combined_slug}_transit.png"
        fig.savefig(output_file, dpi=180, bbox_inches="tight")
        print(f"Transit plot saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    transit()