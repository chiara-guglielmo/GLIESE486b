import argparse
import yaml
from .detection.transit import transit
from .parameters.parameters import Parameters

def main():
    parser = argparse.ArgumentParser(description="Daneel: a Python package to model and analyse exoplanet transits")

    parser.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    parser.add_argument("-i", "--input", type=str, default=None, help="Path to the YAML configuration file")

    parser.add_argument("-o", "--output", action="store_true", help="Save the transit plot in the current working directory")

    args = parser.parse_args()

    if args.transit:
        transit_params = None
        if args.input is not None:
            parameters = Parameters(args.input)
            transit_params = parameters.get("transit")
            if not isinstance(transit_params, dict) or not transit_params:
                parser.error("The YAML file must contain a non-empty 'transit' dictionary.")
        
        transit(transit_params, save=args.output)

if __name__ == "__main__":
    main()



'''
from datetime import datetime, timezone

from daneel.detection import TransitModel
from daneel.parameters import Parameters


def main():
    parser = argparse.ArgumentParser(description="Model and analyse exoplanet transits.")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    actions.add_argument("-d", "--detect", action="store_true", help="Run exoplanet detection")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    parser.add_argument("-o", "--output", default="lc.png", help="Output image for --transit")

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.now(timezone.utc)

    input_pars = Parameters(args.input_file).params

    if args.transit:
        transit = TransitModel(input_pars['transit'])
        transit.plot_light_curve(args.output)
    elif args.detect:
        parser.error("detection is not implemented yet; use --transit")
    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    finish = datetime.now(timezone.utc)
    print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")
'''

if __name__ == "__main__":
    main()
