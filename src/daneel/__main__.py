import argparse
import yaml
from .detection.dataset import DataSetCreation
from .detection.transit import transit
from .detection.classifier_RandomForest import RandomForest_classifier
from .detection.classifier_CNN import CNN_classifier
from .parameters.parameters import Parameters
from pathlib import Path
from .detection.GAN import GAN

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path):

    if path is None:
        return None

    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return str(path.resolve())


def get_config_path(input_files, default_file):

    if input_files is None:
        config_path = PROJECT_ROOT / "examples" / default_file
    else:
        config_path = Path(input_files[0]).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    return config_path

def main():
    parser = argparse.ArgumentParser(
        description="Daneel: a Python package to model and analyse exoplanet transits"
    )

    parser.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    parser.add_argument("-i", "--input", nargs="+", type=str, default=None, help="Paths to the YAML configuration files")

    parser.add_argument("-d", "--detection", choices=["rf", "cnn"], default=None, help="Run exoplanet detection through Random Forest classifier ('rf') or CNN ('cnn')")

    parser.add_argument("-o", "--output", action="store_true", help="Save the output in the current working directory")

    parser.add_argument("-p", "--predict", type=str, default=None, metavar="CSV", help="Classify light curves from a CSV file using a pretrained CNN")

    parser.add_argument("-dr", "--dream", action="store_true", help="Train a GAN and generate synthetic light curves.")

    args = parser.parse_args()

    # Handle transit analysis
    if args.transit:
        transit_params_list = []
        transit_params = None

        if args.input is not None:

            # Loop through the files to extract their dictionaries into a list
            for yaml_file in args.input:
                parameters = Parameters(yaml_file)
                transit_params = parameters.get("transit")

                if not isinstance(transit_params, dict) or not transit_params:
                    parser.error("The YAML file must contain a non-empty 'transit' dictionary.")

                transit_params_list.append(transit_params)

        else:
            transit_params_list = [None]

        transit(transit_params_list, save=args.output)

        # Handle detection through Random Forest
    if args.detection == "rf":

        yaml_file = get_config_path(
            args.input,
            "RF_parameters.yaml"
        )

        parameters = Parameters(str(yaml_file))

        config = {
            "dataset_path": resolve_path(
                parameters.get("dataset_path")
            ),
            "RF_params": parameters.get("RF_params"),
            "model_prefix": parameters.get(
                "model_prefix", "tess_rf_model"
            ),
            "save_path": (
                args.output if args.output
                else parameters.get("save_path", None)
            )
        }

        RandomForest_classifier(config).run_all_configurations()

    # Handle detection through CNN
    if args.detection == "cnn":

        yaml_file = get_config_path(
            args.input,
            "CNN_parameters.yaml"
        )

        parameters = Parameters(str(yaml_file))

        config = {
            "dataset_path": resolve_path(
                parameters.get("dataset_path")
            ),
            "device": parameters.get("device", None),
            "CNN_params": parameters.get("CNN_params"),
        }

        CNN_classifier(config).run_all_configurations()

    # Handle prediction through pretrained CNN
    if args.predict is not None:

        cnn_classifier = CNN_classifier(config={})

        results = cnn_classifier.predict(
            csv_path=args.predict,
            save_path="predictions.csv" if args.output else None
        )

        n_planets = (results["prediction"] == 1).sum()
        n_total = len(results)

        print(f"Detected planets: {n_planets}/{n_total}")
        print(f"Percentage: {100 * n_planets / n_total:.2f}%")

    # Handle GAN training and light curve generation
    if args.dream:

        yaml_file = get_config_path(
            args.input,
            "GAN_parameters.yaml"
        )

        with open(yaml_file, "r") as file:
            config = yaml.safe_load(file) or {}

        config["dataset_path"] = resolve_path(
            config.get("dataset_path")
        )

        GAN(config=config).run(save=args.output)


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

if __name__ == "__main__":
    main()
'''