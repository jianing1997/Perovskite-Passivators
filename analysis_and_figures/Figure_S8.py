from _shared import plot_property_evolution, validate_png


def main() -> None:
    path = plot_property_evolution("Figure S8.png")
    validate_png(path, 1050, 1850)


if __name__ == "__main__":
    main()
