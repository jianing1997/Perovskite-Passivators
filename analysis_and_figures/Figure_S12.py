from _shared import plot_pareto_grid, validate_png


def main() -> None:
    path = plot_pareto_grid("Figure S12.png")
    validate_png(path, 1150, 1150)


if __name__ == "__main__":
    main()
