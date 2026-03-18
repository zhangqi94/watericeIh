"""Color palettes used in analysis scripts (RGB only)."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Okabe-Ito 7-color set (restored, excluding black).
PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "okabe_ito_7": {
        "Orange": (230, 159, 0),         # #E69F00
        "Sky blue": (86, 180, 233),      # #56B4E9
        "Bluish green": (0, 158, 115),   # #009E73
        "Yellow": (240, 228, 66),        # #F0E442
        "Blue": (0, 114, 178),           # #0072B2
        "Vermillion": (213, 94, 0),      # #D55E00
        "Reddish purple": (204, 121, 167),  # #CC79A7
    },
    
    # "vermillion_blue": {
    #     "Vermillion": (213, 94, 0),      # #D55E00
    #     "Blue": (0, 114, 178),           # #0072B2
    # },
    # "blue_vermillion_orange": {
    #     "Blue": (0, 114, 178),           # #0072B2
    #     "Vermillion": (213, 94, 0),      # #D55E00
    #     "Bluish green": (0, 158, 115),   # #009E73
    # },
    # "blue_gradient_4_small_to_large": {
    #     "small": (190, 198, 208),        # #BEC6D0 (light gray with slight blue tint, darker)
    #     "medium_small": (136, 174, 199), # #88AEC7
    #     "medium_large": (68, 137, 178),  # #4489B2
    #     "large": (0, 100, 157),          # #00649D
    # },
    
    
    "two_colors": {
        "red": (238, 32, 36),
        "blue": (67, 120, 188),
    },

    "three_colors": {
        "red": (238, 32, 36),
        "blue": (67, 120, 188),
        "green": (28, 153, 135),
    },

    "blue_gradient_4_small_to_large": {
        "c1": (153, 200, 224),
        "c2": (77, 154, 199),
        "c3": (39, 111, 175),
        "c4": (27, 59, 112),
    },

    "traj_panel_d_blues": {
        "deep_blue": (27, 59, 112),
        "gray_blue": (118, 144, 162),
    },



    "img_palette_01": {
        "c1": (213, 105, 93),
        "c2": (245, 176, 65),
        "c3": (246, 218, 101),
        "c4": (82, 190, 128),
        "c5": (145, 223, 208),
        "c6": (93, 173, 226),
        "c7": (164, 105, 189),
        "c8": (138, 112, 103),
        "c9": (255, 188, 167),
        "c10": (72, 79, 152),
        "c11": (255, 255, 133),
    },

    "img_palette_02": {
        "c1": (27, 59, 112),
        "c2": (39, 111, 175),
        "c3": (77, 154, 199),
        "c4": (153, 200, 224),
        "c5": (212, 230, 239),
        "c6": (248, 244, 242),
        "c7": (251, 216, 195),
        "c8": (242, 164, 129),
        "c9": (214, 96, 77),
        "c10": (181, 32, 46),
        "c11": (112, 12, 34),
    },

    "img_palette_03": {
        "c1": (152, 216, 59),
        "c2": (96, 201, 96),
        "c3": (48, 180, 123),
        "c4": (28, 153, 135),
        "c5": (37, 129, 142),
        "c6": (47, 105, 142),
        "c7": (61, 77, 137),
        "c8": (70, 46, 122),
    },

    "img_palette_04": {
        "c1": (42, 47, 128),
        "c2": (57, 83, 165),
        "c3": (67, 120, 188),
        "c4": (111, 204, 222),
        "c5": (153, 203, 111),
        "c6": (246, 235, 20),
        "c7": (246, 127, 33),
        "c8": (238, 32, 36),
        "c9": (125, 20, 21),
    },

    "img_palette_05": {
        "c1": (0, 99, 53),
        "c2": (0, 157, 83),
        "c3": (119, 181, 106),
        "c4": (179, 206, 126),
        "c5": (241, 241, 183),
        "c6": (242, 203, 123),
        "c7": (232, 136, 84),
        "c8": (227, 58, 40),
        "c9": (167, 25, 39),
    },

    "img_palette_06_fire_to_light": {
        "c1": (89, 1, 0),       # #590100
        "c2": (112, 0, 0),      # #700000
        "c3": (160, 0, 0),      # #A00000
        "c4": (192, 0, 0),      # #C00000
        "c5": (233, 0, 0),      # #E90000
        "c6": (255, 20, 0),     # #FF1400
        "c7": (255, 56, 1),     # #FF3801
        "c8": (255, 114, 2),    # #FF7202
        "c9": (255, 144, 0),    # #FF9000
        "c10": (255, 173, 1),   # #FFAD01
        "c11": (255, 231, 1),   # #FFE701
        "c12": (255, 255, 11),  # #FFFF0B
        "c13": (254, 254, 57),  # #FEFE39
        "c14": (255, 255, 144), # #FFFF90
        # "c15": (239, 239, 190), # #EFEFBE
        # "c16": (255, 255, 244), # #FFFFF4
    },
}





def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """(R, G, B) -> '#RRGGBB'."""
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def _rows_for_plot(
    colors: dict,
) -> list[tuple[str, list[tuple[str, tuple[int, int, int]]]]]:
    """Normalize both structures into rows for plotting."""
    rows: list[tuple[str, list[tuple[str, tuple[int, int, int]]]]] = []
    if not colors:
        return rows

    first_value = next(iter(colors.values()))
    if not isinstance(first_value, dict):
        raise TypeError("colors must be a dictionary of dictionaries")

    first_inner_value = next(iter(first_value.values()))
    if isinstance(first_inner_value, tuple):
        for palette_name, color_map in colors.items():
            entries = [(name, rgb) for name, rgb in color_map.items()]
            rows.append((palette_name, entries))
        return rows

    if isinstance(first_inner_value, list):
        for group_name, palette_map in colors.items():
            for palette_name, rgb_list in palette_map.items():
                entries = [(f"step-{i+1}", rgb) for i, rgb in enumerate(rgb_list)]
                rows.append((f"{group_name}/{palette_name}", entries))
        return rows

    raise TypeError("Unsupported color structure in 'colors'.")


def plot_color_palettes(colors: dict = PALETTES, show: bool = True):
    """Plot palettes and annotate row names in []."""
    rows = _rows_for_plot(colors)
    if not rows:
        raise ValueError("No palettes found.")

    max_colors = max(len(entries) for _, entries in rows)
    fig_width = max(10.0, max_colors * 1.25 + 2.8)
    fig_height = max(2.8, len(rows) * 1.7 + 0.6)

    fig, axes = plt.subplots(len(rows), 1, figsize=(fig_width, fig_height))
    if len(rows) == 1:
        axes = [axes]

    for ax, (row_name, entries) in zip(axes, rows):
        for i, (color_name, rgb) in enumerate(entries):
            rgb_norm = tuple(channel / 255 for channel in rgb)
            ax.add_patch(
                Rectangle(
                    (i, 0.25),
                    1,
                    1,
                    facecolor=rgb_norm,
                    edgecolor="white",
                    linewidth=1.2,
                )
            )
            ax.text(
                i + 0.5,
                0.15,
                f"[{color_name}]",
                ha="center",
                va="top",
                fontsize=9,
                rotation=28,
            )

        ax.set_xlim(-1.9, max_colors)
        ax.set_ylim(-0.58, 1.35)
        ax.axis("off")
        ax.text(-0.1, 0.75, f"[{row_name}]", ha="right", va="center", fontsize=11)

    fig.suptitle("Color Palettes (RGB)", fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    print("[PALETTES]")
    for palette_name, color_map in PALETTES.items():
        print(f"[{palette_name}]")
        for color_name, rgb in color_map.items():
            print(f"[{color_name}]: {rgb}")
        print()

    plot_color_palettes(PALETTES)
