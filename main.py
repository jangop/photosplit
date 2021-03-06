import os
from pathlib import Path
import sklearn.neighbors
import numpy as np
import skimage.io
import skimage.transform
import skimage.feature
import skimage.color
import skimage.morphology
import skimage.segmentation
import skimage.filters
import skimage.measure

from geometry import Point
from plotting import plot_polygons, plot_points

from loguru import logger

IMAGE_SIZE_THRESHOLD = 500


def process_image_file(
    image_path: Path, out_path: Path, debug_path: Path = None, **kwargs
):
    logger.info(f"Processing {image_path}")
    name = image_path.stem
    ext = image_path.suffix
    image = load_image(image_path)
    if debug_path:
        debug_path = debug_path / name
        debug_path.mkdir(parents=True, exist_ok=True)
    photos = extract_photos_from_image(image, debug_dir=debug_path, **kwargs)
    n_photos = len(photos)
    logger.info(f"Extracted {n_photos} photos")
    pad = max(2, n_photos // 10)
    out_path.mkdir(exist_ok=True)
    for i_photo, photo in enumerate(photos):
        save_image(photo, out_path / f"{name}-{i_photo:0{pad}d}{ext}")


def load_image(path: Path) -> np.ndarray:
    return skimage.io.imread(str(path))


def save_image(image: np.ndarray, path: Path):
    skimage.io.imsave(str(path), skimage.img_as_ubyte(image))


def extract_photos_from_image(image, debug_dir: Path = None) -> list[np.ndarray]:
    # Determine size of original.
    original = image
    original_height, original_width = original.shape[:2]
    logger.info(
        f"Original image size: {original_height} × {original_width} (height × width)"
    )

    # Prepare debug directory.
    debug_step = 0

    # Convert to grayscale.
    image = skimage.color.rgb2gray(original)

    # Crop away outer border.
    crop_factor = 0.01
    crop_vertical, crop_horizontal = (
        crop_factor * np.array(image.shape[:2], dtype=float)
    ).astype(int)
    logger.info(
        f"Cropping {crop_vertical} pixels top and bottom, and {crop_horizontal} pixels left and right"
    )
    image = skimage.util.crop(
        image, ((crop_vertical, crop_vertical), (crop_horizontal, crop_horizontal))
    )

    # If image is too large, resize it.
    height, width = image.shape[:2]
    bigger = max(height, width)
    if bigger > IMAGE_SIZE_THRESHOLD:
        factor = IMAGE_SIZE_THRESHOLD / bigger
        # Resize image accordingly.
        logger.info(f"Resizing image by factor {factor}")
        image = skimage.transform.rescale(image, factor, anti_aliasing=True)
    else:
        factor = 1.0

    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-resized.png"
        skimage.io.imsave(str(path), skimage.img_as_ubyte(image))
        debug_step += 1

    # Detect edges.
    edges = skimage.filters.sobel(image)
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-edges.png"
        skimage.io.imsave(path, skimage.img_as_ubyte(edges))
        debug_step += 1

    # Find markers.
    markers = np.zeros_like(edges)
    pad = 2
    markers[:pad, :] = 1
    markers[:, :pad] = 1
    markers[-pad:, :] = 1
    markers[:, -pad:] = 1
    threshold = 0.8
    markers[image < threshold] = 2
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-markers.png"
        skimage.io.imsave(
            path, skimage.img_as_ubyte(skimage.color.label2rgb(markers, bg_label=0))
        )
        debug_step += 1

    # Perform watershed.
    watershed = skimage.segmentation.watershed(edges, markers)
    watershed[watershed == 1] = 0
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-watershed.png"
        skimage.io.imsave(
            path, skimage.img_as_ubyte(skimage.color.label2rgb(watershed, bg_label=0))
        )
        debug_step += 1

    # Label segments.
    labels = skimage.measure.label(watershed)
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-labels.png"
        skimage.io.imsave(
            path,
            skimage.img_as_ubyte(
                skimage.color.label2rgb(labels, image=image, bg_label=0)
            ),
        )
        debug_step += 1

    # Find corners.
    watershed[watershed == 2] = 255
    corners = skimage.feature.corner_peaks(
        skimage.feature.corner_harris(watershed, k=0.001, eps=1e-6, sigma=5), min_distance=5, threshold_rel=0.5
    )
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-corners.png"
        points = [Point(y, x) for x, y in corners]
        plot_points(watershed, points, path)
        debug_step += 1

    sub_pixel_corners = skimage.feature.corner_subpix(watershed, corners)
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-sub_pixel_corners.png"
        points = [Point(y, x) for x, y in sub_pixel_corners]
        plot_points(watershed, points, path)
        debug_step += 1

    # if debug_dir:
    #     ks = [0.001, 0.01, 0.1]
    #     epsilons = [1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01]
    #     sigmas = [0.01, 0.1, 1, 2, 3, 4, 5]
    #     min_dists = [5, 10, 15, 20]
    #     rel_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    #     for k in ks:
    #         for epsilon in epsilons:
    #             for sigma in sigmas:
    #                 for min_dist in min_dists:
    #                     for rel_threshold in rel_thresholds:
    #                         corners = skimage.feature.corner_peaks(
    #                             skimage.feature.corner_harris(
    #                                 watershed, k=k, eps=epsilon, sigma=sigma
    #                             ), min_distance=min_dist, threshold_rel=rel_threshold
    #                         )
    #                         dir = debug_dir / "corners"
    #                         dir.mkdir(exist_ok=True)
    #                         path = dir / f"k-{k}_epsilon-{epsilon}_sigma-{sigma}_min_dist-{min_dist}_rel_threshold-{rel_threshold}.png"
    #                         points = [Point(y, x) for x, y in corners]
    #                         plot_points(watershed, points, path)



    # Find contours.
    contours = skimage.measure.find_contours(watershed, 0.5)
    contours = [contour[:, [1, 0]] for contour in contours]
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-contours.png"
        plot_polygons(image, contours, path, plot_text=False)
        debug_step += 1

    # Simplify contours.
    approximations = [
        skimage.measure.approximate_polygon(contour, tolerance=10)
        for contour in contours
    ]
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-approximations.png"
        plot_polygons(image, approximations, path, plot_text=True)
        debug_step += 1

    # Correct polygons.
    neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    neighbors.fit(sub_pixel_corners)
    def correct_polygon(polygon, neighbors):
        distances, indices = neighbors.kneighbors(polygon)
        coordinates = sub_pixel_corners[indices[:, 0]]
        for i, (point, coordinate) in enumerate(zip(polygon, coordinates)):
            logger.critical(f"{i}: {point} -> {coordinate}")
        return coordinates
    corrected_approximations = [
        correct_polygon(approximation, neighbors)
        for approximation in approximations
    ]
    if debug_dir:
        path = debug_dir / f"{debug_step:02d}-corrected_approximations.png"
        plot_polygons(image, corrected_approximations, path, plot_text=True)
        debug_step += 1




    # Extract polygons.
    photos = []
    for i, approximation in enumerate(corrected_approximations):
        # Extract rectangles.
        if len(approximation) == 5:
            # Leave out the closing point and scale up.
            source_coordinates = approximation / factor + np.array(
                [crop_vertical, crop_horizontal]
            )

            # Determine size.
            target_width = int(
                np.linalg.norm(source_coordinates[1] - source_coordinates[0])
            )
            target_height = int(
                np.linalg.norm(source_coordinates[3] - source_coordinates[0])
            )

            # Prepare target coordinates.
            target_top_left = [0, 0]
            target_top_right = [target_width, 0]
            target_bottom_right = [target_width, target_height]
            target_bottom_left = [0, target_height]
            target_coordinates = np.array(
                [
                    target_top_left,
                    target_top_right,
                    target_bottom_right,
                    target_bottom_left,
                ]
            )

            # Estimate transformation.
            transform = skimage.transform.ProjectiveTransform()
            transform.estimate(source_coordinates[:-1], target_coordinates)

            # Apply transformation.
            warped = skimage.transform.warp(
                original, transform.inverse, output_shape=(target_height, target_width)
            )
            if debug_dir:
                path = debug_dir / f"{debug_step:02d}-{i:02d}-points.png"
                plot_polygons(original, [source_coordinates], path, plot_text=True)
                debug_step += 1
                path = debug_dir / f"{debug_step:02d}-{i:02d}-warped.png"
                plot_polygons(warped, [target_coordinates], path, plot_text=True)
                debug_step += 1

            # Store result.
            photos.append(warped)
    return photos


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract photos from a scanned document."
    )
    parser.add_argument("input", type=str, nargs="+", help="Input file(s).")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument("--debug", type=str, help="Debug directory.", default=None)

    args = parser.parse_args()
    debug_path = Path(args.debug) if args.debug else None
    for input_file in args.input:
        process_image_file(
            image_path=Path(input_file),
            out_path=Path(args.output),
            debug_path=debug_path,
        )


if __name__ == "__main__":
    cli()
