from __future__ import print_function

import argparse
import numpy as np

import neuroglancer
import neuroglancer.cli
import webbrowser


def add_example_layers(state, data_path):
    a = np.load(data_path)

    print(a.shape)

    dimensions = neuroglancer.CoordinateSpace(
        names=["x", "y", "z"], units="nm", scales=[1, 1, 1]
    )

    state.dimensions = dimensions
    state.layers.append(
        name="a",
        layer=neuroglancer.LocalVolume(
            data=a,
            dimensions=dimensions,
            voxel_offset=(1, 1, 1),
            volume_type="image",
        )
    )
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, required=True, help='Path to .npy file to visualize')
    neuroglancer.cli.add_server_arguments(ap)
    args = ap.parse_args()
    neuroglancer.cli.handle_server_arguments(args)
    viewer = neuroglancer.Viewer()
    with viewer.txn() as s:
        a = add_example_layers(s, args.data_path)

    webbrowser.open_new(viewer.get_viewer_url())
