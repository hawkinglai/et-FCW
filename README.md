# t-FCW Reproduction

## Overview

This repository presents grounded standalone t-FCW reproductions and a separate corrected GeoPCSD reconstruction. The historical protocol authority is the external companion source `reconstruction/`; it is not included here while redistribution rights remain unclear. Results and protocol limits are documented in [reproduction status](docs/reproduction_status.md).

## Main reproduced results

| Experiment | Paper | Measured | Provenance |
|---|---:|---:|---|
| ModelNet40 plain accuracy | 84.8% | [85.1297%](results/modelnet40/summary.json) | Historical runner; γ6090 selected using labeled test data |
| ShapeNet-Part plain mIoU | 70.50% | [70.18%](results/shapenet_part/summary.json) | Historical runner; γ230 |
| Table V RISP rotation | 73.9% in each condition | [74.0681, 73.9870, 74.0276, 73.9465%](results/risp_rotation/summary.json) | Original runner, none/none through SO(3)/SO(3) |
| Table VII ModelNet-C | 30 printed values | [30/30 match to two decimals](results/modelnetc/summary.json) | Plain, global/local/united modes |
| Table VIII ShapeNet-C | 10 printed values | [10/10 match to two decimals](results/shapenetc/summary.json) | Plain corruption evaluation |

The ModelNet-C and ShapeNet-C `add_global` files are functionally identified with the paper's Global corruption by numerical agreement; the paper does not expose that filename.

## Corrected GeoPCSD reconstruction

The canonical local corrected implementation is under [ScanObjectNN](experiments/geopcsd_scanobjectnn/) and [ShapeNet-Part](experiments/geopcsd_shapenet_part/). Corrected accuracies are [69.7074%, 72.9776%, and 58.0500%](results/geopcsd_scanobjectnn/summary.json) for the three ScanObjectNN splits; corrected ShapeNet-Part mIoU is [68.68%](results/geopcsd_shapenet_part/summary.json). The change repairs a duplicate-neighbor transcription error introduced during local reconstruction/copying. It does not establish an error in the published paper. See the [correction record](docs/geopcsd_correction.md).

## Repository structure

- `experiments/`: external-source command notes for faithful results and minimal corrected GeoPCSD source.
- `results/`: curated final summaries and essential logs only.
- `docs/`: protocol, status, and correction notes.
- `provenance/`: the one-line intervention patch and SHA-256 manifest.

## Data

Datasets and cached point clouds are excluded. See [DATA.md](DATA.md) for expected placement and unresolved acquisition provenance.

## Environment

See [ENVIRONMENT.md](ENVIRONMENT.md). The completed runs used a PyTorch CUDA environment and, for RISP, PyTorch3D transforms. Absolute paths in the original workstation's command records are not portable.

## Running the verified experiments

The [experiment index](docs/protocol_notes.md) identifies historical entry points and the corrected-source launchers. The five faithful runs require the separately obtained historical `reconstruction/` source and data; their code is not silently replaced by local rewritten runners. The corrected-source launchers use explicit dataset environment variables. Commands are supplied for provenance, not as a claim that this source-only release includes datasets or an environment lock.

## Reproduction limitations

Plain Table II ScanObjectNN and Table III S3DIS lack exact historical settings. Table VI adversarial is partial; historical xyz/GeoPCSD rotation and Tables IX/X executable variants are unresolved. The complete public status summary is [here](docs/reproduction_status.md). Corrected GeoPCSD results are separate from faithful historical reproductions.

## Citation

Please cite the [IEEE Transactions on Multimedia article](https://doi.org/10.1109/TMM.2026.3716075) (2026), DOI **10.1109/TMM.2026.3716075**; machine-readable metadata is in [CITATION.cff](CITATION.cff). The official repository's MIT attribution is retained in [LICENSE.txt](LICENSE.txt); scope and data caveats are in [LICENSE_STATUS.md](LICENSE_STATUS.md).

## Acknowledgments

This release is based on the original et-FCW code by Haijian Lai and coauthors. OpenAI Codex curated and cleaned the release presentation and verified the reported reproduction results against the saved experiment records. The historical implementation and scientific method remain credited to the original authors.
