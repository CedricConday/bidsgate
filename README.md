# bidsgate

**A recovery gate for neuroimaging pipelines.** Inject a known truth into real BIDS data,
run any BIDS app on the result, and score what it recovered. Every pipeline claims to
segment lesions or measure atrophy; this is the test that says by how much.

```bash
pip install bidsgate
bidsgate inject-lesions /data/bids --out /data/derivatives/bidsgate-lesions
# run your lesion segmenter on /data/derivatives/bidsgate-lesions
bidsgate score-lesions --truth /data/derivatives/bidsgate-lesions \
    --pred "/data/derivatives/mytool/{subject}/{base}_seg.nii.gz" --pipeline mytool
```

> Nothing in this repository is evidence about any disease. It is a test of software:
> the injected lesions and volume changes are synthetic, and the only claim made is
> about what a given pipeline recovered from them.

## Why

There is no public ground truth for most of what neuroimaging pipelines report. Lesion
segmenters are compared to expert masks that disagree with each other; morphometry tools
report volumes nobody can check; when a new release shifts the numbers, the changelog
says "improved" and the user has no way to tell. bidsgate gives every pipeline the same
question: here is a scan with a known change in it, what did you find?

This generalises the synthetic backtest of [lesiontrack](https://github.com/CedricConday/lesiontrack),
where injecting known lesion expansions showed a published method recovering a third of
the injected change and firing on noise. The same discipline applies to any pipeline.

## Injections

**Lesions** (`inject-lesions`): ellipsoidal white-matter lesions with Gaussian edges,
placed inside a white-matter estimate from the subject's own T1w, FLAIR-hyperintense and
T1w-hypointense relative to the surrounding white matter. Sizes cycle through 30, 80, 200,
600 and 1500 mm3 so that the scorecard shows a detection floor by lesion size. The truth
is a label map plus a JSON with every lesion's centre, axes, volume and voxel count, the
seed and the contrasts. T1w and FLAIR must share a grid (they do in most BIDS datasets;
register first if not).

**Atrophy** (`inject-atrophy`): a smooth radial contraction of the brain by a known volume
factor (default 0.95, five percent loss) about its centroid, fading to identity over 12 mm
outside the brain mask. The truth JSON records the factor and the brain volume before and
after as measured on the mask itself.

Both write a BIDS derivative dataset: `dataset_description.json`, the modified images with
their sidecars carrying what was done, and the truth files next to them.

## Scoring

`score-lesions` compares a predicted mask (binary or probabilistic, thresholded at 0.5)
with the truth on the truth grid: Dice, lesion-wise sensitivity (a lesion is detected
when any predicted voxel overlaps it), sensitivity by size bin, false-positive components
(predicted components overlapping no lesion) and their volume, and the predicted-over-
injected volume ratio. `score-atrophy` takes the volumes your tool reported before and
after injection and gives recovery: measured change over injected change, 1.0 being
exact.

Both write JSON and a single-file HTML scorecard.

## Limits, stated plainly

* Synthetic lesions are not real lesions. They have the contrast and shape the spec
  says, no more; a pipeline that finds them may still miss real ones, and a pipeline
  that misses them has a problem it cannot blame on pathology.
* The white-matter estimate is intensity-based, not a segmentation. Lesions land in
  bright, deep tissue; on unusual contrasts check the mask.
* Atrophy is global and radial. Regional atrophy needs a region mask; that is the next
  injector.
* Activation injection for fMRI is not built yet.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The tests build a phantom BIDS dataset and check that injected lesions have the recorded
volumes and contrasts, that a perfect prediction scores Dice 1, that a dilated prediction
with one missed lesion and one spurious blob scores as such, that atrophy shrinks the
brain by the requested factor, and that the CLI runs end to end.

MIT. Written by Cedric Conday with Claude (Anthropic) as coding partner.
