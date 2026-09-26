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

## First results: LST-AI v2 on healthy controls

[LST-AI](https://github.com/CompImg/LST-AI) v2.0.0rc1 (CPU image, fast mode) was run on
OpenNeuro [ds007908](https://openneuro.org/datasets/ds007908) controls after twelve
lesions (30 to 1500 mm3) were injected into each T1w and FLAIR. Scorecard and JSON are in
`results/lst-ai-v2/`; the remaining controls are being added as they finish.

| Subject | Detected | Dice | Volume ratio | Sensitivity 0-100 / 100-500 / 500+ mm3 | Extra components beyond 2 mm |
|---|---|---|---|---|---|
| sub-9000 | 9 of 12 | 0.75 | 1.15 | 0.83 / 0.50 / 0.75 | 16 (733 mm3) |
| sub-9001 | 11 of 12 | 0.80 | 1.34 | 0.83 / 1.00 / 1.00 | 32 (851 mm3) |

What the per-lesion tables show: every lesion in the cerebral white matter was found,
down to 26 mm3, with overlap fractions of 0.76 to 1.0. All four misses across the two
subjects are the lesions the placement put lowest in the brain: in sub-9000 the three at
cerebellum and brainstem level (29, 194 and 611 mm3), in sub-9001 the one lesion below
the brain's centre (28 mm3). Whether that is a weakness of the model or a weakness of
injecting supratentorial-looking lesions into infratentorial tissue is exactly the
question the gate raises and a per-region breakdown would answer; it is on the list
below.

The extra components on a healthy control are not necessarily wrong: a control can carry
real incidental white-matter hyperintensities, and the gate cannot tell those from false
positives. It can only say how much the pipeline reported beyond what was injected.

Two of the eight controls were refused by the input checks: sub-9005's FLAIR is on a
different grid from its T1w, and sub-9006's FLAIR shares the grid but not the affine
(17 mm apart), so it was never co-registered. A shape-only check had accepted it. The
gate refusing an input is a result too.

## Injections

**Lesions** (`inject-lesions`): ellipsoidal lesions with soft edges, placed inside a
white-matter estimate, FLAIR-hyperintense and T1w-hypointense relative to the median of
that estimate (gain 0.6 and −0.2 at the core by default). Sizes cycle through 30, 80,
200, 600 and 1500 mm3 so that the scorecard shows a detection floor by lesion size. No two
lesions touch, and every lesion lies deeper inside the brain than its own longest axis.

The brain mask is estimated from the T1w by morphology (tissue above an Otsu threshold,
eroded by 8 mm to cut scalp, optic nerves and cord, every remaining piece over 100 ml
grown back inside tissue, ventricles filled) and must land between 800 and 2000 ml or
the subject is refused. Pass your own mask with `--mask "{subject}_brainmask.nii.gz"`
if you have a better one. White matter is bright T1w tissue more than 6 mm inside that
mask whose FLAIR is within 0.6 to 1.4 of the FLAIR white-matter median, which excludes
CSF and anything outside the FLAIR field of view.

The soft field is 0.5 on the ellipsoid surface and falls off over 1 mm on either side,
so the truth label (the voxels inside the surface) is exactly what a half-maximum
segmenter would recover; a perfect segmenter scores Dice 1 and volume ratio 1, not 2.
The truth is the label map plus a JSON with every lesion's centre, axes, label volume,
nominal volume and voxel count, the seed, the contrasts and the brain volume.

T1w and FLAIR must share grid and affine; a subject that does not is skipped with a
message and nothing is written for it. Every image gets its own seed (a hash of its name
mixed with `--seed`), so `--subject` selection and dataset growth do not change what a
subject receives, and run or acquisition entities are kept in the derivative names.

**Atrophy** (`inject-atrophy`): a smooth radial contraction of the brain by a known volume
factor (default 0.95, five percent loss) about its centroid, fading to identity over 12 mm
outside the brain mask. The same mask estimate and `--mask` option apply. The truth JSON
records the factor and the brain volume before and after as measured on the mask itself.
Note that the skull contracts with the brain inside the falloff zone, so a tool that
normalises to intracranial volume will see less change than was injected; compare raw
volumes.

Both write a BIDS derivative dataset: `dataset_description.json`, the modified images with
their sidecars carrying what was done, and the truth files next to them.

## Scoring

`score-lesions` compares a predicted mask (binary or probabilistic, thresholded at 0.5)
with the truth, which must be on the same grid and affine: Dice, lesion-wise sensitivity
(a lesion is detected when any predicted voxel overlaps it), sensitivity by size bin, and
the predicted-over-injected volume ratio. False positives are every predicted voxel
farther than 2 mm (`--fp-margin`) from any injected lesion, reported as volume and as
18-connected components, so over-segmentation that happens to touch a true lesion still
counts. `score-atrophy` takes the volumes your tool reported before and after injection
and gives recovery: measured change over injected change, 1.0 being exact.

Both write JSON and a single-file HTML scorecard.

## Limits, stated plainly

* Synthetic lesions are not real lesions. They have the contrast and shape the spec
  says, no more; a pipeline that finds them may still miss real ones, and a pipeline
  that misses them has a problem it cannot blame on pathology.
* The white-matter estimate is intensity-based, not a segmentation, and it does not
  know cerebrum from cerebellum. Lesions land anywhere in deep bright tissue; a
  per-region breakdown (and a `--region` mask) is the next scoring feature.
* On real subjects, extra predicted components may be genuine findings. The gate
  reports them; it cannot judge them.
* Atrophy is global and radial. Regional atrophy needs a region mask; that is the next
  injector.
* Activation injection for fMRI is not built yet.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The tests build a head-shaped phantom (brain, skull gap, scalp) and check that the brain
estimate excludes the scalp, keeps both hemispheres across a fissure and fills
ventricles; that a slab of tissue with no plausible brain volume is refused; that injected
lesions have the recorded volumes and contrasts, sit entirely in white matter and never
touch; that the half-maximum set of the added contrast is the label; that a perfect
prediction scores Dice 1, a slab through a lesion counts as a false positive and a
shifted affine is refused; that atrophy shrinks the brain by the requested factor; that
run entities survive into derivative names with distinct seeds; that a subject with a
mismatched FLAIR leaves no partial output; and that the CLI runs end to end.

`scripts/` holds the LST-AI runner used for the result above (`run_lst_ai.sh`, detached
Docker container per subject; `overnight_demo.sh` for the whole cohort).

MIT. Written by Cedric Conday with Claude (Anthropic) as coding partner.
