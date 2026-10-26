# Visibility findings during the frozen comparison

These findings do not change the frozen original experiment or raw labels.

Among its 589 eligible real box instances, 24 are explicitly marked LiDAR
occlusion `100 %`. All belong to one Bergedorf annotation ID; the same ID has
visible observations in other frames. The 24 boxes contain 12-89 raw returns,
so the old geometric five-point rule alone does not establish vehicle visibility.
Eight originally positive TRAIN frames become background-only in each of the
two affected training folds when those targets are excluded.

The retained public DZSF research report44/2023 establishes relevant context:

- Section2.4.3, page21: occluded objects use estimated true-size boxes. A narrow
  observed box is not sufficient justification for changing synthetic labels
  to visible-only point bounds.
- Section2.6.2, page43: objects seen in cameras but invisible in LiDAR can receive
  LiDAR annotations marked100% occluded.
- Section2.6.10, page46: cuboid semantic-segmentation indices are computed from
  points inside the cuboid. They are not independent proof of object-surface
  returns. For the hidden Bergedorf example, the stored semantic list contains
  the same37 points as the geometric cuboid count.

Source: https://edocs.tib.eu/files/e01fn23/1868657744.pdf, retained as
`osdar23_research_report.pdf`. The complete relevant pages were extracted and
visually inspected. The policy issue is in our LiDAR-only eligibility rule;
fully-occluded annotations can be legitimate for other dataset tasks.

`annotation_attribute_audit.json` preserves the metadata counts.
`synthetic_semantic_visibility_audit.json` separately checks actual ray-hit
instance IDs: none of520 evaluated observations across165 unique synthetic
training frames has fewer than five actual actor returns. Those IDs are used
only for annotation auditing, never as model features.

`visibility_aware_comparison.json`, when complete, is a secondary evaluation
with explicit fully-occluded GT ignored. It does not retroactively repair
training supervision or checkpoint selection.

The separate `../pointpillars_visibility_repair_pilot` reruns four real models
in folds3/4, where TRAIN contains the problematic observations and eligible
validation/test does not. It removes hidden positive targets and donor
observations, and ignores hidden-footprint classification anchors while
preserving visible-positive priority. Frame lists and optimizer settings stay
fixed. Synthetic models are unchanged references; after label repair their
visible-positive frame counts are no longer matched to real training.
