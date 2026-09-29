# DRLE Architecture Record

## Initial decision
Use a deterministic variable-order context-to-next-token count model as the falsifiable baseline for the relation-set concept. Contexts are ordered tuples, not sets. Candidate generation is sparse and never enumerates context × vocabulary.

## Data contract
Input rows may contain `prompt`, `messages` (role/content objects), and `category`. For each assistant message, emit a training sequence consisting of the prompt, preceding system/user history, an `assistant:` marker, and the assistant content. The test split must never be mixed into training.

## Scoring contract
For each available suffix context of order k, candidate evidence is occurrence_count × k². Evidence is aggregated across matching orders. Sort by descending integer score, then ascending normalized token string. This heuristic is a baseline, not a calibrated probability.

## Halt contract
No candidate after lookup and constraints yields `NO_SUPPORTED_CONTINUATION`. A missing exact context alone is not a halt; shorter suffix contexts are checked.

## Next engineering gates
1. Verify dataset revision, schema, license, and train/test separation.
2. Add evaluation metrics: held-out token accuracy, top-k accuracy, coverage, perplexity only if a valid probability estimator is implemented, repetition rate, latency, peak RSS, artifact bytes.
3. Compare against unigram/bigram/trigram baselines.
4. Replace JSON artifact with a versioned binary format only after profiling.
5. Add Rust runtime only after Python reference behavior is locked by golden tests.
