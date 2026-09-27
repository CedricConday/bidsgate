"""bidsgate: a recovery gate for neuroimaging pipelines.

Inject a known truth (lesions, atrophy) into real BIDS data, run any BIDS app on
the result, and score what it recovered. Nothing here is evidence about any
disease; it is a test of software.
"""

__version__ = "0.2.0"
