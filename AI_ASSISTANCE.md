# AI assistance and provenance

This project uses AI assistance. The point-in-time feature-store design and requested
evaluation changes were specified by the user. An AI coding assistant implemented
the evaluation revision, wrote tests and documentation, and ran the checks recorded
in `VERIFICATION.md`.

The assisted revision added count-based and recency-based baselines, fixed-parameter
gradient boosting, customer-cluster bootstrap AUC intervals and paired differences,
prediction exports, target wording in API responses, and the evaluation report.
The baseline comparison is an actual run on the existing processed UCI database;
synthetic smoke-test results are labelled separately. The learners do not show a
clear AUC improvement over the 90-day count rule on that holdout.

## Git history

The workspace initially contained source code, tests, processed data and prior
verification artifacts, but no Git repository. The first commit imports that
snapshot; its original development history and individual authorship cannot be
established from the files. Later commits record the changes made in this revision.
They have actual present-day timestamps and a neutral repository-local author,
`Project automation <automation@localhost>`. No author name or contact details were
invented for the user, and the history was not backdated or rewritten to resemble
independent personal development.

These automated commits do not establish that the user personally authored or
reviewed the implementation. Human review is not yet recorded in this revision.
After reviewing, modifying and testing the project, the user should record their
own contributions under their own verified Git identity. Preserve this disclosure
and describe personal work accurately in interviews or a resume.

## Data provenance

UCI Online Retail is external public data, credited in `DATA_NOTICE.md` and the
README. Raw UCI timestamps describe events; they do not include historical ingestion
times. Features reconstruct event-time history from a pinned imported snapshot.
Runtime ingestion metadata and commit history are separate from dataset event time.
