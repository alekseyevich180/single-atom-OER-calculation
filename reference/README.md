# Single-Atom Catalyst References

This directory contains a 1000-record curated library for single-atom catalyst papers on graphene, N-doped carbon/graphene-like supports, and selected other supports.

## Classification

- `Experimental/`: experimental papers, including papers that use DFT or simulations as supporting evidence.
- `Theoretical/`: pure theoretical/computational studies, including DFT screening and mechanism papers without new catalyst synthesis/measurement as the main contribution.

## Index Files

- `single_atom_catalyst_references.csv`: machine-readable master index.
- `single_atom_catalyst_references.xls`: Excel-compatible HTML workbook.
- `tools/build_reference_library.py`: repeatable OpenAlex-based curator/downloader.

## Notes

- Each record includes DOI, current accessible URL, open PDF URL if reported by metadata, PDF download status, local PDF path when downloaded, Chinese introduction, and classification rationale.
- PDFs are downloaded only when an open-access PDF URL is directly reachable from the current network. The script does not bypass publisher paywalls or institutional access controls.
- Multi-element papers are listed in each relevant element folder and in the master index with all detected elements.

## Refresh Command

```powershell
python .\reference\tools\build_reference_library.py --root .\reference --target 1000 --download-pdfs --clean-pdfs
```
