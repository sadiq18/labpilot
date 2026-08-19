"""The validator seam (M12).

Deliberately empty of re-exports. `kaggle.py` imports `evidence.builder`, and
`evidence.builder` imports `models.py` from here — re-exporting the validator
would make importing the *models* pull in the builder mid-initialisation.
Import the submodule you want.
"""
