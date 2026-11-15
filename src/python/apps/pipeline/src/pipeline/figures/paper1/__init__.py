"""Paper 1 publication exhibits, one module per experiment.

Each module renders one experiment's figures or table fragments from its saved
results and, where the experiment binds values into ``numbers.tex``, builds
those records; ``numbers`` fixes the record order. Import the entry point you
need from its module so a stage's dependency closure stays scoped to one
experiment -- do NOT re-export them here.
"""
