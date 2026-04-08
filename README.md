Overview of the tools is here: [GampyDocs.pdf](https://github.com/user-attachments/files/17451723/GampyDocs.pdf)

## Quick Start (Current Workflow)

Run from the parent directory of `Gampy`.

### Assumptions

- You have access to S3DF login/compute nodes.
- You have valid SLURM account/partition access (KIPAC or equivalent).
- `sbatch` is available in your shell.
- `cosima` is installed and available on compute nodes.
- You have write access to the configured scratch location.

### One-time setup

```bash
cd /sdf/home/b/bahrudin/gammaTPC/2026Sensitivity/Gampy
pip install -e .
```

### Parent-directory run pattern

```bash
cd /sdf/home/b/bahrudin/gammaTPC/2026Sensitivity

# Copy runtime script and required inputs to parent directory
cp Gampy/AstroGammaAnalyzer.py .
cp -r Gampy/default_inputs .

# If needed, copy or place an activation .dat in this directory
# (for example)
cp Gampy/ActivationFor550km_2to8keV.dat .

# Run
python AstroGammaAnalyzer.py --scenario nominal
```

### Notes

- If no `Activation*.dat` is found, the script launches activation setup and exits.
- Re-run the same command after activation is available.
- Scenario choices: `optimistic`, `nominal`, `pessimistic`, `pessimistic_good_light`.
