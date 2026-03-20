# How to Run Tests and Examples

Because we added new packages (like `Plots.jl`), you might first need to tell Julia to download and install them. You only need to do this step once per project!

### 1. In your `lumen/TightBinding` project (`/home/attacc/SOFTWARE/lumen/TightBinding/`)

First, make sure you download the new plotting dependencies:
```bash
cd ~/SOFTWARE/lumen/TightBinding/
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**To run the unit tests** (which checks the `read_tb_model`, `hamiltonian`, `get_bands`, `plot_bands`, etc.):
```bash
# Option A: using Julia's built-in package manager 
julia --project=. -e 'using Pkg; Pkg.test()'

# Option B: directly running the file
julia --project=. test/runtests.jl
```

**To run the 2D Boron Nitride Example:**
```bash
julia --project=. examples/boron_nitride_example.jl
```


### 2. In your original `TightBinding.jl` project (`/home/attacc/SOFTWARE/TightBinding.jl/`)

If you want to run the tests in the older project where we built the lattice using `add_atoms!`, you can do:

**To run the unit tests** (this includes the 1D, 2D Square, Graphene, Iron Pnictides, and the Boron Nitride test block we added):
```bash
cd ~/SOFTWARE/TightBinding.jl/
julia --project=. test/runtests.jl 
# (You can also use: julia --project=. -e 'using Pkg; Pkg.test()')
```

**To run the Custom Boron Nitride Example:**
```bash
julia --project=. boron_nitride_example.jl
```
