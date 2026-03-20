using Test
using TightBinding

@testset "TightBinding.jl" begin
    # 1. Test reading
    # Read a sample tight-binding model (e.g., a simple 1D chain with a single orbital).
    model = read_tb_model(joinpath(@__DIR__, "test_hr.dat"))
    @test model.num_wann == 1
    @test model.nrpts == 3
    
    # 2. Test Hamiltonian at k=0 (Gamma point)
    # For a simple 1D tight-binding chain with nearest-neighbor hopping `t`:
    # E(k) = -2t * cos(2π * k). Assuming t = 1, at k=0 we expect H = -2.
    H_gamma = hamiltonian(model, [0.0, 0.0, 0.0])
    @test isapprox(real(H_gamma[1, 1]), -2.0; atol=1e-5)
    @test isapprox(imag(H_gamma[1, 1]), 0.0; atol=1e-5)
    
    # 3. Test Hamiltonian at k=0.5 (X boundary of the 1D Brillouin Zone)
    # At k=0.5, the wavevector is π/a. The expected energy is E(k) = -2(1)cos(π) = 2.
    H_x = hamiltonian(model, [0.5, 0.0, 0.0])
    @test isapprox(real(H_x[1, 1]), 2.0; atol=1e-5)

    # 4. Test computing bands along a path in the Brillouin zone
    # Create a k-path from Gamma (k=0) to X (k=0.5) with 11 points
    kpoints = [[k, 0.0, 0.0] for k in range(0.0, 0.5, length=11)]
    vals = get_bands(model, kpoints)
    
    # We expect 1 band and 11 k-points
    @test size(vals) == (1, 11)
    
    # Verify that the band energies at the path ends match our earlier manual calculations
    @test isapprox(vals[1, 1], -2.0; atol=1e-5)
    @test isapprox(vals[1, 11], 2.0; atol=1e-5)

    # 5. Test if exported functions are available
    @test isdefined(TightBinding, :plot_bands)
    @test isdefined(TightBinding, :read_rotation_matrices)
    @test isdefined(TightBinding, :read_position_matrices)
end

println("Passed all tests.")
