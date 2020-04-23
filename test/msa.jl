using BenchmarkTools, Statistics

include("../src/DiffPsi.jl")
using .DiffPsi

@info "Testing MSA on CPU"

# initialize simulation parameters
soft_aperture = Dict("type" => "soft", 
                     "factor" => 25.0)
hard_aperture = Dict("type" => "hard")
probe = Dict("phase_error" => "spherical", 
             "C1" => 1, 
             "C3" => 1, 
             "C5" => 1, 
             "Scherzer" => false)
σ = 0.01
num_slices = 32
slice_thickness = 0.25
bandlimit = 1.0
λ = 0.05
sampling = 128
kmax = 4
semi_angle = 0.02
simParams = DiffPsi.SimulationState(λ, sampling, kmax, semi_angle, soft_aperture, probe, σ, slice_thickness, bandlimit)
@info "Using the following params" simParams

# initialize scan parameters
scan = DiffPsi.buildScan(simParams)

# build probes 
psi, psi_k, k_arr = DiffPsi.build_probe(simParams, scan)

# simulate 
V = randn(Float32, (simParams.sampling, simParams.sampling, num_slices))
DiffPsi.multislice!(psi, V, k_arr, simParams);
@info("Successfully propagated the Wavefunctions!
ѱ*ѱ: Type=$(typeof(psi)) & Shape=$(size(psi)).
V: Type= $(typeof(V)) & Shape=$(size(V))")
@test true 

# benchmarks
@info "Benchmarking on CPU"
bench = @benchmark DiffPsi.multislice!($psi, $V, $k_arr, $simParams)
    println("Time Elapsed:
        max = $(round(maximum(bench.times)*1e-9; digits=4)) s, 
        min (typical runtime) = $(round(minimum(bench.times)*1e-9; digits=4)) s,
        mean = $(round(mean(bench.times)*1e-9; digits=4)) s, 
        std = $(round(std(bench.times)*1e-9; digits=4)) s")
@test true
