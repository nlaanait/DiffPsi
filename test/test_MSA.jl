using CuArrays, BenchmarkTools, Statistics
include("../src/MSA.jl")
using .MSA


soft_aperture = Dict("type" => "soft", 
                     "factor" => 25.0)
hard_aperture = Dict("type" => "hard")
probe = Dict("phase_error" => "spherical", 
             "C1" => 1, 
             "C3" => 1, 
             "C5" => 1, 
             "Scherzer" => false)
σ = 0.01
num_slices = 64
slice_thickness = 0.25
bandlimit = 1.0
λ = 0.05
sampling = 256
kmax = 4
semi_angle = 0.02
# initialize simulation parameters
simParams = MSA.SimulationState(λ, sampling, kmax, semi_angle, soft_aperture, probe, σ, slice_thickness, bandlimit)
@info "Using the following params" simParams

### cpu
@info "Testing MSA on CPU"
# initialize scan parameters
scan = MSA.buildScan(simParams)
# build probes 
psi, psi_k, k_arr = MSA.build_probe(simParams, scan)
# propagates probes 
V = randn(Float32, (simParams.sampling, simParams.sampling, num_slices))
MSA.multislice!(psi, V, k_arr, simParams);
@info "Successfully propagated Probes Shape: $(size(psi)) through $num_slices slices"

### benchmarks
@info "Benchmarking on CPU"
bench = @benchmark MSA.multislice!($psi, $V, $k_arr, $simParams)
    println("Time Elapsed:
        max = $(round(maximum(bench.times)*1e-9; digits=4)) s, 
        min (typical runtime) = $(round(minimum(bench.times)*1e-9; digits=4)) s,
        mean = $(round(mean(bench.times)*1e-9; digits=4)) s, 
        std = $(round(std(bench.times)*1e-9; digits=4)) s")

### gpu
@info "Testing MSA on GPU"
# move scan parameters to gpu, triggering all execution to take place on gpu
cu_scan = MSA.cu(scan)
# build probes 
psi, psi_k, k_arr = MSA.build_probe(simParams, cu_scan)
# propagates probes 
V = CuArrays.randn(Float32, (simParams.sampling, simParams.sampling, num_slices))
MSA.multislice!(psi, V, k_arr, simParams)
@info "Successfully propagated Probes Shape: $(size(psi)) through $num_slices slices"

### benchmarks
@info "Benchmarking on GPU"
bench = @benchmark MSA.multislice!($psi, $V, $k_arr, $simParams)
    println("Time Elapsed:
        max = $(round(maximum(bench.times)*1e-9; digits=4)) s, 
        min (typical runtime) = $(round(minimum(bench.times)*1e-9; digits=4)) s,
        mean = $(round(mean(bench.times)*1e-9; digits=4)) s, 
        std = $(round(std(bench.times)*1e-9; digits=4)) s")
