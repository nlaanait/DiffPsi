module MSA

using CuArrays, CUDAdrv, CUDAnative
using FFTW, Statistics, Zygote
using BenchmarkTools

CuArrays.allowscalar(false)

"""
    build_fresnelPropagator(k, λ=0.1, Δ=0.1)  
Builds the Fresnel propagator:  
    𝒫 = exp(-iπλ k^2 z) 
# Arguments
- `k::Array`: 2-D array of reciprocal space coordinates.  
- `λ::Float`: wavelength
- `Δ::Float`: propagation distance 
"""
function build_fresnelPropagator!(fresnel_op, k, λ=0.1, z=0.1)
    cons = -1.0f0 * im * Float32(λ * z * π)
    fresnel_op .*= cons .* k .^ 2 
    fresnel_op .= exp.(fresnel_op)
    # return nothing
    # return fresnel_op
end

function build_fresnelPropagator(k, λ=0.1, z=0.1)
    cons = -1.0f0 * im * Float32(λ * z * π)
    fresnel_op = exp.(cons .* k .^ 2)
    return fresnel_op
end

"""
    build_transmPropagator(V, σ=200)  
Builds a transmission propagator 
    𝓣 = exp(-iσV)   
# Arguments
- `V::Array`: 2-d array (potential slice).  
- `σ::Float`: interaction parameter
"""
function build_transmPropagator!(transm, V, σ=200.0f0)
    transm .*= exp.( 1.0f0 * im * Float32(σ) * V) 
end

function build_transmPropagator(V, σ=200.0f0)
    trans = exp.( 1.0f0 * im * Float32(σ) * V) 
    return trans
end

"""
    multislice!(psi, potential, wavelength, slice_thickness, interaction_strength)
Iteratively propagates and transmits an initial wavefunction through a potential.  
   ѱ_{N+1} = 𝓕^{-1} { 𝒫 𝓕 { 𝓣_{N} ѱ_{N} }}  
𝒫: Fresnel propagator is calculated with `MSA.build_fresnelPropagator`   
𝓣: transmission propagator is calculated with `MSA.build_transmPropagator`  
# Arguments
- `psi::Array{Complex}: 2-d array`
- `potential::Array{Array{}}: array with N 2-d arrays`
"""
function multislice!(psi, potential, wavelength, slice_thickness, interaction_strength)
    if isa(psi, CuArray)
        k_arr = CuArrays.randn(Float32, size(psi))
        Fresnel_op = CuArrays.ones(ComplexF32, size(psi))
        transm = CuArrays.ones(ComplexF32, size(psi))
    else
        k_arr = randn(Float32, size(psi))
        Fresnel_op = ones(ComplexF32, size(psi))
        transm = ones(ComplexF32, size(psi)) 
    end
    build_fresnelPropagator!(Fresnel_op, k_arr, wavelength, slice_thickness)
    FFT_op = plan_fft!(psi)
    IFFT_op = plan_ifft!(psi)
    for slice_idx in 1:size(potential,3)
        build_transmPropagator!(transm, potential[:,:,slice_idx], interaction_strength)
        psi .= IFFT_op * ( Fresnel_op .* (FFT_op * (psi .* transm) / sum(size(psi))))
    end
end

function multislice(psi, potential, wavelength, slice_thickness, interaction_strength)
    k_arr = ones(size(psi))
    Fresnel_op = build_fresnelPropagator(k_arr, wavelength, slice_thickness)
    for slice_idx in 1:size(potential,3)
        trans = build_transmPropagator(potential[:,:,slice_idx], interaction_strength)
        psi = ifft(Fresnel_op .* (fft(copy(psi) .* trans)))
    end
    return psi
end

"""
    multislice_benchmark(device="cpu", k_size=(256,256), num_slices=256)
Runs benchmarks of multislice!()  
The `device` keyarg specifies the platform: "cpu" or "gpu"
"""
function multislice_benchmark(device="cpu", k_size=(256, 256), num_slices=256)
    if device == "cpu"
        psi = ones(ComplexF32, k_size)
        potential = ones(Float32, (k_size..., num_slices))
    elseif device == "gpu"
        psi = CuArrays.ones(ComplexF32, k_size)
        potential = CuArrays.ones(ComplexF32, (k_size..., num_slices))
    else
        error("Provided device $device is not available.")
    end
    @info("Running $device Benchmarks... \n Wavefunction dims:$k_size, Potential slices:$num_slices")
   
    bench = @benchmark multislice!($psi, $potential, 0.1, 0.1, 10)
    println("Time Elapsed:
        max = $(round(maximum(bench.times)*1e-9; digits=4)) s, 
        min = $(round(minimum(bench.times)*1e-9; digits=4)) s,
        std = $(round(std(bench.times)*1e-9; digits=4)) s")
end

end