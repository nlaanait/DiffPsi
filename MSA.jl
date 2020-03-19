module MSA

using CuArrays, CUDAdrv, CUDAnative
using FFTW, Statistics, Zygote
using BenchmarkTools

# CuArrays.allowscalar(false)

function FFT_op(x, plan_mat)
    return plan_mat * copy(x)
end

function IFFT_op(x, plan_mat)
    return plan_mat * copy(x)
end

Zygote.@adjoint FFT_op(x) = Zygote.pullback(fft, x)
Zygote.@adjoint IFFT_op(x) = Zygote.pullback(ifft, x)


mutable struct SimulationState
    λ::Float32
    sampling::Int
    kmax::Float32
    semi_angle::Float32
    aperture::Dict
    probe::Dict
    σ::Float32
    slice_thickness::Float32
end

function build_probe(simParams::SimulationState, probe_positions=[[0 ; 0]])
    k_x, k_y = get_kspace_coordinates(simParams)
    k_rad = (k_x .^ 2.f0 + k_y .^ 2.f0) .^ 0.5f0
    aperture = get_aperture(simParams, k_rad)
    phase_error = get_phase_error(simParams, k_rad)
    
    psi_k = aperture .* phase_error
    if isa(probe_positions, CuArray)
        psi_k = cu(psi_k)
        k_x = cu(k_x)
        k_y = cu(k_y)
        psi_x = CuArrays.CuArray{ComplexF32}(undef, size(k_rad)...,length(probe_positions))
    else
        psi_x = Array{ComplexF32}(undef, size(k_rad)...,length(probe_positions))
    end
    FFT = plan_fft!(psi_x,[1,2])
    IFFT = inv(FFT)
    
    phase_shift!(psi_x, k_x, k_y, psi_k, probe_positions)
        
    psi_x .= IFFT * psi_x
    psi_x .= fftshift(psi_x, [1,2])
    norms = sqrt.(sum(abs2, psi_x, dims=[1,2]))
    psi_x ./= norms
    return psi_x, psi_k, k_rad
end

function phase_shift!(psi_x::Array, k_x::Array, k_y::Array, psi_k::Array, probe_positions::Array)
    for (psi, pos) in zip(eachslice(psi_x; dims=3), probe_positions)
        x, y = pos
        kr = k_x .* x + k_y .* y
        phase_shift = exp.(2.f0 * pi * im .* kr)
        psi .= psi_k .* phase_shift
    end
end

function phase_shift!(psi_x::CuArray, k_x::CuArray, k_y::CuArray, psi_k::CuArray, probe_positions::CuArray)
    function ps_kernel!(psi_x, k_x, k_y, psi_k, probe_positions)
        batch_idx = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
        row_idx   = (blockIdx().y - 1) * blockDim().y + threadIdx().y 
        col_idx   = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
        x, y = probe_positions[batch_idx]
        kr = k_x[row_idx, col_idx] * x + k_y[row_idx, col_idx] * y
        phase_shift = CUDAnative.exp(ComplexF32(2.f0 * pi * im * kr))
        psi_x[row_idx, col_idx, batch_idx] = psi_k[row_idx, col_idx] * phase_shift
        return nothing
    end
    threads = (min(32, size(psi_x,1)), min(32, size(psi_x,2)), 1)
    blocks = (ceil(Int, size(psi_x,1)/threads[1]), ceil(Int, size(psi_x, 2)/threads[2]), size(psi_x,3)) 
    CuArrays.@sync begin
        @info "Launching phase_shift_kernel"
        @cuda threads=threads blocks=blocks ps_kernel!(psi_x, k_x, k_y, psi_k, probe_positions)
        @info "Finished phase_shift_kernel"
    end
    synchronize()
end


function get_aperture(simParams, k_rad)
    k_semi = simParams.semi_angle / simParams.λ
    if simParams.aperture["type"] == "soft"
        aperture = (1 .+ exp.( -2 .* simParams.aperture["factor"] .* (k_semi .- k_rad))) .^ (-1)
    else
        heaviside(x) = if x > 0; 0 elseif x == 0. 0.5; else 1. end
        k_rad .-= k_semi
        aperture = heaviside.(k_rad)
    end
    return aperture
end
    
function get_kspace_coordinates(simParams::SimulationState)
    k_start, k_stop, k_step = -simParams.kmax/2, simParams.kmax/2, simParams.sampling
    k_y = [i for i in range(k_start,k_stop,length=k_step), j in range(k_start,k_stop,length=k_step)]
    k_x = [j for i in range(k_start,k_stop,length=k_step), j in range(k_start,k_stop,length=k_step)]
    return k_x, k_y
end

function get_probe_coordinates(simParams::SimulationState; fraction=0.5, origin=[0,0], grid_steps=[8,8])
    grid_range_start = (0.5 .+ origin * simParams.sampling .- simParams.sampling * fraction/4) ./ 2
    grid_range_stop = (0.5 .+ origin * simParams.sampling .+ simParams.sampling * fraction/4) ./ 2
    x_range = range(grid_range_start[1], stop=grid_range_stop[1], length=grid_steps[1])
    y_range = range(grid_range_start[2], stop=grid_range_stop[2], length=grid_steps[2])
    probe_positions = [(-i, j) for i in x_range, j in y_range]
    return probe_positions
end

function get_phase_error(simParams::SimulationState, k_rad)
    probe = simParams.probe
    if probe["phase_error"] == "spherical"
        C1 = probe["C1"]
        C3 = probe["C3"]
        C5 = probe["C5"]
        if probe["Scherzer"] 
            if C3 > 1.0 
                C1 = (1.5 * C3 * simParams.λ)
            else
                @warn "Spherical Aberration is too small- Not using Scherzer condition"
            end
        end
        λ = simParams.λ
        chi = 2.f0 * π / λ * (-1.f0/2 * C1 * (k_rad .* λ) .^ 2.f0 + 
                            1.f0/4 * C3 * (k_rad .* λ) .^ 4.f0 + 
                            1.f0/6 * C5 * (k_rad .* λ) .^ 6.f0)
        return exp.(-1.f0 * im .* chi)
    end
end
            

"""
    build_fresnelPropagator(k, λ=0.1, Δ=0.1)  
Builds the Fresnel propagator:  
    𝒫 = exp(-iπλ k^2 z) 
# Arguments
- `k::Array`: 2-D array of reciprocal space coordinates.  
- `λ::Float`: wavelength
- `Δ::Float`: propagation distance 
"""
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
function build_transmPropagator(V, σ=200.0f0)
    trans = exp.(1.0f0 * im * Float32(σ) * V) 
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
function multislice!(psi, potential, k_arr, simParams::SimulationState)
    Fresnel_op = build_fresnelPropagator(k_arr,simParams.λ, simParams.slice_thickness)
    FFT_op = plan_fft!(psi[:,:,1], [1,2])
    IFFT_op = plan_ifft!(psi[:,:,1], [1,2])
    for psi_pos in eachslice(psi; dims=3)
        interact!(psi_pos, potential, simParams, FFT_op, IFFT_op, Fresnel_op)
        psi_pos .= FFT_op * psi_pos
    end
end

function interact!(probe::SubArray, potential::Array, simParams, FFT_op, IFFT_op, Fresnel_op)
    for slice_idx in 1:size(potential,3)
        trans_op = build_transmPropagator(potential[:,:,slice_idx], simParams.σ)
        probe .= IFFT_op * ( Fresnel_op .* (FFT_op * (probe .* trans_op)))
    end 
end


function multislice(psi, potential, k_arr, simParams::SimulationState) 
    psi_buff = Zygote.Buffer(psi)
    psi_buff[:] = psi[:]
    Fresnel_op = build_fresnelPropagator(k_arr, simParams.λ, simParams.slice_thickness)
    for probe_idx in 1:size(psi_buff,3)
        psi_last = copy(psi_buff[:,:, probe_idx])
        for slice_idx in 1:size(potential,3)
            trans = build_transmPropagator(potential[:,:,slice_idx], simParams.σ) 
            psi_buff[:,:,probe_idx] = ifft(Fresnel_op .* fft(psi_last .* trans, [1,2]), [1,2])
            psi_last = copy(psi_buff[:,:,probe_idx])
        end
        psi_buff[:,:,probe_idx] = fft(psi_last)
    end
    return copy(psi_buff)
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