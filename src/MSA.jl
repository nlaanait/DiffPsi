# module MSA

using CuArrays, CUDAdrv, CUDAnative
using FFTW, Statistics, Zygote


mutable struct SimulationState
    λ::Float32
    sampling::Int
    kmax::Float32
    semi_angle::Float32
    aperture::Dict
    probe::Dict
    σ::Float32
    Δ::Float32
    bandwidth::Float32
end

mutable struct Scan
    x::Union{Array, CuArray}
    y::Union{Array, CuArray}
end

function cu(scan::Scan)
    return Scan(CuArrays.cu(scan.x), CuArrays.cu(scan.y))
end

function build_probe(simParams::SimulationState, 
                     pos=Scan(zeros(Float32, 1, 1, 1), zeros(Float32, 1, 1, 1)))
    # build probe in k-space
    k_x, k_y = get_kspace_coordinates(simParams)
    k_rad = (k_x .^ 2.f0 + k_y .^ 2.f0) .^ 0.5f0
    aperture = get_aperture(simParams, k_rad)
    phase_error = get_phase_error(simParams, k_rad)
    psi_k = aperture .* phase_error

    # transform probe to real space
    if isa(pos.x, CuArray)
        @info "Executing on GPU"
        psi_x = CuArrays.CuArray{ComplexF32}(undef, size(k_rad)...,length(pos.x))
        k_rad, psi_k = map(CuArrays.cu, (k_rad, psi_k))
    else
        psi_x = Array{ComplexF32}(undef, size(k_rad)...,length(pos.x))
    end
    FFT = plan_fft!(psi_x,[1,2])
    IFFT = inv(FFT)
    
    phase_shift!(psi_x, k_x, k_y, psi_k, pos)
        
    psi_x .= IFFT * psi_x
    psi_x .= fftshift(psi_x, [1,2])
    norms = sqrt.(sum(abs2, psi_x, dims=[1,2]))
    psi_x ./= norms
    psi_k = Complex{Float32}.(psi_k)
    return psi_x, psi_k, k_rad
end

function phase_shift!(psi_x::Array, psi_k, k_x, k_y, pos::Scan)
    psi_k, k_x, k_y = map( x -> repeat(x, outer=(1, 1, length(pos.x))), (psi_k, k_x, k_y))
    kr = k_x .* pos.x + k_y .* pos.y
    phase_shift = exp.(2.f0 * pi * im .* kr)
    psi_x .= psi_k .* phase_shift
end

function phase_shift!(psi_x::CuArray, psi_k, k_x, k_y, scan::Scan)
    psi_k, k_x, k_y = map( x -> CuArrays.cu(x), (psi_k, k_x, k_y))
    psi_k, k_x, k_y = map( x -> repeat(x, outer=(1, 1, length(scan.x))), (psi_k, k_x, k_y))
    kr = k_x .* scan.x + k_y .* scan.y
    phase_shift = exp.(2.f0 * pi * im .* kr)
    psi_x .= psi_k .* phase_shift
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
        aperture = (1.f0 .+ exp.( -2.f0 .* simParams.aperture["factor"] .* (k_semi .- k_rad))) .^ (-1.f0)
    else
        k_rad .-= k_semi
        aperture = heaviside.(k_rad)
    end
    return aperture
end

heaviside(x) = if x > 0; 0.f0 elseif x == 0 0.5f0; else 1.f0 end

function get_bandwidth_mask(simParams::SimulationState)
    radius = simParams.bandwidth * simParams.sampling
    grid_start, grid_stop, grid_num = -simParams.sampling/2, simParams.sampling/2 , simParams.sampling
    y = [i for i in range(grid_start,grid_stop,length=grid_num), j in range(grid_start,grid_stop,length=grid_num)]
    x = [j for i in range(grid_start,grid_stop,length=grid_num), j in range(grid_start,grid_stop,length=grid_num)]
    r = radius .- sqrt.( x .^ 2 + y .^ 2)
    if simParams.aperture["type"] == "soft"
        return 1 ./ (1 .+ exp.( -2 .* simParams.aperture["factor"] .* r)) 
    else
        return heaviside.(-r)
    end
end

function get_kspace_coordinates(simParams::SimulationState)
    k_start, k_stop, k_num = -simParams.kmax/2, simParams.kmax/2, simParams.sampling
    k_y = [i for i in range(k_start,k_stop,length=k_num), j in range(k_start,k_stop,length=k_num)]
    k_x = [j for i in range(k_start,k_stop,length=k_num), j in range(k_start,k_stop,length=k_num)]
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

function buildScan(simParams::SimulationState; fraction=0.5, origin=[0,0], grid_steps=[8,8])
    grid_range_start = (0.5 .+ origin * simParams.sampling .- simParams.sampling * fraction/4) ./ 2
    grid_range_stop = (0.5 .+ origin * simParams.sampling .+ simParams.sampling * fraction/4) ./ 2
    x_range = range(grid_range_start[1], stop=grid_range_stop[1], length=grid_steps[1])
    y_range = range(grid_range_start[2], stop=grid_range_stop[2], length=grid_steps[2])
    probe_positions = [[-i, j] for i in x_range, j in y_range]
    probe_positions = hcat(probe_positions...)
    x, y = probe_positions[1,:], probe_positions[2,:]
    x = reshape(x, 1, 1, :)
    y = reshape(y, 1, 1, :)
    scan = Scan(x, y)
    return scan
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
𝒫: Fresnel propagator is calculated with `build_fresnelPropagator`   
𝓣: transmission propagator is calculated with `build_transmPropagator`  
# Arguments
- `psi::Array{Complex}: 2-d array`
- `potential::Array{Array{}}: array with N 2-d arrays`
"""
function multislice!(psi::Union{Array, CuArray}, potential, k_arr, simParams::SimulationState)
    Fresnel_op = build_fresnelPropagator(k_arr,simParams.λ, simParams.Δ)
    FFT_op = plan_fft!(psi, [1,2])
    IFFT_op = plan_ifft!(psi, [1,2])
    for slice_idx in 1:size(potential, 3)
        trans_op = trans_op = build_transmPropagator(potential[:,:,slice_idx], simParams.σ)
        psi .= IFFT_op * (Fresnel_op .* (FFT_op * (psi .* trans_op))) 
    end
    psi .= FFT_op * psi
end
