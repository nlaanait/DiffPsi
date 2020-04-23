using NPZ, Zygote

include("../src/DiffPsi.jl")
using .DiffPsi

# set parameters of forward model
soft_aperture = Dict("type" => "soft", 
                     "factor" => 25.0)
hard_aperture = Dict("type" => "hard")
probe = Dict("phase_error" => "spherical", 
             "C1" => 1e2, 
             "C3" => -5e4, 
             "C5" => 3e3, 
             "Scherzer" => false)
sigma = 0.01
num_slices = 3
slice_thickness = 0.25
bandlimit = 1.0
simParams = DiffPsi.SimulationState(0.05, 128, 4, 0.02, soft_aperture, probe, 
                                    sigma, slice_thickness, bandlimit)

# initialize scan parameters
scan = DiffPsi.buildScan(simParams)

# load target potential
v_file = npzread("data/potential_445.npy")
crop_top = map(Int, [size(v_file)[1] * (1/4), size(v_file)[2] * (3/4)])
slice = range(crop_top[1], stop=crop_top[2]-1)
v_file = v_file[slice, slice]

V_trg = randn(ComplexF32, (simParams.sampling, simParams.sampling, num_slices)) 
for idx in 1:size(V_trg, 3)
    V_trg[:,:,idx] = v_file
end

# generate solution from the forward model
@info("Simulating solution w/ the Forward Model...") 
psi_trg, psi_k, k_arr = DiffPsi.build_probe(simParams, scan)
cpy_trg = copy(psi_trg)
DiffPsi.multislice!(psi_trg, V_trg, k_arr, simParams)
psi2_trg = abs.(psi_trg)
@info("Successfully propagated the Wavefunctions!
ѱ: Type=$(typeof(psi_trg)) & Shape=$(size(psi_trg)).
V: Type= $(typeof(V_trg)) & Shape=$(size(V_trg))") 
# DiffPsi.comp_plot(psi2_trg[:,:,5], angle.(psi_k), V_trg)
@test true

@info("Initializing Differentiable Forward Model, ∂F...")
psi_mixing = 0.f0
V_mixing = 0.f0
psi_k_phase = angle.(psi_k[:,:,1]) 
psi_in_phase = psi_mixing * psi_k_phase + (1 - psi_mixing) * randn(Float32, size(psi_k[:,:,1]))
V_in = V_mixing * V_trg + (1 - V_mixing) * randn(Float32, size(V_trg)) 
# V and phase must be complex-valued to match gradients
V_in .+= 0.f0 * im
psi_in_phase .+= 0.f0 * im 

# build forward model
init_data = DiffPsi.initialData(simParams, scan)
model = DiffPsi.ForwardModel(psi_in_phase, V_in, simParams, "∂F")
psi2_pred = DiffPsi.forward(model, init_data)
@info("Successfully propagated the Wavefunctions!
ѱ*ѱ: Type=$(typeof(psi2_pred)) & Shape=$(size(psi2_pred)).
V: Type= $(typeof(V_trg)) & Shape=$(size(V_trg))")
@test true 


@info("Differentiating ∂F...")
grads = gradient(model) do m
    return DiffPsi.loss(m, psi2_trg, init_data)
end
grads = grads[1][]
@info("Successfully Differentiated the Forward Model!
∂V: Type=$(typeof(grads.potential)) & Shape=$(size(grads.potential)))
∂arg(ѱ): Type=$(typeof(grads.phase)) & Shape=$(size(grads.phase))")
@test true
# grad_plot(grads)

@info("Optimizing ∂F...")

