using Test, CUDAapi 

if has_cuda()
    @testset "CUDA Tests" begin
        include("cuda_msa.jl")
    end
else
    @warn("Skipping CUDA Tests since no CUDA-enabled GPU has been detected")
end

# @testset "MSA Tests" begin
#     include("msa.jl")
# end

@testset "Grad Tests" begin
    include("grad_msa.jl")
end 