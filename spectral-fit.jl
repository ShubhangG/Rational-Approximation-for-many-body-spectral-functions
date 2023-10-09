using VFitApproximation
using JSON
using DataFrames
using JLD2
using LinearAlgebra
using LaTeXStrings
using Statistics
using Polynomials
using PyCall
using ColorSchemes

"""
Given a vector of the real and imaginary parts of the poles, the following function computes the residues φ. Then returns φ and ξ (the poles) as a tuple
The 'state' vector is a vector of length '2m' (m being the number of poles). The first half of the vector consists of the real part of the poles and the second half consists of imaginary part of the poles.
Inputs: state= Vector of length 2*number of poles containing the real and imaginary values of the poles
        α    = The regularization parameter that keeps the residues at reasonably small values
        f    = The spectra data
Outputs: (ξ,φ) = A tuple of complex poles and their complex residues 
"""
function extract_model(state::Vector{Float64}; α::Float64, f::DataFrame)
    λ = f[:,1]
    N = length(λ)
    m = Int(length(state)/2)
    R = f[:,2]

    # Extract poles ξ from state
    x = state[1:m]
    Y = state[m+1:2m]
    y = -exp.(Y)
    ξ = x + im*y

    # Solve for optimal residues φ
    A = zeros(Float64, N, m)
    B = zeros(Float64, N, m)
    for i = 1:N
        for j = 1:m
            A[i, j] = real(1 / (λ[i] - ξ[j]))
            B[i, j] = imag(1 / (λ[i]  - ξ[j]))
        end
    end
    C = [B'*B B'*A; A'*B A'*A] + α*I
    temp = C \ [B'*R; A'*R]
    φ = temp[1:m] + im*temp[m+1:2m]

    return (ξ, φ)
end

"""
This function is used to obtain the barycentric weights φ and ψ given the poles by running a least square approximation to a matrix equation
    given by equation (11) in the paper
    Inputs  ξ      A vector of complex poles,
            α      The regression parameter
            λ      The sample points,
            Y      The function values that is being approximated at the sample points
    
    Outputs φ The barycentric weights of the numerator p(x)
            ψ The barycentric weights of the denominator q(x)
"""
function get_φ_ψ_from_polesVFIT(ξ::Vector{ComplexF64};α::ComplexF64,λ::Vector,Y::Vector)
    N = length(λ)
    m=length(ξ)
    C1 = zeros(ComplexF64,N,length(ξ))
    C2 = zeros(ComplexF64,N,length(ξ))
    for j in 1:N                                              #Build the initial matrix in RKFIT paper eq (A.4) to be solved
        C1[j,:] = 1 ./(λ[j] .- ξ)
        C2[j,:] = -Y[j] ./(λ[j] .- ξ)
    end
    C = hcat(C1,C2)
    Γ =  Γ_regrr(α,0,m)
    A=C
    #b=W*Y
    P = (A'A+ Γ)\(A'Y)

    return P[1:m], P[m+1:2m]

end

"""
This function builds the Tikhanov regularizer matrix 
    Inputs  λ_φ      The regularizer weight on the residues φ
            λ_ψ      The regularizer weight on the denominator barycentric weight ψ
    Outputs
            Γ        2mx2m Diagonal matrix containing λ_φ on top half and λ_ψ on bottom half
"""
function Γ_regrr(λ_φ,λ_ψ,m)
    iden1 =Diagonal([λ_φ,λ_ψ])
    iden2 = Matrix{ComplexF64}(I,m,m)
    return kron(iden1,iden2) 
end


"""
Function that extracts dataframe of spectras from json file given. It returns the dataframe and the set of parameters that generated each of the spectras. In this case the phonon frequency Ω.
Inputs- filename::String = Name of json file that contains the data
Output- spectra_df::DataFrame = A dataframe of 100 spectras evaluated at 400 frequency points. The first column is the ω value.
        Ω::Vector = A vector of 100 boson frequencies that were tuned to generate the relevant 100 spectras
"""
function extract_dataframe_from_file(filename)
    bnl_spectra_info = JSON.parsefile(filename)
    Ω = bnl_spectra_info["Omega"]
    G_im = bnl_spectra_info["G_imag"] 
    ω_grid =  bnl_spectra_info["omega_grid"]

    ω_grid = convert(Array{Float64,1},ω_grid)
    Ω = convert(Array{Float64,1},Ω)
    #all_keys = collect(keys(bnl_spectra_info))

    spectra_df = DataFrame(ω = ω_grid)
    ctr = 1
    for g in G_im
        spectra_df[!,"Ω = $(round(Ω[ctr],digits=3))"] = -1.0*g/π
        ctr=ctr+1
    end

    return spectra_df,Ω
end



"""
This function fits the 100 spectra corresponding to 100 different phonon frequencies Ω (|Ω|=100) at a particular coupling strength λ for the paper
    Inputs- spectra_phonon    A |Ω|x|ω| Matrix containing 100 spectral functions A(ω) evaluated at |ω|=400 datapoints
            ω                 The frequency grid over which the spectral functions A(ω) are defined
            Ω                 A vector of Ω (phonon frequency) values corresponding to the 100 spectra
            m                 Number of poles using to fit
            starting_ξ_block  A |Ω|xm matrix of starting pole positions to fit each of these spectral functions (used along with columnwise set True, used if you want to set provide a prior set of poles as starting configurations)
            iterations        Number of iterations of the algorithm you want to run (default 21)
            regularizer       The value(s) of regularizer you want to use during the fitting procedure; default = 1e-3, If providing a vector of values, set annealing = True
            conjugacy         Indicates whether half the poles should be forced to be conjugate of the others (set to True while fitting real functions), default=False
            columnwise        Indicates if you are providing poles as initial guess for fitting (useful if you want to use the poles of spectra with λ close to the one you are fitting as initial conditions)
            annealing         Indicates if you are want your regularization annealed as the fitting takes place (use a vector of values in the regularizer parameter if you set this value as True)
    
    Output- ξ_poles           The final poles after fitting procedure
            err_mat           The training error       
            Spectra_approxs
"""
function get_poles_per_coupling(spectra_phonon,ω,Ω,m,starting_ξ_block;iterations=21,regularizer::Union{Number,Vector}=1e-3,conjugacy=false,columnwise=false,annealing=false)  #58.171 s (44078894 allocations: 3.94 GiB)    
    spectra_df = DataFrame(ω = ω)
    ctr = 1
    
    #K=K-1
    for a in eachrow(spectra_phonon)
        spectra_df[!,"Ω = $(round(Ω[ctr],digits=3))"] = a
        ctr=ctr+1
    end
    
    grid_size,K = size(spectra_df)
    #Initial set of poles
    K = ncol(spectra_df)-1
    Spectra_approxs = zeros(K,grid_size)
    if !columnwise
        starting_ξ = [starting_ξ_block[1,:][1:2:end]; conj(starting_ξ_block[1,:][1:2:end])]
        sort!(starting_ξ,by=x->(real(x),imag(x)))
    end
    ξ_poles = zeros(ComplexF64,K,m)
    if length(regularizer)==1
        annealing=false
    end

    if annealing
        N=length(regularizer)
        iterations = Int(ceil(iterations/N))
        err_mat = zeros(K,iterations*N)
    else
        err_mat = zeros(K,iterations-1)
    end

    for j in 2:ncol(spectra_df)
        if columnwise
            starting_ξ = [starting_ξ_block[(j-1),:][1:2:end]; conj(starting_ξ_block[(j-1),:][1:2:end])]
            sort!(starting_ξ,by=x->(real(x),imag(x)))
        end
        local approx
        if annealing
            for n in eachindex(regularizer)
                if n==1
                    ξ_start = rand(ComplexF64,Int(length(starting_ξ)/2))
                    starting_ξ=[ξ_start;conj(ξ_start)]
                    sort!(starting_ξ,by=x->(real(x),imag(x)))
                    #approx, err2 = VFitApproximation.vfitting(spectra_df[:,[1,j]],m,starting_ξ;iterations=iterations,force_conjugacy=conjugacy,regression_param=regularizer[n])
                    approx, err2 = VFitApproximation.vfitting(spectra_df[:,[1,j]],m,starting_ξ;iterations=iterations,force_conjugacy=conjugacy,regression_param=regularizer[n])
                    train_err = err2[1]
                else
                    approx, err2 = VFitApproximation.vfitting(spectra_df[:,[1,j]],m,approx.poles;iterations=iterations,force_conjugacy=conjugacy,regression_param=regularizer[n])
                    train_err=[train_err;err2[1]]
                end
            end
        else
            ξ_start = rand(ComplexF64,Int(length(starting_ξ)/2))
            starting_ξ=[ξ_start;conj(ξ_start)]
            sort!(starting_ξ,by=x->(real(x),imag(x)))
            approx, err = VFitApproximation.vfitting(spectra_df[:,[1,j]],m,starting_ξ;iterations=iterations,force_conjugacy=conjugacy,regression_param=regularizer)  
            train_err = err[1]
        end

        
        ξ_poles[(j-1),:] .= approx.poles
        err_mat[(j-1),begin:length(train_err)] = train_err
        approxImG = VFitApproximation.rapprox.((approx,),spectra_df[:,1])
        Spectra_approxs[(j-1),:] = real(approxImG)
        if !columnwise
            starting_ξ = approx.poles
        end
    end

    return ξ_poles,err_mat,Spectra_approxs
end


"""
This function runs the large 100x100 dataset we used in the paper. The spectras are arranged as a |λ|x|Ω|x|ω| matrix. 



"""
function running_2_parameter_fits_larger_data(spectras_ALL,ω,Ω,couplings,num_poles,savedatafilename;iterations=21,regularizer::Union{Number,Vector}=0.0,conjugacy=false,columnwise=false,multidescent=false)
    approx_spectras_all = zeros(size(spectras_ALL))
    num_of_couplings, num_of_omegas, num_of_grid_points = size(spectras_ALL)
    VFIT_2x2_data = Dict()

    VFIT_poles = zeros(ComplexF64,num_of_couplings,num_of_omegas,num_poles)
    if multidescent
        number_of_descents = length(regularizer)
        iterations_per_run = Int(ceil(iterations/number_of_descents))
        VFIT_error_at_convergences = zeros(num_of_couplings,num_of_omegas,iterations_per_run*number_of_descents)
    else
        VFIT_error_at_convergences = zeros(num_of_couplings,num_of_omegas,iterations-1)
    end

 
    poles = rand(ComplexF64,num_of_omegas,num_poles)
    for j in 1:num_of_couplings
        poles,err_mat_,approx = get_poles_per_coupling(spectras_ALL[j,:,:],ω,Ω,num_poles,poles;iterations=iterations,regularizer=regularizer,conjugacy=conjugacy,columnwise=columnwise,annealing=multidescent)
        approx_spectras_all[j,:,:] = approx
        VFIT_poles[j,:,:] = poles
        VFIT_error_at_convergences[j,:,:] = err_mat_
        println("coupling = $(round(couplings[j],digits=3)) run over")
    end
    VFIT_2x2_data["approximated_spectras"] = approx_spectras_all
    VFIT_2x2_data["poles"] = VFIT_poles
    VFIT_2x2_data["convergence errors"] = VFIT_error_at_convergences

    jldsave(savedatafilename;VFIT_2x2_data)

    return VFIT_2x2_data
end 