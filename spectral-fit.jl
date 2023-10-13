using VFitApproximation
using JSON
using DataFrames
using JLD2
using LinearAlgebra
using LaTeXStrings
using Statistics
using Polynomials
using PyCall
using CairoMakie
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
This function runs the large 100x100 dataset we used in the paper. The spectras are arranged as a |λ|x|Ω|x|ω| matrix which are fed into this function and after fitting a dictionary is provided
    of the approximated spectras, the errors and the final poles. We run this function to do complete the fit provided by 
    Inputs-  spectras_ALL        A |λ|x|Ω|x|ω| (100x100x400 for the paper) sized matrix that contains all the spectras that we are going to fit
             ω                   The frequency grid over which the spectral functions A(ω) are defined
             Ω                   A vector of Ω (phonon frequency) values corresponding to the 100 different spectra generated by tuning this parameter
             couplings           A vector of λ (electron-phonon couplings) corresponding to the 100 different spectra generated by tuning this parameter
             num_poles           The number of poles we will use to fit each of these functions 
             savedatafilename    Filename used to save the output dictionary 
             iterations          Number of iterations of the algorithm you want to run (default 21)
             regularizer         The value(s) of regularizer you want to use during the fitting procedure; default = 1e-3, If providing a vector of values, set multidescent = True
             conjugacy           Indicates whether half the poles should be forced to be conjugate of the others (set to True while fitting real functions), default=False
             columnwise          Indicates if you are providing poles as initial guess for fitting (useful if you want to use the poles of spectra with λ close to the one you are fitting as initial conditions)
             multidescent        Indicates if you are want your regularization annealed as the fitting takes place (use a vector of values in the regularizer parameter if you set this value as True)

    Outputs- VFIT_2x2_data       A dictionary containing the approximated spectras for each of the spectral functions, the poles obtained for each of those functions and the errors at final iteration/convergence

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

"""
This function takes in a pickle file of all 100x100 spectral data and runs our rational fitting algorithm.
    Inputs:- picklefile::String            The filename of the pickle file containing all the 100x100 data
             num_of_poles::Integer         The total number of poles 
             fit_savefilenamejld2::String  The filename of the JLD2 file containing the dictionary of rational fits of functions
             iterations::Integer           The number of iterations of the algorithm 
             regularizer                   The strength(s) of the regularizer γ in the algorithm 
             columnwise::Bool              Indicates if the fitting procedure should use the converged poles of the previously fit spectra as inputs. The spectra being fit in increasing order of coupling strength 
             conjugacy::Bool               Indicates if half the poles should be forced to be conjugates of the other
             multidescent::Bool            Indicates if the regularization should be annealed. Requires vector of inputs at the regularizer.

    Outputs:- All_spectral_data::DataFrame 100x100 Spectral Data 
              VFIT_2x2_dict:Dict            A dictionary containing the approximated spectras for each of the spectral functions, the poles obtained for each of those functions and the errors at final iteration/convergence
"""
function run_vfit_on100x100_data(picklefile,num_of_poles,fit_savefilenamejld2;iterations=21,regularizer::Union{Number,Vector}=1e-2,columnwise=false,conjugacy=false,multidescent=false)
    
    py"""
    import pickle
 
    def load_pickle(fpath):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        return data
    """

    load_pickle = py"load_pickle"

    All_spectral_data = load_pickle(picklefile)
    Ω = All_spectral_data["Omega"]
    ω  = All_spectral_data["omega_grid"]
    couplings = All_spectral_data["dimensionless_coupling"]
    spectras_ALL = All_spectral_data["A"]
    @time VFIT_2x2_dict = running_2_parameter_fits_larger_data(spectras_ALL,ω,Ω,couplings,num_of_poles,fit_savefilenamejld2;iterations=iterations,regularizer=regularizer,conjugacy=conjugacy,columnwise=columnwise,multidescent=multidescent)
    #plot_spectras_and_poles(spectras_ALL,ω,Ω,couplings,VFIT_2x2_dict)

    return All_spectral_data,VFIT_2x2_dict

end


"""
This function plots Figure 1 in our paper that shows the sample of Holstein spectral functions. 
    Inputs- f_df::DataFrame       These are the 100 Holstein spectral functions of either varying phonon frequencies or varying coupling strength
            Ω::Vector             This is the phonon frequecy values, can also be used to enter the coupling strenght values
            filename::String      The filename you want to save the plot as 
"""
function fig_1_BNLData_colorbar(f_df,Ω,filename)
    ω = f_df[:,1]
    A = f_df[:,2:end]

    ax = fig[1,1] = Axis(fig,
                    title=L"Spectral Function at various $\Omega$",
                    xlabel=L"\omega",
                    ylabel=L"A(\omega)",
                    xlabelsize=80,
                    ylabelsize=80,
                    xtickalign=1,
                    ytickalign=1,
                    xticksize=40,
                    yticksize=40,
                    xticklabelsize=60,
                    yticklabelsize=60,
                    titlesize=80)
    
    colorschm = ColorSchemes.coolwarm
    lines!(ω,Matrix(A[:,1:25:end]),color=:coolwarm)
    
    Colorbar(fig[1,2],limits=(Ω[1],Ω[end]),colormap=colorschm,label=L"$\Omega$ value",labelsize=100,ticklabelsize=80,ticksize=30,size=50)

    
    savefig(filename)

end

"""
This function plots Figure 3 in our paper that shows a comparison between approximating spectral functions using Chebyschev polynomials vs using rational approximation
    Inputs- spectra_df::DataFrame      The DataFrame containing all the spectral functions. The first column being the ω grid points over which the A(ω) is defined
            filename::String           The filename you want to save the plot as
"""
function fig_2_ChebyschevVsVFIT_MAElogplot(spectra_df::DataFrame,filename)
    ω = spectra_df[:,1]
    K = ncol(spectra_df)-1
    
    #Setup Chebyschev over all spectra
    parameter_vector = [10,20,40,100,160,250,320]
    upts_ =  (2*ω .- (ω[1]+ω[end]))/(ω[end]-ω[1])

    A_cheb_approxs = zeros(length(ω),length(parameter_vector),K)
    cheb_errors = -1.0*ones(length(parameter_vector))
    if isfile("Chebyschev_MAE_over_all_Ω_increasing_$(length(parameter_vector))paras_endw$(parameter_vector[end]).jld2")
        cheb_errors = load("Chebyschev_MAE_over_all_Ω_increasing_$(length(parameter_vector))paras_endw$(parameter_vector[end]).jld2","cheb_errors")
    else  
        for (index,params) in enumerate(parameter_vector)
            err_per_k = 0.0
            for k in 1:K
                A = spectra_df[:,k+1]
                chebT = fit(ChebyshevT,ω,A,params)
                spec_approx = chebT.(upts_)
                A_cheb_approxs[:,index,k] .= spec_approx
                err_per_k += mean(abs.(spec_approx .- A))
            end
            cheb_errors[index] = err_per_k/K
        end
        jldsave("Chebyschev_approx_of_100_spectra_$(length(parameter_vector))_diff_parameters_endw$(parameter_vector[end]).jld2";A_cheb_approxs)
        jldsave("Chebyschev_MAE_over_all_Ω_increasing_$(length(parameter_vector))paras_endw$(parameter_vector[end]).jld2";cheb_errors)
    end
    #Setup VFIT over all spectra
    vfit_paras = [2,4,8,16,32,64]
    VF_approxs = zeros(length(ω),length(vfit_paras),K)
    VF_errs = -1.0*ones(length(vfit_paras))
    VF_poles = Array{Any}(undef,length(vfit_paras),K)
    if isfile("Vfit_MAE_over_all_Ω_increasing_no_of_poles_$(length(vfit_paras))_polehopping.jld2")
            VF_errs = load("Vfit_MAE_over_all_Ω_increasing_no_of_poles_$(length(vfit_paras))_polehopping.jld2","VF_errs")
    else
        for (idx,m) in enumerate(vfit_paras)
            starting_ξ = rand(Complex{Float64},m)
            vf_err_perk = 0.0 
            for k in 1:K
                spectra = spectra_df[:,[1,k+1]]
                approx, err = VFitApproximation.vfitting(spectra,m,starting_ξ)
                approxImg = VFitApproximation.rapprox.((approx,),ω)
                VF_approxs[:,idx,k] .= real.(approxImg)
                VF_poles[idx,k] = approx.poles
                vf_err_perk += mean(abs.(approxImg .- spectra[:,2]))
                starting_ξ = approx.poles
            end
            VF_errs[idx] = vf_err_perk/K
        end
            jldsave("VFIT_approx_of_100_spectra_$(length(vfit_paras))_diff_parameters.jld2";VF_approxs)
            jldsave("VFIT_poles_of_100_spectra_$(length(vfit_paras))_diff_parameters.jld2";VF_poles)
            jldsave("Vfit_MAE_over_all_Ω_increasing_no_of_poles_$(length(vfit_paras)).jld2";VF_errs)
    end
    
    #Setup log-log plots
    fig = Figure(resolution=(1200,800))
    ax = fig[1,1] = Axis(fig,
            xticksize=30,
            yticksize=30,
            xticklabelsize=30,
            yticklabelsize=30,
            xtickalign=1,
            ytickalign=1,
            xlabel = L"$$(# of parameters)",
            ylabel= L"$\log_{10}$(MAE)",
            xlabelsize = 40,
            ylabelsize= 40,
            title = L"Chebyschev vs Rational approximation$$",
            titlesize = 40,
            yscale=log10)
            #yminorticksvisible = true)

    scatterlines!(ax,parameter_vector,cheb_errors,label=L"$\,$Chebyschev approximation")
    scatterlines!(ax,vfit_paras,VF_errs,label=L"$\,$Rational approximation")
    axislegend(ax,labelsize=30)

    save(filename,fig)
end

"""
This function plots the left half of Figure 6, that indicates the location of the poles in the complex plane
    Inputs:- ξ_poles::Matrix     = A Kx2m matrix of poles. Where K is the number of spectra and m is the number of poles
             Ω::Vector           = A vector of 100 boson frequencies that were tuned to generate the relevant 100 spectras. If plotting with respect to changing coupling strength, set coupling_or_Ω =True and use this parameter as the vector of coupling strenghts
             filename::String    = The filename you want to save your final image as
             coupling_or_Ω::Bool = Indicates whether the parameter (colorbar) indicates changing coupling strength (true) or phonon frequency (false) 
"""
function plot_single_neg_poles_1x5(ξ_poles,Ω,filename;coupling_or_Ω=false)
    fig = CairoMakie.Figure(resolution=(1000,500))
    ax = []
    K= size(ξ_poles,1)
    m = size(ξ_poles,2)
    τ = Int(ceil(m/2))
    for j in 1:τ
        axj = fig[1:2,j] = CairoMakie.Axis(fig,
                            xticksize=10,
                            yticksize=10,
                            xticklabelsize=20,
                            yticklabelsize=20,
                            xtickalign=1,
                            ytickalign=1,
                            xticks=WilkinsonTicks(3),
                            yticks=WilkinsonTicks(3),
                            spinewidth=3,
                            xticklabelfont="Times Roman",
                            yticklabelfont="Times Roman",
                            #xminortickvisible=false,
                            #yminortickvisible=false,
                            )

        hidedecorations!(axj,ticks=false,ticklabels=false)
        push!(ax,axj)
    end

    ξ_vec_neg = []
    ξ_vec_pos = []
    for i in 1:K
        sort!(ξ_poles[i,:],by=x-> (real(x),imag(x)))
        idxs = imag.(ξ_poles[i,:]) .< 0
        push!(ξ_vec_neg,ξ_poles[i,:][idxs])
        push!(ξ_vec_pos,ξ_poles[i,:][map(!,idxs)])
    end

    ξ_mat_neg = reduce(vcat,transpose.(ξ_vec_neg))
    ξ_mat_pos = reduce(vcat,transpose.(ξ_vec_pos))
    sort!(ξ_mat_neg,dims=2,by=x->(real(x),imag(x)))
    conj_size = size(ξ_mat_neg,2)

    
    for j in 1:conj_size
        x=real.(ξ_mat_neg[:,j])
        y=imag.(ξ_mat_neg[:,j])
        y_scaled = (y .* 1e2)
        CairoMakie.scatter!(ax[j],x,y_scaled,color=1:length(x),colormap=:viridis,markersize=15)
        Label(fig[1,j,Top()],halign= :left, L"\times 10^{-2}",textsize=20)
    end
    ylims!(ax[1],[-5.0031,-5.0019])
    ax[1].yticks = [-5.003,-5.002]
    #ax[2].yticks = [-6,-4]
    # ax[3].xticks=[-2.16,-2.12]
    # ax[1].yticks=[-5.0,-4.9]
    if coupling_or_Ω
        cbar_label = L"$\lambda$"
    else
        cbar_label = L"$\Omega/t$"
    end
    
    ylabel = Label(fig[1:2,0],L"Im($z$)",textsize=30, rotation=pi/2)
    xlabel = Label(fig[3,1:τ],L"Re($z$)",textsize=30)
    Colorbar(fig[4,1:τ],limits=(Ω[1],Ω[end]),colormap=:viridis,label=cbar_label,labelsize=30,ticklabelsize=20,ticksize=10,size=20,vertical=false,flipaxis=false, ticklabelfont="Times Roman")
    save(filename,fig)
    return (ξ_mat_pos,ξ_mat_neg)
end

"""
This function plots an equivalent of Fig 4, where the Mean approximation errors are plotted with respect to the parameter chosen (coupling strength λ or phonon frequency Ω)
    Inputs:- ME::Vector          The mean absolute errors of the spectral functions
             SE::Vector          The Standard error on the MAEs. 
             _x_label::String    The x axis label on the plot (Ω or λ)
             x_parameter::Vector The values on the x axis (Ω or λ)
             filename::String    The filename you want to save your figure


"""
function error_plot_for_couplings_and_Omegas(ME,SE,_x_label,x_parameter,filename)
    fig=Figure()
    ax = Axis(fig[1,1],
                xlabel=_x_label,
                ylabel="Mean Absolute Error",
                xlabelsize=20,
                ylabelsize=20,
                xticksize=10,
                yticksize=10,
                xticklabelsize=15,
                yticklabelsize=15,
                xtickalign=1,
                ytickalign=1
                # titlesize=35)
    )
    errorbars!(ax,x_parameter,ME,SE,whiskerwidth=3)

    save(filename,fig)
end


"""
This function makes some of the plots in the paper and runs the whole VFIt fit for 100 spectra with varying phonon frequency Ω but a fixed coupling strength λ. It is a subset of the fits done in the paper but is instructive

"""
function run_100spectra_data_and_plots()

    spectra_df, Ω = extract_dataframe_from_file("Example_22_04_01.json")

    mvfit = 10                                                                                              #Number of supports
    starting_ξ = rand(Complex{Float64},mvfit)                                                               #Initial set of poles
    rational_approxs = []
    err_for_each_spectra = []
    Spectra_approxs = []

    for j in 2:ncol(spectra_df)
        starting_ξ = [starting_ξ[1:2:end]; conj(starting_ξ_block[1:2:end])]
        approx, err = VFitApproximation.vfitting(spectra_df[:,[1,j]],mvfit,starting_ξ;iterations=51,force_conjugacy=true)                      #This function runs the VFIT algorithm
        push!(rational_approxs,approx)
        push!(err_for_each_spectra,err[1])
        approxImG = VFitApproximation.rapprox.((approx,),spectra_df[:,1])
        push!(Spectra_approxs,real(approxImG))
    end

    K = ncol(spectra_df)-1
    iters=length(err_for_each_spectra[1])
    ξ_poles = zeros(ComplexF64,K,mvfit)
    err_mat = zeros(K,iters)
    for j in 1:K
        ξ_poles[j,:] .= rational_approxs[j].poles
        err_mat[j,:] .= err_for_each_spectra[j]
    end

    mean_per_iter = mean(err_mat,dims=1)
    std_per_iter = std(err_mat,dims=1)

    jldsave("best_poles_from_VFit.jld2";ξ_poles)
    jldsave("best_rational_approxes_from_VFIT.jld2";rational_approxs)
    jldsave("100best_spectra_approximated_VFIT.jld2";Spectra_approxs)
    jldsave("VFITerrors_during_training_$(m)_poles.jld2";err_mat)

    iters = size(err_mat,2)
    plot_pole_locations(ξ_poles,Ω,"VFIT_poleLocations_$(mvfit)_poles.png")
    

    #Mean Square error for each spectra
    num_of_Ωs = length(Ω)
    SE_per_Ω = zeros(num_of_Ωs)
    ME_per_Ω = zeros(num_of_Ωs)
    for j in 1:num_of_Ωs
        err = abs.(spectra_df[:,j+1] .- Spectra_approxs[j])
        ME_per_Ω[j] = mean(err)
        SE_per_Ω[j] = std(mean(err,dims=2))/√(num_of_Ωs)
    end

    fig_1_BNLData_colorbar(spectra_df,Ω,"SampleSpectralData_provided.svg")
    fig_2_ChebyschevVsVFIT_MAElogplot(spectra_df,"VFit_vs_Cheby_MAE_plot.svg")
    plot_single_neg_poles_1x5(ξ_poles,Ω,"1x5VFIT_poleinterpolations_5params_100Spectra.svg")
    error_plot_for_couplings_and_Omegas(ME_per_Ω,SE_per_Ω,"Ω",Ω,"MAEplot_100spectra.svg")
    
end