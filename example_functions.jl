using VFitApproximation
using LinearAlgebra

"""
An example of fitting a square root function
    Inputs:- num_supp_pts::Integer      Number of poles for the rational approximation 
             start_::Integer            Starting index for data points
             fin_::Integer              Final Index for starting points

    Outputs:- R::VFit                   The rational approximation that is a struct of numerator weights φ, denominator weights ψ and poles 
              v::Vector                 The approximated values
              vers::Vector              The errors between the true values and the approximated values 
"""
function fit_sqrt(num_supp_pts,start_,fin_)

    ls = collect(range(start_,fin_,length=100))
    sort!(ls)
    xi = rand(Complex{Float64},num_supp_pts)

    R,ers = VFitApproximation.vfitting(sqrt,num_supp_pts,xi,ls,tol=1e-12,iterations=101)

    print(R.poles)
    
    pts = collect(range(start_,fin_,length=2000))
    v = VFitApproximation.rapprox.((R,),pts)
    vers = abs.(v .- sqrt.(pts))
    
    return R,v,vers
end

"""
An example of fitting a relativistic BreitWigner function 
    Inputs:- num_supp_pts::Integer      Number of poles for the rational approximation 
             start_::Integer            Starting index for data points
             fin_::Integer              Final Index for starting points

    Outputs:- R::VFit                   The rational approximation that is a struct of numerator weights φ, denominator weights ψ and poles 
              v::Vector                 The approximated values
              vers::Vector              The errors between the true values and the approximated values 
"""
function BreitWigner(num_supp_pts,start_,fin_)
    function BreitW(x)
        M=50
        Γ=8
        γ=M*sqrt(M^2+Γ^2)
        k=2*sqrt(2)*M*Γ*γ/(π*sqrt(M^2+γ))
        return k/((x^2-M^2)^2+M^2*Γ^2)
    end 
    ls = collect(range(start_,fin_,length=100))
    sort!(ls)
    xi = rand(Complex{Float64},num_supp_pts)

    R,ers = VFitApproximation.vfitting(BreitW,num_supp_pts,xi,ls,tol=1e-12,iterations=101,force_conjugacy=true)
    print(R.poles)
    
    pts = collect(range(start_,fin_,length=2000))
    v = VFitApproximation.rapprox.((R,),pts)
    vers = abs.(v .- BreitW.(pts))
    
    return R,v,vers
end

"""
An example of fitting a Lorentzian function
    Inputs:- num_supp_pts::Integer      Number of poles for the rational approximation 
             start_::Integer            Starting index for data points
             fin_::Integer              Final Index for starting points

    Outputs:- R::VFit                   The rational approximation that is a struct of numerator weights φ, denominator weights ψ and poles 
              v::Vector                 The approximated values
              vers::Vector              The errors between the true values and the approximated values 
"""
function fit_lorentzian(num_supp_pts,start_,fin_)

    ls = collect(range(start_,fin_,length=100))
    sort!(ls)
    xi = rand(Complex{Float64},num_supp_pts)
    lorenz_(x) = 1/(x^2+ϵ^2)
    R,ers = VFitApproximation.vfitting(lorenz_,num_supp_pts,xi,ls,tol=1e-12,iterations=101)

    print(R.poles)
    
    pts = collect(range(start_,fin_,length=2000))
    v = VFitApproximation.rapprox.((R,),pts)
    vers = abs.(v .- sqrt.(pts))
    
    return R,v,vers
end



"""
An example of fitting an absolute value function
    Inputs:- num_supp_pts::Integer      Number of poles for the rational approximation 
             start_::Integer            Starting index for data points
             fin_::Integer              Final Index for starting points
             x_0::Integer               The point of the kink of the absolute value function

    Outputs:- R::VFit                   The rational approximation that is a struct of numerator weights φ, denominator weights ψ and poles 
              v::Vector                 The approximated values
              vers::Vector              The errors between the true values and the approximated values 
"""
function fit_abs(num_supp_pts,start_,fin_,x_0)

    ls = collect(range(start_,fin_,length=100))
    sort!(ls)
    xi = rand(Complex{Float64},num_supp_pts)
    funcer(x) = abs(x-x_0)
    R,ers = VFitApproximation.vfitting(funcer,num_supp_pts,xi,ls,tol=1e-12,iterations=101)

    print(R.poles)
    
    pts = collect(range(start_,fin_,length=2000))
    v = VFitApproximation.rapprox.((R,),pts)
    vers = abs.(v .- sqrt.(pts))
    
    return R,v,vers
end
