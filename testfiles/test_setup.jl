using CSV
using Distributions
# using Plots
# using StatsPlots
using QuadGK
using DataFrames
using Ipopt
using JuMP
using LoopVectorization
using StatsBase
using LinearAlgebra
using Random
using ForwardDiff
using FiniteDiff

include("../library/bt_american.jl")
include("../library/ewCov.jl")
include("../library/expost_factor.jl")
include("../library/fitted_model.jl")
include("../library/gbsm.jl")
include("../library/missing_cov.jl")
include("../library/return_calculate.jl")
include("../library/return_accumulate.jl")
include("../library/RiskStats.jl")
include("../library/simulate.jl")
include("../library/optimizers.jl")
include("../library/expost_factor.jl")

#Test 1 - missing covariance calculations
#Generate some random numbers with missing values.
function generate_with_missing(n,m; pmiss=.25)
    x = Array{Union{Missing,Float64},2}(undef,n,m)

    for i in 1:n, j in 1:m
        if rand() >= pmiss
            x[i,j] = randn()
        end
    end
    return x
end

Random.seed!(2)
x = generate_with_missing(10,5,pmiss=.2)
CSV.write("data/test1.csv",DataFrame(x,:auto))

x = CSV.read("data/test1.csv",DataFrame)
#1.1 Skip Missing rows - Covariance
cout = missing_cov(Matrix(x),skipMiss=true)
CSV.write("data/testout_1.1.csv",DataFrame(cout,:auto))
#1.2 Skip Missing rows - Correlation
cout = missing_cov(Matrix(x),skipMiss=true,fun=cor)
CSV.write("data/testout_1.2.csv",DataFrame(cout,:auto))
#1.3 Pairwise - Covariance
cout = missing_cov(Matrix(x),skipMiss=false)
CSV.write("data/testout_1.3.csv",DataFrame(cout,:auto))
#1.2 Pairwise - Correlation
cout = missing_cov(Matrix(x),skipMiss=false,fun=cor)
CSV.write("data/testout_1.4.csv",DataFrame(cout,:auto))

#Test 2 - EW Covariance
Random.seed!(3)
x = generate_with_missing(40,5,pmiss=0.0)
CSV.write("data/test2.csv",DataFrame(x,:auto))

x = CSV.read("data/test2.csv",DataFrame)
#2.1 EW Covariance λ=0.97
cout = ewCovar(Matrix(x),0.97)
CSV.write("data/testout_2.1.csv",DataFrame(cout,:auto))
#2.2 EW Correlation λ=0.94
cout = ewCovar(Matrix(x),0.94)
sd = 1 ./ sqrt.(diag(cout))
cout = diagm(sd) * cout * diagm(sd)
CSV.write("data/testout_2.2.csv",DataFrame(cout,:auto))
#2.3 EW Cov w/ EW Var(λ=0.94) EW Correlation(λ=0.97)
cout = ewCovar(Matrix(x),0.97)
sd1 = sqrt.(diag(cout))
cout = ewCovar(Matrix(x),0.94)
sd = 1 ./ sqrt.(diag(cout))
cout = diagm(sd1) * diagm(sd) * cout * diagm(sd) * diagm(sd1)
CSV.write("data/testout_2.3.csv",DataFrame(cout,:auto))

#Test 3 - non-psd matrices

#3.1 near_psd covariance
cin = CSV.read("data/testout_1.3.csv",DataFrame)
cout = near_psd(Matrix(cin))
CSV.write("data/testout_3.1.csv",DataFrame(cout,:auto))

#3.2 near_psd Correlation
cin = CSV.read("data/testout_1.4.csv",DataFrame)
cout = near_psd(Matrix(cin))
CSV.write("data/testout_3.2.csv",DataFrame(cout,:auto))

#3.3 Higham covariance
cin = CSV.read("data/testout_1.3.csv",DataFrame)
cout = higham_nearestPSD(Matrix(cin))
CSV.write("data/testout_3.3.csv",DataFrame(cout,:auto))

#3.2 Higham Correlation
cin = CSV.read("data/testout_1.4.csv",DataFrame)
cout = higham_nearestPSD(Matrix(cin))
CSV.write("data/testout_3.4.csv",DataFrame(cout,:auto))

#4 cholesky factorization
cin = Matrix(CSV.read("data/testout_3.1.csv",DataFrame))
n,m = size(cin)
cout = zeros(Float64,(n,m))
chol_psd!(cout,cin)
CSV.write("data/testout_4.1.csv",DataFrame(cout,:auto))


#5 Normal Simulation

Random.seed!(4)
cin = fill(0.75,(5,5)) + diagm(fill(0.25,5))
sd = 0.1 * randn(5).^2
cin = sd' .* cin .* sd
CSV.write("data/test5_1.csv",DataFrame(cin,:auto))
cin = fill(0.75,(5,5)) + diagm(fill(0.25,5))
cin[1,2] = 1
cin[2,1] = 1
cin = sd' .* cin .* sd
CSV.write("data/test5_2.csv",DataFrame(cin,:auto))
cin = fill(0.75,(5,5)) + diagm(fill(0.25,5))
cin[1,2] = 0
cin[2,1] = 0
cin = sd' .* cin .* sd
CSV.write("data/test5_3.csv",DataFrame(cin,:auto))

#5.1 PD Input
cin = CSV.read("data/test5_1.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin))
CSV.write("data/testout_5.1.csv",DataFrame(cout,:auto))

# 5.2 PSD Input
cin = CSV.read("data/test5_2.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin))
CSV.write("data/testout_5.2.csv",DataFrame(cout,:auto))

# 5.3 nonPSD Input, near_psd fix
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin,fixMethod=near_psd))
CSV.write("data/testout_5.3.csv",DataFrame(cout,:auto))

# 5.4 nonPSD Input Higham Fix
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
cout = cov(simulateNormal(100000, cin,fixMethod=higham_nearestPSD))
CSV.write("data/testout_5.4.csv",DataFrame(cout,:auto))

# 5.5 PSD Input - PCA Simulation
cin = CSV.read("data/test5_2.csv",DataFrame) |> Matrix
cout = cov(simulate_pca(cin,100000,pctExp=.99))
CSV.write("data/testout_5.5.csv",DataFrame(cout,:auto))

# Test 6

# 6.1 Arithmetic returns
prices = CSV.read("data/test6.csv",DataFrame)
rout = return_calculate(prices,dateColumn="Date")
CSV.write("data/test6_1.csv",rout)

# 6.2 Log returns
prices = CSV.read("data/test6.csv",DataFrame)
rout = return_calculate(prices,method="LOG", dateColumn="Date")
CSV.write("data/test6_2.csv",rout)

# Test 7

d = Normal(.05,.05)
x = rand(d,100)
CSV.write("data/test7_1.csv",DataFrame([x],:auto))

d = TDist(10)*.05 + .05
x = rand(d,100)
kurtosis(x)
CSV.write("data/test7_2.csv",DataFrame([x],:auto))

corr = fill(0.5,(3,3)) + I(3)*.5
sd = [.02,.03,.04]
covar = diagm(sd)*corr*diagm(sd)
x = rand(MvNormal([0,0,0],covar),100)'
e = rand(TDist(10)*.05 + .05,100)
B = [1,2,3]
y = x*B + e
cout = DataFrame(x,:auto)
cout[!,:y] = y
CSV.write("data/test7_3.csv",cout)


# 7.1 Fit Normal Distribution
cin = CSV.read("data/test7_1.csv",DataFrame) |> Matrix
fd = fit_normal(cin[:,1])
CSV.write("data/testout7_1.csv",DataFrame(:mu=>[fd.errorModel.μ],:sigma=>[fd.errorModel.σ]))

# 7.2 Fit TDist
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
CSV.write("data/testout7_2.csv",DataFrame(:mu=>[fd.errorModel.μ],:sigma=>[fd.errorModel.σ],:nu=>[fd.errorModel.ρ.ν]))

# 7.3 Fit T Regression
cin = CSV.read("data/test7_3.csv",DataFrame)
fd = fit_regression_t(cin.y,Matrix(select(cin,Not(:y))))
CSV.write("data/testout7_3.csv",
    DataFrame(:mu=>[fd.errorModel.μ],
            :sigma=>[fd.errorModel.σ],
            :nu=>[fd.errorModel.ρ.ν],
            :Alpha=>[fd.beta[1]],
            :B1=>[fd.beta[2]],
            :B2=>[fd.beta[3]],
            :B3=>[fd.beta[4]]            
))


# Test 8

# Test 8.1 VaR Normal
cin = CSV.read("data/test7_1.csv",DataFrame) |> Matrix
fd = fit_normal(cin[:,1])
CSV.write("data/testout8_1.csv",
    DataFrame(Symbol("VaR Absolute")=>[VaR(fd.errorModel)],
            Symbol("VaR Diff from Mean")=>[-quantile(Normal(0,fd.errorModel.σ),0.05)]
))

# Test 8.2 VaR TDist
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
CSV.write("data/testout8_2.csv",
    DataFrame(Symbol("VaR Absolute")=>[VaR(fd.errorModel)],
            Symbol("VaR Diff from Mean")=>[-quantile(TDist(fd.errorModel.ρ.ν)*fd.errorModel.σ,0.05)]
))

# Test 8.3 VaR Simulation
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
sim = fd.eval(rand(10000))
CSV.write("data/testout8_3.csv",
    DataFrame(Symbol("VaR Absolute")=>[VaR(sim)],
            Symbol("VaR Diff from Mean")=>[VaR(sim .- mean(sim))]
))


# Test 8.4 ES Normal
cin = CSV.read("data/test7_1.csv",DataFrame) |> Matrix
fd = fit_normal(cin[:,1])
CSV.write("data/testout8_4.csv",
    DataFrame(Symbol("ES Absolute")=>[ES(fd.errorModel)],
            Symbol("ES Diff from Mean")=>[ES(Normal(0,fd.errorModel.σ))]
))

# Test 8.5 ES TDist
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
CSV.write("data/testout8_5.csv",
    DataFrame(Symbol("ES Absolute")=>[ES(fd.errorModel)],
            Symbol("ES Diff from Mean")=>[ES(TDist(fd.errorModel.ρ.ν)*fd.errorModel.σ)]
))

# Test 8.6 VaR Simulation
cin = CSV.read("data/test7_2.csv",DataFrame) |> Matrix
fd = fit_general_t(cin[:,1])
sim = fd.eval(rand(10000))
CSV.write("data/testout8_6.csv",
    DataFrame(Symbol("ES Absolute")=>[ES(sim)],
            Symbol("ES Diff from Mean")=>[ES(sim .- mean(sim))]
))

# Test 9
A = rand(Normal(0,.03),200)
B = 0.1*A + rand(TDist(10)*.02,200)
CSV.write("data/test9_1_returns.csv",DataFrame(:A=>A,:B=>B))

# 9.1
cin = CSV.read("data/test9_1_returns.csv",DataFrame)
prices = Dict{String,Float64}()
prices["A"] = 20.0
prices["B"] = 30

models = Dict{String,FittedModel}()
models["A"] = fit_normal(cin.A)
models["B"] = fit_general_t(cin.B)

nSim = 100000

U = [models["A"].u models["B"].u]
spcor = corspearman(U)
uSim = simulate_pca(spcor,nSim)
uSim = cdf.(Normal(),uSim)

simRet = DataFrame(:A=>models["A"].eval(uSim[:,1]), :B=>models["B"].eval(uSim[:,2]))

portfolio = DataFrame(:Stock=>["A","B"], :currentValue=>[2000.0, 3000.0])
iteration = [i for i in 1:nSim]
values = crossjoin(portfolio, DataFrame(:iteration=>iteration))

nv = size(values,1)
pnl = Vector{Float64}(undef,nv)
simulatedValue = copy(pnl)
for i in 1:nv
    simulatedValue[i] = values.currentValue[i] * (1 + simRet[values.iteration[i],values.Stock[i]])
    pnl[i] = simulatedValue[i] - values.currentValue[i]
end

values[!,:pnl] = pnl
values[!,:simulatedValue] = simulatedValue

risk = select(aggRisk(values,[:Stock]),[:Stock, :VaR95, :ES95, :VaR95_Pct, :ES95_Pct])

CSV.write("data/testout9_1.csv",risk)

#10.1 Risk Parity, normal assumption
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
rpp,status = riskParity(cin)
CSV.write("data/testout10_1.csv",DataFrame(:W=>rpp))

#10.2 Risk Parity, normal assumption.  Half a risk share to X5
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
rpp,status = riskParity(cin,riskBudget=[1,1,1,1,0.5])
CSV.write("data/testout10_2.csv",DataFrame(:W=>rpp))

#10.3 Max Sharpe Ratio, normal assumption, w>0
means = [i for i in 0.09:-0.01:0.05]
CSV.write("data/test10_3_means.csv",DataFrame(:Mean=>means))

cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
means = CSV.read("data/test10_3_means.csv",DataFrame).mean
rf = 0.04
msr, status = maxSR(cin,means,rf)
CSV.write("data/testout10_3.csv",DataFrame(:W=>msr))

#10.4 Max Sharpe Ratio, normal assumption, 0.1 <= w <= 0.5
cin = CSV.read("data/test5_3.csv",DataFrame) |> Matrix
means = CSV.read("data/test10_3_means.csv",DataFrame).mean
rf = 0.04
bounds = hcat(fill(0.1,5),fill(0.5,5))
msr, status = maxSR(cin,means,rf,bounds)
CSV.write("data/testout10_4.csv",DataFrame(:W=>msr))

#11.1 Expost Attribution
stWgt = [.3, .2, .5]
CSV.write("data/test11_1_weights.csv",DataFrame(:W=>stWgt))

returns = CSV.read("data/test11_1_returns.csv",DataFrame)
stWgt = CSV.read("data/test11_1_weights.csv",DataFrame).W
Attribution, weights, factorWeights = expost_factor(stWgt,returns,returns,I(3))
select!(Attribution, Not(:Alpha))
CSV.write("data/testout11_1.csv",Attribution)

#11.2 Expost Factor Attribution
Random.seed!(112)
factor_returns = rand(MvNormal([0,0,0],I(3)),100)'*.01
beta = reshape(rand(Uniform(-1,1),6),2,3)
stock_returns = factor_returns * beta' + rand(MvNormal([0,0],I(2)),100)'*.01
stWgt = [.5, .5]
CSV.write("data/test11_2_weights.csv",DataFrame(:W=>stWgt))
CSV.write("data/test11_2_factor_returns.csv",DataFrame(factor_returns,[:F1, :F2, :F3]))
CSV.write("data/test11_2_stock_returns.csv",DataFrame(stock_returns,[:S1, :S2]))
CSV.write("data/test11_2_beta.csv",hcat(DataFrame(:Stock=>["S1","S2"]),DataFrame(beta,[:F1, :F2, :F3])))

stWgt = CSV.read("data/test11_2_weights.csv",DataFrame).W
factor_returns = CSV.read("data/test11_2_factor_returns.csv",DataFrame) |> Matrix
stock_returns = CSV.read("data/test11_2_stock_returns.csv",DataFrame) |> Matrix
beta = CSV.read("data/test11_2_beta.csv",DataFrame)[!,2:end] |> Matrix
Attribution, weights, factorWeights = expost_factor(stWgt,stock_returns,factor_returns,beta)
CSV.write("data/testout11_2.csv",Attribution)

#12.1 European Options GBSM with Greeks
options = filter(r->!ismissing(r.ID),CSV.read("data/test12_1.csv",DataFrame))
outVals = [gbsm(o["Option Type"] == "Call",o.Underlying,o.Strike,o.DaysToMaturity/o.DayPerYear,o.RiskFreeRate,o.RiskFreeRate - o.DividendRate, o.ImpliedVol; includeGreeks=true) for o in eachrow(options)]

values = [v.value for v in outVals]
deltas = [v.delta for v in outVals]
gammas = [v.gamma for v in outVals]
vegas = [v.vega for v in outVals]
rhos = [v.rho for v in outVals]
thetas = [v.theta for v in outVals]

CSV.write("data/testout12_1.csv",DataFrame(:ID=>options.ID,:Value=>values,:Delta=>deltas,:Gamma=>gammas,:Vega=>vegas,:Rho=>rhos,:Theta=>thetas))

#12.2 American Options with continous dividends including Greeks
options = filter(r->!ismissing(r.ID),CSV.read("data/test12_1.csv",DataFrame))

outVals = [bt_american(o["Option Type"] == "Call",o.Underlying,o.Strike,o.DaysToMaturity/o.DayPerYear,o.RiskFreeRate,o.RiskFreeRate - o.DividendRate, o.ImpliedVol,500) for o in eachrow(options)]

function fcall(_parms)
    parms = collect(_parms)
    bt_american(true,parms[1],parms[2],parms[3],parms[4],parms[5],parms[6],500)
end
function fput(_parms)
    parms = collect(_parms)
    bt_american(false,parms[1],parms[2],parms[3],parms[4],parms[5],parms[6],500)
end

deltas = Float64[]
gammas = Float64[]
vegas = Float64[]
rhos = Float64[]
thetas = Float64[]

for o in eachrow(options)
    parms = [o.Underlying,o.Strike,o.DaysToMaturity/o.DayPerYear,o.RiskFreeRate,o.RiskFreeRate - o.DividendRate, o.ImpliedVol]
    if o["Option Type"] == "Call"
        v = fcall(parms)
        grad = FiniteDiff.finite_difference_gradient(fcall,parms)
        push!(deltas,grad[1])
        d = 1.5
        parms[1] += d
        gamma1 = fcall(parms)
        parms[1] -= 2d
        gamma2 = fcall(parms)
        gamma = (gamma1 + gamma2 - 2*v)/(d^2)
        push!(gammas,gamma)
        push!(vegas,grad[6])
        push!(rhos,grad[4])
        push!(thetas,grad[3])
    else
        v = fput(parms)
        grad = FiniteDiff.finite_difference_gradient(fput,parms)
        push!(deltas,grad[1])
        d = 1.5
        parms[1] += d
        gamma1 = fput(parms)
        parms[1] -= 2d
        gamma2 = fput(parms)
        gamma = (gamma1 + gamma2 - 2*v)/(d^2)
        push!(gammas,gamma)
        push!(vegas,grad[6])
        push!(rhos,grad[4])
        push!(thetas,grad[3])
        
    end
end 

CSV.write("data/testout12_2.csv",DataFrame(:ID=>options.ID,:Value=>outVals,:Delta=>deltas,:Gamma=>gammas,:Vega=>vegas,:Rho=>rhos,:Theta=>thetas))

#12.3 American Options with Discrete Dividends including Greeks
options = filter(r->!ismissing(r.ID),CSV.read("data/test12_3.csv",DataFrame))
options.DividendDates = [parse.(Int,v) for v in split.(options.DividendDates,",")]
options.DividendAmts = [parse.(Float64,v) for v in split.(options.DividendAmts,",")]

options.N = options.DaysToMaturity * 2
options.DividendDates = options.DividendDates * 2

outVals = [bt_american(o["Option Type"] == "Call",
                o.Underlying,
                o.Strike,
                o.DaysToMaturity/o.DayPerYear,
                o.RiskFreeRate,
                o.DividendAmts,
                o.DividendDates,
                o.ImpliedVol,
                o.N) for o in eachrow(options)]

CSV.write("data/testout12_3.csv",DataFrame(:ID=>options.ID,:Value=>outVals))