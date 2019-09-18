using Flux.Tracker
using Flux
using Flux:throttle,glorot_uniform
using BSON:@save
using BSON:@load
using Base.Iterators: repeated
using Flux:@treelike
using PyPlot
using FastGaussQuadrature
using SparseGrids
using LinearAlgebra
using Flux: @epochs
using DataFrames
using CSV
using Sobol
using Pkg; Pkg.activate("cuda"); Pkg.instantiate()
using CuArrays
using ForwardDiff
using CUDAnative
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float64,1}, ::Val{2}) = x*x

f(x)=2*π^2 .*sum(cos.(π*x[i,:]') for i=1:8, dims=1)
acti(x)=@. x/(1+CUDAnative.exp(-x))


struct Block{F,S,T}
  W1::S
  W2::S
  b1::T
  b2::T
  σ::F
end

Block(W1 , W2, b1, b2) = Block(W1,W2 ,b1 ,b2, identity)

function Block(in1::Integer,in2::Integer, out1::Integer,out2::Integer, σ = identity;
  initW1 = glorot_uniform, initW2 =glorot_uniform, initb1 = zeros,initb2=zeros)
  return Block(param(initW1(out1, in1)),param(initW2(out2,in2)), param(initb1(out1)), param(initb2(out2)),σ)
end

@treelike Block

function (a::Block)(x)
  W1,W2, b1,b2, σ = a.W1,a.W2,a.b1, a.b2, a.σ
  σ.(W2 *σ.(W1*x .+ b1) .+ b2) .+ x
end





M=Chain(
Dense(8,30),
Block(30,30,30,30,acti),
Block(30,30,30,30,acti),
Block(30,30,30,30,acti),
Block(30,30,30,30,acti),
Dense(30,1)
)|>gpu

#=
@load "test1FD.bson" M
@load "test2FD.bson" weights
Flux.loadparams!(M, weights)
=#


#=

function loss(s,y=1)
  loss = sum((M(s) - 0.25*(s[1,:].^2+s[2,:].^2 .-1)').^2)
  return loss
end
=#

function lossFD(N)
  notes=rand(8,N)
  notes =notes |>gpu
  d=rand(1)/100 |>gpu
  losses=0
  MMM=M(notes)
  for i=1:8
    l1=zeros(8)
    l1[i]=1
    l1 = l1 |>gpu
    losses  += (sum(0.5*((M(notes .+ d.*l1) .- MMM)./d).^2)/N)[1]
  end
  losses += (sum(0.5*(π^2 * (MMM).^2))/N)[1]
  losses += (sum(-f(notes) .*MMM)/N)[1]
  return losses
end

function loss_true(xx=1,yy=1,xx1=1,xx2=2)
  points=zeros(8,5000)
    for i=1:5000
      points[:,i]=next!(truep)
    end
    points=points  |>gpu
  F_true(x)= sum(cos.(pi*x[i,:]') for i=1:8, dims=1)
  errors=sqrt(sum((M(points)-F_true(points)).^2)/2000)
  return errors
end

truep=SobolSeq(8)

function test(N;traintime=2000)
  errdf=DataFrame(time=Int[],absolute_error=Float64[],loss_FD=Float64[])
  cntr=1
  evalcb = function()
    loss_true1=loss_true()
    loss_FD = lossFD(N)
    push!(errdf,[Tracker.data(cntr),Tracker.data(loss_true1),Tracker.data(loss_FD)])
    cntr+=1
    if cntr%500==0
      @show(cntr,loss_FD,loss_true1)
      CSV.write("$(N)rand8Dexm2.csv",errdf)
      @save "$(N)randpoint8Dexm2m.bson" M
      weights=Tracker.data.(params(M))
      @save "$(N)randpoint8Dexm2w.bson" weights
    end
  end
  θ=Flux.params(M)
  opt=ADAM()
  dataset=[(N) for i=1:traintime]
  Flux.train!(lossFD, params(M), zip(dataset), opt, cb=evalcb)
  return errdf
end
