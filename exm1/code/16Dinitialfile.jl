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
using Sobol
using CSV



f(x)= π^2 .* sum(cos.(pi*x[i,:]') for i=1:16, dims=1)
acti(x)=@. swish(x)



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
Dense(16,32),
Block(32,32,32,32,acti),
Block(32,32,32,32,acti),
Block(32,32,32,32,acti),
Dense(32,1)
)

function A(x)
  points=@. abs(x^2-1)
  A = sin.(minimum(points,dims=1))
  return A
end

function B(x)
  points1=@. pi*x
  points2=@. abs(x^2-1)
  B1 = sum(cos.(points1), dims=1)
  B2 = cos.(minimum(points2,dims=1))
  return B1.*B2
end


F(x) =M(x).*A(x) .+ B(x)
