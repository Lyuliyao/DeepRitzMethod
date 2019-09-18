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

f(x)= π^2 .* sum(cos.(pi*x[i,:]') for i=1:2, dims=1)
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
Dense(2,8),
Block(8,8,8,8,acti),
Block(8,8,8,8,acti),
Block(8,8,8,8,acti),
Dense(8,1)
)

function A(x)
  points=@. abs(x^2-1)
  A = exp.(minimum(points,dims=1)) .-1
  return A
end

function B(x)
  points1=@. pi*x
  points2=@. abs(x^2-1)
  B1 = sum(cos.(points1), dims=1)
  B2 = exp.(minimum(points2,dims=1))
  return B1.*B2
end


F(x) =M(x).*A(x) .+ B(x)

function lossFD(N)
  point=zeros(2,N)
  for i=1:N
    point[:,i]=next!(inner)
  end
  d=rand(1)/100
  notes=2*point.-1
  wt=4*ones(N)/N
  losses=0
  for i=1:2
    l2=zeros(2)
    l1=zeros(2)
    l2[i]=2
    l1[i]=1
    losses += ((0.5*((-1/12*F(notes .+ d.*l2   ) .+ 2/3*F(notes .+ d.*l1   ) .- 2/3*F(notes .- d.*l1   ) .+ 1/12*F(notes .- d.*l2   ))./d).^2) *wt)[1]
  end
  losses += ((-f(notes) .*F(notes))*wt)[1]
  return losses
end


function loss_true(xx=1,yy=1,xx1=1,xx2=2)
  points=rand(2,2000)
  points = 2* points .-1
  F_true(x)= sum(cos.(pi*x[i,:]') for i=1:2, dims=1)
  errors=sqrt(sum((F(points)-F_true(points)).^2)/2000)
  return errors
end

inner=SobolSeq(2)
BC=SobolSeq(1)

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
      CSV.write("$(N)Sobol2Dexm1.csv",errdf)
      @save "$(N)Sobolpoint2Dexm1m.bson" M
      weights=Tracker.data.(params(M))
      @save "$(N)Sobolpoint2Dexm1w.bson" weights
    end
  end
  θ=Flux.params(M)
  opt=ADAM()
  dataset=[(N) for i=1:traintime]
  Flux.train!(lossFD, params(M), zip(dataset), opt, cb=evalcb)
  return errdf
end
