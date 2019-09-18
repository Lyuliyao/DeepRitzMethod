include("rand2Dexm1.jl")
result=DataFrame(Dimension=Int[],points=Int[],sampling_method=String[],relative_error_average=Float64[])
@load "2Dtrainm.bson" M
@load "2Dtrainw.bson" weights
Flux.loadparams!(M,weights)
x2=range(-1,1,length=51)
x_test=zeros(2,51*51)
for i=1:51
  for j=1:51
    x_test[:,51*(i-1)+j]=[x2[i];x2[j]]
  end
end
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
fontsize1=18
font1 = Dict("family"=>"sans-serif",
    "color"=>"k",
    "weight"=>"normal",
    "size"=>fontsize1)

F(x) =M(x).*A(x) .+ B(x)
F_true(x)= sum(cos.(pi*x[i,:]') for i=1:2, dims=1)
clf()
fig = figure("2Dcomparetureandtrain",figsize=(20,10))
subplot(121)
suptitle("Compare the 2 Dimensional true solution and the train solution",fontsize=fontsize1)
p1=contourf(x2,x2,Tracker.data(reshape(F(x_test),51,51)),10,fill=true)
ax = gca()
xlabel("x₁",fontdict=font1)
ylabel("x₂",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
cbar = colorbar()
cbar.ax.tick_params(labelsize=fontsize1)
PyPlot.title("train solution",fontdict=font1)
subplot(122)
p2=contourf(x2,x2,reshape(F_true(x_test),51,51),10,fill=true)
ax = gca()
cbar = colorbar()
cbar.ax.tick_params(labelsize=fontsize1)
xlabel("x₁",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
ylabel("x₂",fontdict=font1)
PyPlot.title("true solution",fontdict=font1)
savefig("C:\\Users\\lyu\\Documents\\deep-ritz\\LaTeX_DL_468198_240419\\figure\\comparetrueandtestDBC.eps")
using Statistics

clf()

"
125 point 2D problem
"

errdf1=CSV.read("125Sobol2Dexm1.csv")
errdf2=CSV.read("125rand2Dexm1.csv")
p1=scatter(errdf1.time,log.(errdf1.absolute_error),0.5,label="Sobol")
p1=scatter(errdf2.time,log.(errdf2.absolute_error),0.5,label="rand")

clf()
"
250 point 2D problem
"
errdf1=CSV.read("250Sobol2Dexm1.csv")
errdf2=CSV.read("250rand2Dexm1.csv")
p1=scatter(errdf1.time,log.(errdf1.absolute_error),0.5,label="Sobol")
p1=scatter(errdf2.time,log.(errdf2.absolute_error),0.5,label="rand")

legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
push!(result, (2,250,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,250,"Rand", mean(errdf2.absolute_error[end-1000:end])))


fig=figure("2DcompareQMCandMC",figsize=(20,20))
subplot(2,2,1)
suptitle("compare the accuracy of QMC and MC method in fixed number of point in 2 Dimensional problems",fontsize=fontsize1)

"
500 point 2D problem
"
errdf1=CSV.read("500Sobol2Dexm1.csv")
errdf2=CSV.read("500rand2Dexm1.csv")
p1=scatter(errdf1.time,log.(errdf1.absolute_error),1,label="Sobol")
p2=scatter(errdf2.time,log.(errdf2.absolute_error),1,label="rand")

push!(result, (2,500,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,500,"Rand", mean(errdf2.absolute_error[end-1000:end])))
legend(loc="upper right",fancybox="true",scatterpoints=1000)
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of mini-batch is 500 in each iteration",fontdict=font1)

subplot(2,2,2)

"
1000 point 2D problem
"
errdf1=CSV.read("1000Sobol2Dexm1.csv")
errdf2=CSV.read("1000rand2Dexm1.csv")
p1=scatter(errdf1.time,log.(errdf1.absolute_error),1,label="Sobol")
p2=scatter(errdf2.time,log.(errdf2.absolute_error),1,label="rand")

push!(result, (2,1000,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,1000,"Rand", mean(errdf2.absolute_error[end-1000:end])))

ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of mini-batch is 1000 in each iteration",fontdict=font1)


subplot(2,2,3)
"
2000 point 2D problem
"
errdf1=CSV.read("2000Sobol2Dexm1.csv")
errdf2=CSV.read("2000rand2Dexm1.csv")
scatter(errdf1.time,log.(errdf1.absolute_error),1,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error),1,label="rand")

push!(result, (2,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])))

ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of mini-batch is 2000 in each iteration",fontdict=font1)

subplot(2,2,4)
"
4000 point 2D problem
"
errdf1=CSV.read("4000Sobol2Dexm1.csv")
errdf2=CSV.read("4000rand2Dexm1.csv")
scatter(errdf1.time,log.(errdf1.absolute_error),1,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error),1,label="rand")

ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of mini-batch is 4000 in each iteration",fontdict=font1)
push!(result, (2,4000,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,4000,"Rand", mean(errdf2.absolute_error[end-1000:end])))

savefig("C:\\Users\\lyu\\Documents\\deep-ritz\\LaTeX_DL_468198_240419\\figure\\2DcompareDBC.eps")

"
500 point 4D problem
"
errdf1=CSV.read("500Sobol4Dexm1.csv")
errdf2=CSV.read("500rand4Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(4*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(4*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D500pointcompareexm1")

push!(result, (4,500,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(4*√(2))))
push!(result, (4,500,"Rand", mean(errdf2.absolute_error[end-1000:end])/(4*√(2))))


"
1000 point 4D problem
"
errdf1=CSV.read("1000Sobol4Dexm1.csv")
errdf2=CSV.read("1000rand4Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(4*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(4*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D1000pointcompareexm1")


push!(result, (4,1000,"Sobol",mean(errdf1.absolute_error[end-1000:end])/(4*√(2))))
push!(result, (4,1000,"Rand",mean(errdf2.absolute_error[end-1000:end])/(4*√(2))))


"
2000 point 4D problem
"
errdf1=CSV.read("2000Sobol4Dexm1.csv")
errdf2=CSV.read("2000rand4Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(4*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(4*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D2000pointcompareexm1")


push!(result, (4,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(4*√(2))))
push!(result, (4,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(4*√(2))))


"
4000 point 4D problem
"
errdf1=CSV.read("4000Sobol4Dexm1.csv")
errdf2=CSV.read("4000rand4Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(4*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(4*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D4000pointcompareexm1")


push!(result, (4,4000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(4*√(2))))
push!(result, (4,4000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(4*√(2))))


"
500 point 8D problem
"
errdf1=CSV.read("500Sobol8Dexm1.csv")
errdf2=CSV.read("500rand8Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/32),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/32),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D500pointcompareexm1")


push!(result, (8,500,"Sobol", mean(errdf1.absolute_error[end-1000:end])/32))
push!(result, (8,500,"Rand", mean(errdf2.absolute_error[end-1000:end])/32))



"
1000 point 8D problem
"
errdf1=CSV.read("1000Sobol8Dexm1.csv")
errdf2=CSV.read("1000rand8Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/32),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/32),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D1000pointcompareexm1")


push!(result, (8,1000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/32))
push!(result, (8,1000,"Rand", mean(errdf2.absolute_error[end-1000:end])/32))

"
2000 point 8D problem
"
errdf1=CSV.read("2000Sobol8Dexm1.csv")
errdf2=CSV.read("2000rand8Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/32),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/32),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D2000pointcompareexm1")


push!(result, (8,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/32))
push!(result, (8,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])/32))


"
10000 point 8D problem
"
errdf1=CSV.read("10000Sobol8Dexm1.csv")
errdf2=CSV.read("10000rand8Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/32),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/32),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D10000pointcompareexm1")

push!(result, (8,10000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/32))
push!(result, (8,10000,"Rand", mean(errdf2.absolute_error[end-1000:end])/32))



"
2000 point 16D problem
"
errdf1=CSV.read("2000Sobol16Dexm1.csv")
errdf2=CSV.read("2000rand16Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(512*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(512*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D2000pointcompareexm1")

push!(result, (16,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(512*√(2))))
push!(result, (16,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(512*√(2))))


"
5000 point 16D problem
"
errdf1=CSV.read("5000Sobol16Dexm1.csv")
errdf2=CSV.read("5000rand16Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(512*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(512*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D5000pointcompareexm1")


push!(result, (16,5000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(512*√(2))))
push!(result, (16,5000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(512*√(2))))


clf()

"
10000 point 16D problem
"
errdf1=CSV.read("10000Sobol16Dexm1.csv")
errdf2=CSV.read("10000rand16Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(512*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(512*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D10000pointcompareexm1")


push!(result, (16,10000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(512*√(2))))
push!(result, (16,10000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(512*√(2))))
clf()


"
20000 point 16D problem
"
errdf1=CSV.read("20000Sobol16Dexm1.csv")
errdf2=CSV.read("20000rand16Dexm1.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(512*√(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(512*√(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D20000pointcompareexm1")


#push!(result, (16,10000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(512*√(2))))
push!(result, (16,20000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(512*√(2))))
push!(result, (16,20000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(512*√(2))))
clf()


CSV.write("result.csv",result)

compareresult=DataFrame(Dimension=Int[],points=Int[],sampling_method=String[],relative_error_average=Float64[])

fig = figure("2DcomparefixDBC",figsize=(20,10))
subplot(121)


"
compare part 2D 10000
"
errdfS500=CSV.read("500sobol2Dexm1.csv")
scatter(errdfS500.time,log.(errdfS500.absolute_error),0.5,label="Sobol")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR10000=CSV.read("10000rand2Dexm1.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error),0.5,label="rand")

push!(compareresult, (2,500,"Sobol", mean(errdfS500.absolute_error[end-1000:end])))
push!(compareresult, (2,10000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])))

legend(loc="upper right",fancybox="true",scatterpoints=1000)
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of MC's mini-batch is 10000 \n while that of QMC method is 500 in each iteration",fontdict=font1)

subplot(122)


"
compare part 2D 100000
"
errdfS2000=CSV.read("2000Sobol2Dexm1.csv")
scatter(errdfS2000.time,log.(errdfS2000.absolute_error),0.5,label="Sobol")
errdfR100000=CSV.read("100000rand2Dexm1.csv")
scatter(errdfR100000.time,log.(errdfR100000.absolute_error),0.5,label="rand")
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of MC's mini-batch is 100000 \n while that of QMC method is 2000 in each iteration",fontdict=font1)

suptitle("compare the mini-batch size of MC with that of QMC method to achieve the same accuracy in 2 Dimensional problems",fontsize=fontsize1)
savefig("C:\\Users\\lyu\\Documents\\deep-ritz\\LaTeX_DL_468198_240419\\figure\\2D2compareDBC.eps")


push!(compareresult, (2,2000,"Sobol", mean(errdfS2000.absolute_error[end-1000:end])))
push!(compareresult, (2,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])))


"
compare part 4D 10000
"
errdfS500=CSV.read("500sobol4Dexm1.csv")
scatter(errdfS500.time,log.(errdfS500.absolute_error/(4*√(2))),0.5,label="Sobol500")
errdfR10000=CSV.read("10000rand4Dexm1.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error/(4*√(2))),0.5,label="rand10000")
legend(loc="upper right",fancybox="true")
savefig("4DS10000compareexm1")
clf()


push!(compareresult, (4,500,"Sobol", mean(errdfS500.absolute_error[end-1000:end])/(4*√(2))))
push!(compareresult, (4,10000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])/(4*√(2))))


"
compare part 4D 100000
"
errdfS4000=CSV.read("4000sobol4Dexm1.csv")
scatter(errdfS4000.time,log.(errdfS4000.absolute_error/(4*√(2))),0.5,label="Sobol4000")
errdfR100000=CSV.read("100000rand4Dexm1.csv")
scatter(errdfR100000.time,log.(errdfR100000.absolute_error/(4*√(2))),0.5,label="rand100000")
legend(loc="upper right",fancybox="true")
savefig("4DS100000compareexm1")


push!(compareresult, (4,4000,"Sobol", mean(errdfS4000.absolute_error[end-1000:end])/(4*√(2))))
push!(compareresult, (4,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])/(4*√(2))))


clf()


"
compare part 8D
"
errdfS2000=CSV.read("2000sobol8Dexm1.csv")
scatter(errdfS2000.time,log.(errdfS2000.absolute_error/(32)),0.5,label="Sobol2000")
errdfR10000=CSV.read("10000rand8Dexm1.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error/(32)),0.5,label="rand10000")
legend(loc="upper right",fancybox="true")
savefig("8DS10000compareexm1")



push!(compareresult, (8,2000,"Sobol", mean(errdfS2000.absolute_error[end-1000:end])/32))
push!(compareresult, (8,10000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])/32))

clf()
"
compare part 8D
"
errdfS10000=CSV.read("10000sobol8Dexm1.csv")
scatter(errdfS10000.time,log.(errdfS10000.absolute_error/(32)),0.5,label="Sobol10000")
errdfR100000=CSV.read("100000rand8Dexm1.csv")
scatter(errdfR100000.time,log.(errdfR100000.absolute_error/(32)),0.5,label="rand100000")
legend(loc="upper right",fancybox="true")
savefig("8DS100000compareexm1")
push!(compareresult, (8,10000,"Sobol", mean(errdfS10000.absolute_error[end-1000:end])/32))
push!(compareresult, (8,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])/32))

clf()
"
compare part 16D
"
errdfS1000=CSV.read("1000sobol16Dexm1.csv")
scatter(errdfS1000.time,log.(errdfS1000.absolute_error/(512*√(2))),0.5,label="Sobol1000")
errdfR10000=CSV.read("10000rand16Dexm1.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error/(512*√(2))),0.5,label="rand10000")
legend(loc="upper right",fancybox="true")
savefig("16DS10000compareexm1")


push!(compareresult, (16,1000,"Sobol", mean(errdfS1000.absolute_error[end-1000:end])/(512*√(2))))
push!(compareresult, (16,10000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])/(512*√(2))))


clf()
"
compare part 16D
"
errdfS5000=CSV.read("5000sobol16Dexm1.csv")
scatter(errdfS5000.time,log.(errdfS5000.absolute_error/(512*√(2))),0.5,label="Sobol5000")
errdfR100000=CSV.read("100000rand16Dexm1.csv")
scatter(errdfR100000.time,log.(errdfR100000.absolute_error/(512*√(2))),0.5,label="rand100000")
legend(loc="upper right",fancybox="true")
savefig("16DS100000compareexm1")

push!(compareresult, (16,5000,"Sobol", mean(errdfS5000.absolute_error[end-1000:end])/(512*√(2))))
push!(compareresult, (16,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])/(512*√(2))))

CSV.write("compareresult.csv",compareresult)
