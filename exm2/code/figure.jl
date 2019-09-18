include("C:\\Users\\lyu\\Documents\\deep-ritz\\julia\\exm1new\\rand2Dexm1.jl")
result=DataFrame(Dimension=Int[],points=Int[],sampling_method=String[],relative_error_average=Float64[])
using Statistics
@load "500Sobolpoint2Dexm2m.bson" M
@load "500Sobolpoint2Dexm2w.bson" weights
Flux.loadparams!(M,weights)
x2=range(0,1,length=51)
x_test=zeros(2,51*51)
for i=1:51
  for j=1:51
    x_test[:,51*(i-1)+j]=[x2[i];x2[j]]
  end
end

fontsize1=18
font1 = Dict("family"=>"sans-serif",
    "color"=>"k",
    "weight"=>"normal",
    "size"=>fontsize1)

F_true(x)= sum(cos.(pi*x[i,:]') for i=1:2, dims=1)
clf()
fig = figure("2Dcomparetureandtrain",figsize=(20,10))
subplot(121)
suptitle("Compare the 2 Dimensional true solution and the train solution",fontsize=fontsize1)
p1=contourf(x2,x2,Tracker.data(reshape(M(x_test),51,51)),10,fill=true)
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
savefig("C:\\Users\\lyu\\Documents\\deep-ritz\\LaTeX_DL_468198_240419\\figure\\comparetrueandtestNBC.eps")


@load "250Sobolpoint2Dexm2m.bson" M
@load "250Sobolpoint2Dexm2w.bson" weights


fig=figure("2DcompareQMCandMC",figsize=(20,20))
subplot(2,2,1)
suptitle("compare the accuracy of QMC and MC method in fixed number of point in 2 Dimensional problems",fontsize=fontsize1)


"
250 point 2D problem
"
errdf1=CSV.read("250Sobol2Dexm2.csv")
errdf2=CSV.read("250rand2Dexm2.csv")
scatter(errdf1.time,log.(errdf1.absolute_error),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error),0.5,label="rand")
push!(result, (2,250,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,250,"Rand", mean(errdf2.absolute_error[end-1000:end])))
legend(loc="upper right",fancybox="true",scatterpoints=1000)
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of mini-batch is 250 in each iteration",fontdict=font1)

subplot(2,2,2)

"
500 point 2D problem
"
errdf1=CSV.read("500Sobol2Dexm2.csv")
errdf2=CSV.read("500rand2Dexm2.csv")
scatter(errdf1.time,log.(errdf1.absolute_error),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error),0.5,label="rand")
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

subplot(2,2,3)
"
1000 point 2D problem
"
errdf1=CSV.read("1000Sobol2Dexm2.csv")
errdf2=CSV.read("1000rand2Dexm2.csv")
scatter(errdf1.time,log.(errdf1.absolute_error),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error),0.5,label="rand")
push!(result, (2,1000,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,1000,"Rand", mean(errdf2.absolute_error[end-1000:end])))
legend(loc="upper right",fancybox="true",scatterpoints=1000)
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of mini-batch is 1000 in each iteration",fontdict=font1)

subplot(2,2,4)

"
2000 point 2D problem
"
errdf1=CSV.read("2000Sobol2Dexm2.csv")
errdf2=CSV.read("2000rand2Dexm2.csv")
scatter(errdf1.time,log.(errdf1.absolute_error),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error),0.5,label="rand")
push!(result, (2,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])))
push!(result, (2,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])))
legend(loc="upper right",fancybox="true",scatterpoints=1000)
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of mini-batch is 2000 in each iteration",fontdict=font1)
savefig("2D")
savefig("C:\\Users\\lyu\\Documents\\deep-ritz\\LaTeX_DL_468198_240419\\figure\\2DcompareNBC.eps")

"
500 point 4D problem
"
errdf1=CSV.read("500Sobol4Dexm2.csv")
errdf2=CSV.read("500rand4Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/sqrt(2)),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/sqrt(2)),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D500pointcompareexm2")

push!(result, (4,500,"Sobol", mean(errdf1.absolute_error[end-1000:end])/sqrt(2)))
push!(result, (4,500,"Rand", mean(errdf2.absolute_error[end-1000:end])/sqrt(2)))


"
1000 point 4D problem
"
errdf1=CSV.read("1000Sobol4Dexm2.csv")
errdf2=CSV.read("1000rand4Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/sqrt(2)),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/sqrt(2)),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D1000pointcompareexm2")


push!(result, (4,1000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/sqrt(2)))
push!(result, (4,1000,"Rand", mean(errdf2.absolute_error[end-1000:end])/sqrt(2)))


"
2000 point 4D problem
"
errdf1=CSV.read("2000Sobol4Dexm2.csv")
errdf2=CSV.read("2000rand4Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/sqrt(2)),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/sqrt(2)),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D2000pointcompareexm2")


push!(result, (4,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/sqrt(2)))
push!(result, (4,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])/sqrt(2)))


"
5000 point 4D problem
"
errdf1=CSV.read("5000Sobol4Dexm2.csv")
errdf2=CSV.read("5000rand4Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/sqrt(2)),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/sqrt(2)),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("4D5000pointcompareexm2")
push!(result, (4,5000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/sqrt(2)))
push!(result, (4,5000,"Rand", mean(errdf2.absolute_error[end-1000:end])/sqrt(2)))

"
500 point 8D problem
"
errdf1=CSV.read("500Sobol8Dexm2.csv")
errdf2=CSV.read("500rand8Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/2),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/2),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D500pointcompareexm2")


push!(result, (8,500,"Sobol", mean(errdf1.absolute_error[end-1000:end])/2))
push!(result, (8,500,"Rand", mean(errdf2.absolute_error[end-1000:end])/2))


"
1000 point 8D problem
"
errdf1=CSV.read("1000Sobol8Dexm2.csv")
errdf2=CSV.read("1000rand8Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/2),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/2),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D1000pointcompareexm2")


push!(result, (8,1000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/2))
push!(result, (8,1000,"Rand", mean(errdf2.absolute_error[end-1000:end])/2))


"
2000 point 8D problem
"
errdf1=CSV.read("2000Sobol8Dexm2.csv")
errdf2=CSV.read("2000rand8Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/2),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/2),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D2000pointcompareexm2")


push!(result, (8,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/2))
push!(result, (8,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])/2))


"
10000 point 8D problem
"
errdf1=CSV.read("10000Sobol8Dexm2.csv")
errdf2=CSV.read("10000rand8Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/2),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/2),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("8D10000pointcompareexm2")


push!(result, (8,10000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/2))
push!(result, (8,10000,"Rand", mean(errdf2.absolute_error[end-1000:end])/2))


"
1000 point 16D problem
"
errdf1=CSV.read("1000Sobol16Dexm2.csv")
errdf2=CSV.read("1000rand16Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(2sqrt(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(2sqrt(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D1000pointcompareexm2")



push!(result, (16,1000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(2sqrt(2))))
push!(result, (16,1000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(2sqrt(2))))

"
2000 point 16D problem
"
errdf1=CSV.read("2000Sobol16Dexm2.csv")
errdf2=CSV.read("2000rand16Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(2sqrt(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(2sqrt(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D2000pointcompareexm2")


push!(result, (16,2000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(2sqrt(2))))
push!(result, (16,2000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(2sqrt(2))))


"
5000 point 16D problem
"
errdf1=CSV.read("5000Sobol16Dexm2.csv")
errdf2=CSV.read("5000rand16Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(2sqrt(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(2sqrt(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D5000pointcompareexm2")


push!(result, (16,5000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(2sqrt(2))))
push!(result, (16,5000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(2sqrt(2))))



"
10000 point 16D problem
"
errdf1=CSV.read("10000Sobol16Dexm2.csv")
errdf2=CSV.read("10000rand16Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(2sqrt(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(2sqrt(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D10000pointcompareexm2")


push!(result, (16,10000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(2sqrt(2))))
push!(result, (16,10000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(2sqrt(2))))



"
20000 point 16D problem
"
errdf1=CSV.read("20000Sobol16Dexm2.csv")
errdf2=CSV.read("20000rand16Dexm2.csv")
clf()
scatter(errdf1.time,log.(errdf1.absolute_error/(2sqrt(2))),0.5,label="Sobol")
scatter(errdf2.time,log.(errdf2.absolute_error/(2sqrt(2))),0.5,label="rand")
legend(loc="upper right",fancybox="true")
savefig("16D20000pointcompareexm2")


push!(result, (16,20000,"Sobol", mean(errdf1.absolute_error[end-1000:end])/(2sqrt(2))))
push!(result, (16,20000,"Rand", mean(errdf2.absolute_error[end-1000:end])/(2sqrt(2))))

clf()
CSV.write("result.csv",result)


fig = figure("2DcomparefixNBC",figsize=(20,10))
subplot(121)
suptitle("compare the mini-batch size of MC with that of QMC method to achieve the same accuracy in 2 Dimensional problems",fontsize=fontsize1)
"
compare part 2D 10000
"
errdfS250=CSV.read("250sobol2Dexm2.csv")
scatter(errdfS250.time[1:9999],log.(errdfS250.absolute_error[1:9999]),1,label="Sobol")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR10000=CSV.read("10000rand2Dexm2.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error),1,label="rand")

compareresult=DataFrame(Dimension=Int[],points=Int[],sampling_method=String[],relative_error_average=Float64[])
push!(compareresult, (2,10000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])))
push!(compareresult, (2,250,"Sobol", mean(errdfS250.absolute_error[end-1000:end])))

legend(loc="upper right",fancybox="true",scatterpoints=1000)
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of MC's mini-batch is 10000 \n while that of QMC method is 250 in each iteration",fontdict=font1)



"
compare part 2D 100000
"

subplot(1,2,2)
errdfS500=CSV.read("500sobol2Dexm2.csv")
scatter(errdfS500.time[1:9999],log.(errdfS500.absolute_error[1:9999]),1,label="Sobol")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR100000=CSV.read("100000rand2Dexm2.csv")
scatter(errdfR100000.time,log.(errdfR100000.absolute_error),1,label="rand")
legend(loc="upper right",fancybox="true",scatterpoints=1000)
ax=gca()
xlabel("iteration",fontdict=font1)
ylabel("log(error)",fontdict=font1)
setp(ax.get_xticklabels(),fontsize=fontsize1)
setp(ax.get_yticklabels(),fontsize=fontsize1)
legend(loc="upper right",fancybox="true",fontsize=fontsize1,scatterpoints=1000)
PyPlot.title("The size of MC's mini-batch is 100000 \n while that of QMC method is 500 in each iteration",fontdict=font1)
savefig("2D")
savefig("C:\\Users\\lyu\\Documents\\deep-ritz\\LaTeX_DL_468198_240419\\figure\\2D2compareNBC.eps")

clf()

push!(compareresult, (2,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])))
push!(compareresult, (2,500,"Sobol", mean(errdfS500.absolute_error[end-1000:end])))



"
compare part 4D 10000
"
errdfS500=CSV.read("500sobol4Dexm2.csv")
scatter(errdfS500.time[1:9999],log.(errdfS500.absolute_error[1:9999]/sqrt(2)),0.5,label="Sobol500")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR10000=CSV.read("10000rand4Dexm2.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error/sqrt(2)),0.5,label="rand10000")
legend(loc="upper right",fancybox="true")
savefig("4DS10000compareexm2")
clf()


push!(compareresult, (4,10000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])/sqrt(2)))
push!(compareresult, (4,500,"Sobol", mean(errdfS500.absolute_error[end-1000:end])/sqrt(2)))


"
compare part 4D 100000
"
errdfS2000=CSV.read("2000sobol4Dexm2.csv")
scatter(errdfS2000.time[1:9999],log.(errdfS2000.absolute_error[1:9999]/sqrt(2)),0.5,label="Sobol1000")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR10000=CSV.read("100000rand4Dexm2.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error/sqrt(2)),0.5,label="rand100000")
legend(loc="upper right",fancybox="true")
savefig("4DS100000compareexm2")
clf()

push!(compareresult, (4,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])/sqrt(2)))
push!(compareresult, (4,2000,"Sobol", mean(errdfS2000.absolute_error[end-1000:end])/sqrt(2)))


"
compare part 8D 10000
"
errdfS1000=CSV.read("1000sobol8Dexm2.csv")
scatter(errdfS1000.time[1:9999],log.(errdfS1000.absolute_error[1:9999]/2),0.5,label="Sobol500")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR10000=CSV.read("10000rand8Dexm2.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error/2),0.5,label="rand10000")
legend(loc="upper right",fancybox="true")
savefig("8DS10000compareexm2")
clf()


push!(compareresult, (8,10000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])/2))
push!(compareresult, (8,1000,"Sobol", mean(errdfS1000.absolute_error[end-1000:end])/2))


"
compare part 8D 100000
"
errdfS2000=CSV.read("2000sobol8Dexm2.csv")
scatter(errdfS2000.time[1:9999],log.(errdfS2000.absolute_error[1:9999]/2),0.5,label="Sobol2000")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR100000=CSV.read("100000rand8Dexm2.csv")
scatter(errdfR100000.time,log.(errdfR100000.absolute_error/2),0.5,label="rand100000")
legend(loc="upper right",fancybox="true")
savefig("8DS100000compareexm2")
clf()


push!(compareresult, (8,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])/2))
push!(compareresult, (8,2000,"Sobol", mean(errdfS2000.absolute_error[end-1000:end])/2))


"
compare part 16D 10000
"
errdfS1000=CSV.read("1000sobol16Dexm2.csv")
scatter(errdfS1000.time[1:9999],log.(errdfS1000.absolute_error[1:9999]/(2sqrt(2))),0.5,label="Sobol1000")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR10000=CSV.read("10000rand16Dexm2.csv")
scatter(errdfR10000.time,log.(errdfR10000.absolute_error/(2sqrt(2))),0.5,label="rand10000")
legend(loc="upper right",fancybox="true")
savefig("16DS10000compareexm2")
clf()

push!(compareresult, (16,100000,"Rand", mean(errdfR10000.absolute_error[end-1000:end])/(2sqrt(2))))
push!(compareresult, (16,1000,"Sobol", mean(errdfS1000.absolute_error[end-1000:end])/(2sqrt(2))))



"
compare part 16D 10000
"
errdfS5000=CSV.read("5000sobol16Dexm2.csv")
scatter(errdfS5000.time[1:9999],log.(errdfS5000.absolute_error[1:9999]/(2sqrt(2))),0.5,label="Sobol5000")
#errdfR500=CSV.read("500rand2Dexm1.csv")
#scatter(errdfR500.time,log.(errdfR500.absolute_error),0.5,label="rand500")
errdfR100000=CSV.read("100000rand16Dexm2.csv")
scatter(errdfR100000.time,log.(errdfR100000.absolute_error/(2sqrt(2))),0.5,label="rand100000")
legend(loc="upper right",fancybox="true")
savefig("16DS100000compareexm2")
clf()

push!(compareresult, (16,100000,"Rand", mean(errdfR100000.absolute_error[end-1000:end])/(2sqrt(2))))
push!(compareresult, (16,5000,"Sobol", mean(errdfS5000.absolute_error[end-1000:end])/(2sqrt(2))))
CSV.write("compareresult.csv",compareresult)
