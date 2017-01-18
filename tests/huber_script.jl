# Sparse robust regression, using perspective duality.
#
# Kellie MacPhee
# December 13, 2016
#
# Solve the primal problem
# min     ||x||₁
# s.t.    huber(b-Ax) ≤ σ
# using Chambolle-Pock on primal and perspective dual problems.
#
# Compare: 1. CP on primal vs. perspective dual
#               a) Iterations to obtain active set (% of false positives and negatives)
#               b) Convergence to optimal objective value
#               c) Feasibility at each iteration
#
#          2. Two methods of recovering primal solutions from perspective dual
#               a) Iterations to obtain active set (false positives and negatives)
#               b) Convergence to optimal objective value
#               c) Feasibility at each iteration
#
workspace()

sasha_path = "/Users/saravkin11/Dropbox/Gauge Duality/code/julia"
kellie_path = "/Users/kelliemacphee/Dropbox/Research/Gauge Duality/code/julia"
path = sasha_path

include(string(path, "/aux/genSparseHuber.jl"))
include(string(path, "/aux/myHuber.jl"))
include(string(path, "/algorithms/cp.jl"))
include(string(path, "/aux/proj_infball.jl"))
include(string(path, "/algorithms/inf_hubPrimal.jl"))
include(string(path, "/algorithms/ProjSplx.jl"))
include(string(path, "/algorithms/inf_huber.jl"))

using Plots; pyplot()

# Set up the problem.
m = 120; n = 512
η = 1 # huber parameter
k = 20 # sparsity of true solution (number of nonzeros)
s = 5 # number of spikes/outliers in corrupted measurement b
srand(10); (A, b, x0, outliers, σ) = genSparseHuber(m, n, k, s, .1)

maxIters = 100 # how long to run CP

#
# First, run Chambolle-Pock on primal.
#

println("Running CP on primal.")

Profile.clear()
@profile (pr_iterates, pr_steps, xfinal)  = inf_hubPrimal(A, b, σ, η; maxIts = maxIters, damp=0.99, tol=1e-9, Gsteplengthen=1)
Profile.print(format = :flat)




#
# Second, run Chambolle-Pock on perspective dual,
# and recover primal iterates using two methods.
#

println("Running CP on perspective dual.")

Profile.clear()
@profile (perdual_iterates, dualdual_iterates, du_steps, yCP, βCP) = inf_huber(A, b, σ, η; maxIts=maxIters, damp=0.99, tol=1e-5, Gsteplengthen=2)
Profile.print(format = :flat)


println("Recovering x-values from CP on dual.")
(j,k) = size(dualdual_iterates)
pr_iterates_one = zeros(j,k) # all primal iterates from method one
pr_iterates_two = zeros(j,k) # all primal iterates from method two

for i=1:k
  z = dualdual_iterates[:,i]
  y = perdual_iterates[1:end-1,i]; β = perdual_iterates[end,i]

  # Recover using method one: scale Lagrange dual of perspective dual (double dual) iterates.
  val =  norm(A'*y,Inf)  # estimate optimal value of double dual (=optimal value of perspective dual)
  pr_iterates_one[:,i] = (1/val) * z

  # # Recover using method two: derived optimality conditions.
  # pr_iterates_two[:,i] = inf_hub_recover(y, β, A, b, σ, η; tol=0.01)
end

xfinal_one = pr_iterates_one[:,end] # final primal iterate from method one
# xfinal_two = pr_iterates_two[:,end] # final primal iterate from method two

#
# Plot convergence to known optimal value of primal problem,
# (normalized difference between current objective value and optimal value).
#

println("Plotting results.")
opt = norm(x0,1)

cppr_obj = transpose(mapslices(sum, abs(pr_iterates), 1)) # take absolute sum of columns in pr_iterates, reutrn col vector of obj values
cpdu_obj_one = transpose(mapslices(sum, abs(pr_iterates_one), 1))
# cpdu_obj_two = transpose(mapslices(sum, abs(pr_iterates_two), 1))

plot_obj = plot( (1/opt)*(cppr_obj - opt), line=:dash, lab="CP on primal");
plot!( (1/opt)*(cpdu_obj_one - opt), lab="CP on dual, recovery method 1")
# plot!( (1/opt)*(cpdu_obj_two - opt), lab="CP on dual, recovery method 2")
title!("Convergence of objective values to optimum")
plot!(left_margin=20px,bottom_margin=20px, right_margin=20px, top_margin=20px)
xaxis!("iteration"); yaxis!("(objective - optimum)/optimum", ylims = [-0.5,0.5])
gui(plot_obj)
savefig("huber_objconv.png")

#
# Plot feasibility of primal iterates.
#

function violates_feas(x)
  ρ = myHuber(b-A*x,η)
  cons_viol = ρ - σ

  return max(0, cons_viol)
end

cppr_feas = transpose(mapslices(violates_feas, pr_iterates, 1))
cpdu_feas_one = transpose(mapslices(violates_feas, pr_iterates_one, 1))
# cpdu_feas_two = transpose(mapslices(violates_feas, pr_iterates_two, 1))

plot_feas = plot(cppr_feas, line=:dash, lab="CP on primal");
plot!(cpdu_feas_one, lab="CP on dual, recovery method 1")
# plot!(cpdu_feas_two, lab="CP on dual, recovery method 2")
title!("Feasibility violations")
plot!(left_margin=20px,bottom_margin=20px, right_margin=20px, top_margin=20px)
xaxis!("iteration"); yaxis!("feasibility violation", ylims=[0,1])
gui(plot_feas)
savefig("huber_feasviol.png")

#
# Plot percentage of false positives and negatives for sparsity.
#

tol = 0.05 # for rounding to zero - don't want to be too sensitive

function false_zeros(x)
  counter = 0
  for i=1:length(x)
    if abs(x[i]) <= tol && x0[i] != 0
      counter += 1
    end
  end
  return counter
end

function false_nzeros(x)
  counter = 0
  for i=1:length(x)
    if abs(x[i]) > tol && x0[i] == 0
      counter += 1
    end
  end
  return counter
end

#
# Plot false zeros.
#

cppr_falsezeros= transpose(mapslices(false_zeros, pr_iterates, 1))
cpdu_falsezeros_one = transpose(mapslices(false_zeros, pr_iterates_one, 1))
# cpdu_falsezeros_two = transpose(mapslices(false_zeros, pr_iterates_two, 1))

# # as a fraction:
# plotFZ = plot( (1/(n-k)) * cppr_falsezeros, line=:dash, lab="CP on primal")
# plot!( (1/(n-k)) * cpdu_falsezeros_one, lab="CP on dual, recovery method 1")
# plot!( (1/(n-k)) * cpdu_falsezeros_two, lab="CP on dual, recovery method 2")
# title!("False zeros")
# xaxis!("iteration"); yaxis!("Fraction of false zeros")
# plot!(left_margin=20px,bottom_margin=20px, right_margin=20px, top_margin=20px)

# as a total number:
plotFZ2 = plot( cppr_falsezeros, line=:dash, lab="CP on primal")
plot!( cpdu_falsezeros_one, lab="CP on dual, recovery method 1")
# plot!( cpdu_falsezeros_two, lab="CP on dual, recovery method 2")
title!("False zeros")
xaxis!("iteration"); yaxis!("Number of false zeros", ylims=[0,5])
plot!(left_margin=20px,bottom_margin=20px, right_margin=20px, top_margin=20px)
gui(plotFZ2)
savefig("huber_fz.png")

#
# Plot false nonzeros.
#
cppr_falsenzeros= transpose(mapslices(false_nzeros, pr_iterates, 1))
cpdu_falsenzeros_one = transpose(mapslices(false_nzeros, pr_iterates_one, 1))
# cpdu_falsenzeros_two = transpose(mapslices(false_nzeros, pr_iterates_two, 1))

# # as a fraction:
# plot_FNZ = plot( (1/k) * cppr_falsenzeros, line=:dash, lab="CP on primal");
# plot!( (1/k) * cpdu_falsenzeros_one, lab="CP on dual, recovery method 1")
# # plot!( (1/k) * cpdu_falsenzeros_two, lab="CP on dual, recovery method 2")
# title!("False nonzeros")
# xaxis!("iteration"); yaxis!("Fraction of false nonzeros")
# plot!(left_margin=20px,bottom_margin=20px, right_margin=20px, top_margin=20px)
# gui(plot_FNZ)

# as a total number:
plot_FNZ2 = plot( cppr_falsenzeros, line=:dash, lab="CP on primal");
plot!( cpdu_falsenzeros_one, lab="CP on dual, recovery method 1")
# plot!( cpdu_falsenzeros_two, lab="CP on dual, recovery method 2")
title!("False nonzeros")
xaxis!("iteration"); yaxis!("Number of false nonzeros")
plot!(left_margin=20px,bottom_margin=20px, right_margin=20px, top_margin=20px)
gui(plot_FNZ2)
savefig("huber_fnz.png")
