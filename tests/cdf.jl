# Test sparse robust regression, using perspective duality,
# on a large number of examples. How many Iterations
# does it take for CP on the perspective dual, with recovery,
# to converge to an optimal solution?
#
# Written by: Kellie MacPhee
# Last updates: January 10, 2017
#
# Solve the primal problem
# min     ||x||₁
# s.t.    huber(b-Ax) ≤ σ
# using Chambolle-Pock on the perspective dual.

include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/algorithms/inf_huber.jl")
include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/aux/myHuber.jl")
include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/aux/genSparseHuber.jl")
include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/algorithms/inf_hub_recover.jl")

using Plots; pyplot()

# Set up the general type of problem.
m = 120; n = 512
η = 1 # huber parameter
k = 20 # sparsity of true solution (number of nonzeros)
s = 5 # number of spikes/outliers in corrupted measurement b

maxIters = 800
objTol = 1e-3
# feasTol = 1e-2 # can also set as a fraction of σ (inside the loop)
num_test = 500 # how many examples to run

# Function to count number of iterations for primal iterates to converge.
# (as measured by feasibilty and objective value being near optimal)
# when we run CP on the dual and then recover primal iterates.
#
# kwargs (for CP):    maxIters, damp, tol (for exiting CP), Gsteplengthen, Θ, verbose
#
function iter_to_converge(A, b, σ, η, xtrue; feas_tol=1e-8, obj_tol=1e-8, kwargs...)
  opt = norm(xtrue,1) # optimal value of primal objective

  # Run CP to solve the perspective dual.
  (perdual_its, dualdual_its, du_steps, yCP, βCP) = inf_huber(A, b, σ, η; kwargs... )

  for i=1:size(dualdual_its,2)
    # Recover primal iterate (by scaling method).
    z = dualdual_its[:,i]
    y = perdual_its[1:end-1,i]
    val =  norm(A'*y,Inf) # estimate optimal value of double dual (=optimal value of perspective dual). scale down a little bit
    x = (1/val) * z  # primal iterate

    # # Recover primal iterate (by optimality conditions) - slow!!!
    # z = dualdual_its[:,i]
    # y = perdual_its[1:end-1,i]; β = perdual_its[end,i]
    # x = inf_hub_recover(y, β, A, b, σ, η; tol=0.01)

    # Test whether this primal iterate is approximately optimal.
    feas_viol = myHuber(b-A*x,η) - σ
    obj_diff = norm(x,1) - opt
    @printf("iteration: %4i, dist to feasibility: %10.3e, dist to objective value: %10.3e\n", i, feas_viol, obj_diff)

    # If approximately optimal, return this iteration number.
    if  feas_viol <= feas_tol && obj_diff <= obj_tol
      return i
    end
  end

  # If never approximately optimal (within maxIters), return Inf.
  return Inf
end

convergence = fill(Inf, num_test) # how many iterations to converge, by probem instance

for t in 1:num_test
  # Generate problem instance.
  srand(t); (A, b, xtrue, outliers, σ) = genSparseHuber(m, n, k, s, .1)

  # Count number of iterations to converge on this problem instance.
  feasTol = 0.01*σ
  test = iter_to_converge(A, b, σ, η, xtrue; feas_tol=feasTol, obj_tol=objTol, Θ=1, damp=0.99, Gsteplengthen=0.4, maxIts=maxIters, tol=1e-5, verbose=false)
  @printf("Iterations to converge: %4i", test)
  convergence[t] = test
end

converged = deleteat!(convergence, findin(convergence, Inf))
num_converged = length(converged)
@printf("Number of trials converged: %i", num_converged)

# Plot convergence.
pdf = histogram(converged, nbins=50, leg=false)
xaxis!("iteration", xlims=[0,800]); yaxis!("number of trials converging")
title!( "Sparse robust regression for 500 problem instances")
plot!(left_margin=20px,bottom_margin=20px, right_margin=20px, top_margin=20px)
gui(pdf)
# savefig("huber_pdf.png")
