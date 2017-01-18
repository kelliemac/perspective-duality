# Applying Chambolle-Pock to sparse robust regression,
# i.e. huber constraint in primal problem.
#
# Written by: Kellie MacPhee and Michael Friedlander
# Last updated: January 10, 2017

#include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/aux/myHuber.jl")
#include("/Users/kmacphee/Dropbox/Research/Gauge Duality/code/julia/algorithms/cp.jl")
using JuMP, Gurobi, ProjSplx

# Using Gurobi, project (y₀, β₀) onto huber π∘ constraints:
# <b,y> - β - σξ ≥ 1
# -ξ ≤ y ≤ ξ,    ξ ≥ 0,    β ≥ 0
# (2√η * y, ξ - 2β, ξ + 2β) ∈ SOC
# (Notes: β = - μ,  W = [-I; I],  w = e,  L = √η*I. )
function proj_huber_pid(y₀, β₀, b, σ, η; kwargs... )

  n = length(y₀)

  m = Model(solver=GurobiSolver(OutputFlag=0, BarQCPConvTol=1e-8))
  @variable(m, y[1:n])
  @variable(m, β ≥ 0)
  @variable(m, ξ ≥ 0)

  @constraint(m, dot(b,y) - β - 1 ≥ σ*ξ) # had == instead before. better computationally?
  @constraint(m, poly[i=1:n],      y[i] ≤ +ξ )
  @constraint(m, poly[i=1:n], -ξ ≤ y[i]      )
  @constraint(m, norm([2β-ξ; (2*√η)*y ]) ≤ 2β+ξ )

  # Set up least squares objective.
  @variable(m, t≥0)
  @constraint(m, norm([y-y₀; β-β₀]) ≤ t )
  @objective(m, Min, t)

  status = solve(m)

  y = getvalue(y)
  β = getvalue(β)

  # # Verify that the new point is feasible.
  #
  # err1 = min(0, dot(b,y) - β - σ*η*dot(y,y)/(2β) - 1)
  # err2 = min(0, dot(b,y) - β - σ*norm(y,Inf) - 1)
  # err = min(err1, err2)
  #
  # #println("err = $err1, $err2")

  return y, β

end


# Apply Chambolle-Pock to the perspective dual problem
# minimize ||A'y||_∞ subj to  <b,y> - β - σ gπ∘(y,β) ≥ 1
# where g is a huber function with parameter η.
#
# Define
# x = (y, β),   K = [A' 0]
# F(Kx) = ||Kx||_∞ ≡ || [A' 0] [y; β] ||_∞
# G( x)                  ≡ δ( (y,β) | level-set)
function inf_huber( A, b, σ, η; kwargs... )

  (m, n) = size(A)

  # Provide prox of F conjugate, which is projection onto unit 1-norm ball.
  proxFc(v, σ) = projnorm1(v, 1.0)

  # Provide prox of G, which is projection onto level set.
  function proxG(w, τ)
    (y, β) = proj_huber_pid( w[1:end-1], w[end], b, σ, η; kwargs... )
    return [y; β]
  end

  # Provide linear operator.
  K = [A' zeros(n)]

  # Initialize.
  y₀ = zeros(m); β₀= 1/m # guess for a feasible point
  x₀ = [y₀; β₀]
  z₀ = zeros(n) # lagrange dual initial iterate for CP. presumably has small entries,
  #                         because of relationship between double dual optimum and primal optimum.

  # Run CP on problem and return results.
  (pr_iterates, du_iterates, steps) = chamb_pock( proxFc, proxG, K, x₀, z₀; kwargs...)
  yfinal = pr_iterates[1:end-1,end]; βfinal = pr_iterates[end,end]

  return (pr_iterates, du_iterates, steps, yfinal, βfinal)

end
