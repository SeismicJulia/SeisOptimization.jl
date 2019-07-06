"""
   Lanczos bidiagonalization with partial reorthogonalization, A can be matrix or
forward operator. k is the steps. r0 is the starting vector. the output U is a m*k
matrix, Bk is a k*k bidiagonal matrix and V is a n*k matrix and the residue vector
r0 make A*V_{k} = U_{k} * B_{k} + u_{k+1} * bet_{k+1] * e_{k}' and r0 = u_{k+1} * beta[k+1]

the call of the functon is
(U, Bk, V, r0, ierr, workload) = lanbpro(A, k, r0)

work =[numReoU, numIptU; numReoV, numIptV]

numReoU: number of orthgonalization for U
numIptU: number of inner product for U
numReoV: number of orthgonalization for V
numIptV: number of inner product for V

"""
function lanbpro(A, k::Ti; m=[], n=[], params=[],
                 delta=[], Anorm=[], Uold=[], Bold=[], Vold=[], r0=[],
                 examine_flag=false, elr_flag=true) where {Ti<:Int64}

    # A can be a function or matrix
    matrix_flag = isa(A, Matrix)

    if matrix_flag
       etype = eltype(A)
       (m, n) = size(A)
    else
       if m == [] || n==[]
          error("must specify the size of operator")
       end
       etype = Float64
    end

    # initialize some parameters
    smleps = eps(etype)

    # relaxiation coefficient for the inner product after reorthogonalization
    m2 = etype(3./2.)
    n2 = etype(3./2.)

    # desired level of orthogonaloty (the defult value)
    delta = delta == [] ? sqrt(smleps/k) : delta

    # critia for selecting subset of vectors
    eta   = etype(smleps^(3./4.) / sqrt(k))

    # the estimated error of matrix multiplication
    eps1  = etype(sqrt(maximum([m n])) * eps(etype) / 2.0)

    # force orthogonalization at every step (still select subset of vectors to orthogonal aginst)
    forceRO_flag = delta == 0 ? true : false

    # parameter for Gram-schmit method
    gamma   = etype(1.0 / sqrt(2.0))
    gs_iflag = 2 # modified iterated Gram-schmidt

    # initialize the estimate of the maximum of the eigenvalue of A
    if Anorm == []
       Anorm         = zero(etype)
       estAnorm_flag = true

       # relaxiation coefficient for the estimation of ||A||_2
       fudge         = etype(1.01)

       if Bold != []
          error("must provide Anorm when Bold is not empty")
       end
    end

    # the initial input
    if r0 != []
       p = copy(r0)
    else
       p = rand(etype, m) - etype(0.5)
    end

    # initialize output
    U = zeros(etype, m, k)
    B = zeros(etype, k, k)
    V = zeros(etype, n, k)
    beta  = zeros(etype, k+1)
    alpha = zeros(etype, k  )
    indexSet = []
    ierr     = 0

    # start with the first vector, no existed decompositions
    if Uold == [] && Bold == [] && Vold == []
       beta[1] = norm(p)

       iptU = zeros(etype, k+1); iptU[1] = one(etype);
       iptV = zeros(etype, k  ); iptV[1] = one(etype);

       iptUmax = zeros(etype, k)
       iptVmax = zeros(etype, k)

       # count the number of inner product
       numReoU = 0; numIptU = 0;
       numReoV = 0; numIptV = 0;

       # the flag to remove coupling
       coupling_flag = false

       j0 = 1

    elseif Uold != [] && Bold != [] && Vold != []

       # input with the existing factorizations
       j0 = size(Uold, 2)
       U[:, 1:j0] = Uold
       V[:, 1:j0] = Vold

       alpha[1:j0]    = diag(Bold)
       if j0 > 1
          beta[2:j0]  = diag(Bold, -1)
       end
       beta[j0+1] = norm(p)

       # check for convergence
       if j0 < k && beta[j0+1] * delta < Anorm * smleps
          forceRO_flag = true
          ierr         = j0
       end
       indexSet = collect(1:j0)
       (p, beta[j0+1], rr) = reorthogonalization(U, p, beta[j0+1], indexSet,
                                                 gamma=gamma, iflag=gs_flag)

       # statistic
       numReoU = 1; numIptU = rr * j0;
       numReoV = 0; numIptV = 0;

       # apply the same orthogonalization to V
       coupling_flag = true

       if estAnorm_flag
          Anorm = fudge * sqrt(norm(B'*B, 1) )
       end

       iptU = m2 * smleps * ones(k+1)
       iptV = zeros(etype, k)

       iptUmax = zeros(etype, k)
       iptVmax = zeros(etype, k)

       j0 = j0 + 1
    end

    # examine the accuracy of the estimated inner product
    # allocate some space for the true inner product and keep the history
    if examine_flag
       iptU_his   = zeros(etype, k, k); iptV_his   = zeros(etype, k, k);
       iptU_true  = zeros(etype, k, k); iptV_true  = zeros(etype, k, k);
       iptU_after = zeros(etype, k, k); iptV_after = zeros(etype, k, k);
    end

    # ==========================================================================
    #               start the main loop of lanczos iteration
    # ==========================================================================
    j = j0
    for j = j0 : k

        # each iteration start with assign U vectors
        if beta[j] != 0.0
           U[:,j]   = p / beta[j]
        else
           # restarted with a new vector
           U[:,j]   = p
        end

        # replace the estimated Anorm with the largest Ritz value
        # and the l2 norm of A is converged
        if j == 6
           Btmp = vcat(diagm(alpha[1:j-1])+diagm(beta[2:j-1],-1), zeros(1, j-1))
           Btmp[j,j-1] = beta[j]

           # last time update the estimate of the norm of A
           Anorm = fudge * norm(Btmp, 2)
           estAnorm_flag = false
        end

        # lanczos step to generate v_j
        if j == 1

           # for the first vector
           if matrix_flag
              r = A' * U[:,1]
           else
              r = A(U[:,1], 2; params...)
           end

           alpha[1] = norm(r)
           if estAnorm_flag
              Anorm = fudge * alpha[1]
           end

        else

           # from second V vector to the last one
           if matrix_flag
              r = A' * U[:,j] - beta[j] * V[:,j-1]
           else
              r = A(U[:,j], 2; params...) - beta[j] * V[:,j-1]
           end

           alpha[j] = norm(r)

           # extended local reorthogonalize
           if alpha[j] < gamma * beta[j] && elr_flag && !forceRO_flag

              normold = alpha[j]
              condition = true
              while condition

                    t = dot(V[:,j-1], r)
                    r = r - V[:,j-1] * t
                    beta[j] = beta[j] + t

                    alpha[j] = norm(r)

                    if alpha[j] < gamma * normold
                       normold = alpha[j]
                    else
                       condition = false
                    end
                    # do I need to count this inner product?
                    # numIptV = numIptV + 1
                    # numReoV = numReoV + 1
              end
           end

           #  update the estimate of the inner product of V
           if estAnorm_flag
              if j == 2
                 tmp = fudge * sqrt(alpha[  1]^2 + beta[2]^2 + alpha[  2]*beta[  2])
              else
                 tmp = fudge * sqrt(alpha[j-1]^2 + beta[j]^2 + alpha[j-1]*beta[j-1] + alpha[j]*beta[j])
              end
              Anorm = maximum([Anorm, tmp])
           end

           # update the estimate of the inner product of V
           # only estimate when full RO is off
           if !forceRO_flag && alpha[j] != 0.0
              iptV       = updateIptV(iptV, iptU, j, alpha, beta, Anorm, eps1)
              iptVmax[j] = maximum(abs.(iptV[1:j-1]))
           end

           if examine_flag && j > 1
              iptV_his[ 1:j-1, j-1] = iptV[1:j-1]
              iptV_true[1:j-1, j-1] = V[: ,1:j-1]' * (r/alpha[j])
           end

           if elr_flag
              iptV[j-1] = n2 * smleps
           end

           # if the estimated orthogonality is worse than delta
           # select the set of vectors to Reorthogonalize against
           if (forceRO_flag || iptVmax[j] > delta || coupling_flag) && alpha[j] > 0.0

              #  force full orthogonalize or η == 0
              if forceRO_flag || eta == 0.
                 indexSet = collect(1:j-1)
              elseif !coupling_flag
                 indexSet = selectVectors(iptV, j-1, delta, eta, 0, 0, 0)
              end

              # Reorthogonalize against
              (r, alpha[j], rr) = reorthogonalization(V, r, alpha[j], indexSet, gamma, gs_iflag)

              numIptV = numIptV + rr * length(indexSet)
              numReoV = numReoV + 1

              iptV[indexSet] = n2 * eps(etype)

              coupling_flag = coupling_flag ? false : true

           end

        end

        # Check for convergence or failure to maintain semiorthogonality
        if alpha[j] < maximum([m,n]) * Anorm * smleps && j < k
           # we deflate by setting it to 0 and attempt to restart with a basis by
           # replacing r with a random starting vector
           alpha[j] = 0
           bailout  = true

           # try three times
           for attempt = 1 : 3

               r = rand(etype, m) - etype(0.5)

               if matrix_flag
                  r = A' * r
               else
                  r = A(r, 2; params...)
               end

               normold = norm(r)

               indexSet = collect(1:j-1)
               (r, normnew, rr) = reorthogonalization(V, r, normold, indexSet, gamma, gs_iflag)

               numIptV = numIptV + rr * length(indexSet)
               numReoV = numReoV + 1

               iptV[indexSet] = n2 * smleps

               if normnew > 0
                  # a new vector is find
                  bailout = false
                  break
               end

            end

            # converged
            if bailout
               j = j-1
               ierr = -j
               break
            else
               r = r / normnew
               coupling_flag = true

               # turn-off forced full reorthogonalize if delta > 0.0
               if delta > 0.0
                  forceRO_flag = false
               end
            end

        elseif j < k && !forceRO_flag && Anorm * smleps > delta*alpha[j]

            ierr = j

        end  # end of the bailout

        if alpha[j] != 0.0
           V[:,j] = r / alpha[j]
        else
           V[:,j] = r
        end

        # kepp the history
        if j > 1 && examine_flag
           iptV_after[1:j-1, j-1] = iptV[1:j-1]
           iptV_true[ 1:j-1, j-1] = V[:,1:j-1]' * r / alpha[j]
        end

        # ======================================================================
        #                                 U[:,j+1]
        # ======================================================================
        if matrix_flag
           p = A * V[:,j] - alpha[j] * U[:,j]
        else
           p = A(V[:,j], 1; params...) - alpha[j] * U[:,j]
        end
        beta[j+1] = norm(p)

        # extended local orthogonalization
        if beta[j+1] < gamma*alpha[j] && elr_flag && !forceRO_flag
           normold = beta[j+1]
           condition = true
           while condition
                 t = dot(U[:,j], p)
                 p = p - U[:,j]*t
                 beta[j+1] = norm(p)
                 alpha[j]  = alpha[j] + t

                 if beta[j+1] < gamma * normold
                    normold = beta[j+1]
                 else
                    condition = false
                 end
           end
        end

        # update the estimate of largest singular value of A
        if estAnorm_flag

           if j == 1
              tmp = fudge * sqrt(alpha[1]^2 + beta[2  ]^2)
           else
              tmp = fudge * sqrt(alpha[j]^2 + beta[j+1]^2 + alpha[j]*beta[j])
           end
           Anorm = maximum([Anorm, tmp])
        end

        # update the estimate of the orthogonality for the columns of U
        if !forceRO_flag && beta[j+1] != 0
           iptU = updateIptU(iptU, iptV, j, alpha, beta, Anorm, eps1)
           iptUmax[j] = maximum(abs.(iptU[1:j]))
        end

        # keep all the history
        if examine_flag
           # keep the history of estimated inner product
           iptU_his[ 1:j,j] = iptU[1:j]

           # keep the history of true inner product
           tmp = p / beta[j+1]
           iptU_true[1:j,j] = U[:,1:j]' * tmp
        end

        if elr_flag
           iptU[j] = m2 * smleps
        end

        # if the maximum of iptU is worse than delta, then orthogonalize U[:,]+1]
        # against subset of U[:,i] with 1 <= i <=j
        # priority forceRO > coupling > iptUmax
        if (forceRO_flag || iptUmax[j] > delta || coupling_flag) && beta[j+1] > 0.0

           # full orthogonalize
           if forceRO_flag || eta == 0
              indexSet = collect(1:j)

           # select subset which iptU[i] >= eta
           elseif !coupling_flag
              indexSet = selectVectors(iptU, j, delta, eta, 0, 0, 0)

           # aviod propagation from V to U
           else
              indexSet = vcat(indexSet, maximum(indexSet)+1)
           end

           (p, beta[j+1], rr) = reorthogonalization(U, p, beta[j+1], indexSet, gamma, gs_iflag)

           numIptU = numIptU + rr * length(indexSet)
           numReoU = numReoU + 1

           iptU[indexSet] = m2 * smleps

           # if this orthogonalization is to aviod coupling,
           # then next time doesn't need orth
           coupling_flag = coupling_flag ? false : true

        end

        # check for convergence or failure to maintain,
        # converged before j ==k
        if beta[j+1] < maximum([m,n]) * Anorm * smleps && j < k

           #  restart it with new random vector
           beta[j+1] = zero(etype)
           bailout   = true

           for attempt = 1 : 3
               p = rand(etype, n) - etype(0.5)
               if matrix_flag
                  p = A * p
               else
                  p = A(p, 1; params...)
               end

               normold = norm(p)
               indexSet = collect(1:j)
               (p, normnew, rr) = reorthogonalization(U, p, normold, indexSet, gamma, gs_iflag)

               numIptU = numIptU + rr * length(indexSet)
               numReoU = numReoU + 1

               iptU[indexSet] = m2 * smleps

               if normnew > 0.0
                  bailout = false
                  break
               end

           end

           # indicate the step error happens
           if bailout
              ierr = -j
              break
           else

              p = p / normnew
              coupling_flag = true

              if delta > 0.0
                 forceRO_flag = false
              end
           end

        elseif j<k && !forceRO_flag && Anorm * smleps > delta * beta[j+1]

           ierr = j

        end # end of convergence check

        # keep the history of the iteration
        if examine_flag
           iptU_after[1:j, j] = iptU[1:j]
           iptU_true[ 1:j, j] = U[:,1:j]' * (p/beta[j+1])
        end

    end # end of lanczos iteration

    # finish the iterations
    # incase the iteration finished before kth one
    k = j<k ? j : k
    Bk = spdiagm((alpha[1:k], beta[2:k+1]), (0,-1), k+1, k)
    if beta[k+1] > 0
       U = hcat(U[:,1:k], p/beta[k+1])
    else
       U = hcat(U[:,1:k], p)
    end
    V  = V[:,1:k]

    if examine_flag
       optout= Dict(:numReoU=>numReoU, :numReoV=>numReoV, :numIptU=>numIptU, :numIptV=>numIptV,
                    :iptU_his=>iptU_his, :iptU_true=>iptU_true, :iptU_after=>iptU_after,
                    :iptV_his=>iptV_his, :iptV_true=>iptV_true, :iptV_after=>iptV_after,
                    :iptUmax=>iptUmax, :iptVmax=>iptVmax)
       return U, Bk, V, optout
    else
       return U, Bk, V
    end

end

"""
   Gram-Schmidt reorthogonalization of r with respect to the subset of the columns
of Q, the subset is specified by indexSet. The columns of the matrix Q must be orthonormal.
"""
function reorthogonalization(Q::Matrix{Tv}, r::Vector{Tv}, normold::Tv,
                             index::Vector{Ti}, gamma::Tv, iflag::Ti) where {Tv<:Number, Ti<:Int64}

    (m, n)   = size(Q)
    normnew  = copy(normold)

    numReo  = 0
    re = copy(r)

    while normnew < gamma * normold || numReo == 0

       if iflag == 1

          t  = Q[:, index]' * re
          re = re - Q[:, index] * t

       else

          for i in index
              t = dot(Q[:,i], re)
              re= re - Q[:,i] * t
          end
       end

       normold = normnew
       normnew = norm(re)
       numReo  = numReo + 1

       if numReo > 4
          re = zeros(r)
          normnew = 0
          return re, normnew, numReo
       end

    end

    return re, normnew, numReo

end


"""
   update the estimate of the inner product of the left lanczos vectors recursively
   μ[i,i]    = 1 for 1<= i <= j+1  and  ν[j,0] = 0

   for 1 <= i <= j
   μ[j+1,i]' = α[i]ν[j,i] + β[i]ν[j,i-1] - α[j]μ[j,i]
   μ[j+1,i]  = ( μ[j+1,i]' + sign(μ[j+1,i]')ϵ ) / β[j+1]

   μ[j+1, j+1] = 1
"""
# function updateIptU(iptU_old::Vector{Tv}, iptV::Vector{Tv}, j::Ti,
#                     alpha::Vector{Tv}, beta::Vector{Tv}, Anorm::Tv, eps1::Tv) where {Tv<:AbstractFloat, Ti<:Int64}
#
#     etype = eltype(alpha)
#     iptU_new = zeros(iptU_old)
#
#     T = Anorm * eps1
#
#     # for the first iteration
#     if j == 1
#        # j=1, μ[2,1] = α[1]ν[1,1] + β[1]ν[1,0] - α[1]μ[1,1] = α[1] + 0 - α[1] = 0
#        iptU_new[1] = T / beta[2]
#
#     # later iterations
#     else
#
#        # j != 1 and i = 1, μ[j+1,1]' = α[1]ν[j,1] + β[i]ν[j,0] - α[j]μ[j,1] = α[1]ν[j,1] + 0 - α[j]μ[j,1]
#        iptU_new[1]     = alpha[1]*iptV[1]                     - alpha[j] * iptU_old[1]
#        iptU_new[1]     = (iptU_new[1] + sign(iptU_new[1]) * T) / beta[j+1]
#
#        # i = 2 : j-1,  μ[j+1,i]' = α[i]ν[j,i] + β[i]ν[j,i-1] - α[j]μ[j,i]
#        for k = 2 : j-1
#            iptU_new[k] = alpha[k]*iptV[k] + beta[k]*iptV[k-1] - alpha[j] * iptU_old[k]
#            iptU_new[k] = (iptU_new[k] + sign(iptU_new[k]) * T) / beta[j+1]
#        end
#
#        # i = j, μ[j+1,j]' = α[i]ν[j,j] + β[j]ν[j,j-1] - α[j]μ[j,j] = α[i] + β[i]ν[j,j-1] - α[j] = β[j]ν[j,j-1]
#        iptU_new[j]     =                    beta[j]*iptV[j-1]
#        iptU_new[j]     = (iptU_new[j] + sign(iptU_new[j]) * T) / beta[j+1]
#
#     end
#
#     # j+1
#     iptU_new[j+1] = one(etype)
#
#     return iptU_new
#
# end
function updateIptU(iptU_old::Vector{Tv}, iptV::Vector{Tv}, j::Ti,
                    alpha::Vector{Tv}, beta::Vector{Tv}, Anorm::Tv, eps1::Tv) where {Tv<:AbstractFloat, Ti<:Int64}

    etype = eltype(alpha)
    iptU_new = zeros(iptU_old)

    # for the first iteration
    if j == 1
       # j=1, μ[2,1] = α[1]ν[1,1] + β[1]ν[1,0] - α[1]μ[1,1] = α[1] + 0 - α[1] = 0
       T = eps1 * (sqrt(alpha[1]^2 + beta[2]^2) + sqrt(alpha[1]^2+beta[1]^2))
       T = T + eps1 * Anorm
       iptU_new[1] = T / beta[2]

    # later iterations
    else

       # j != 1 and i = 1, μ[j+1,1]' = α[1]ν[j,1] + β[i]ν[j,0] - α[j]μ[j,1] = α[1]ν[j,1] + 0 - α[j]μ[j,1]
       iptU_new[1]     = alpha[1]*iptV[1]                     - alpha[j] * iptU_old[1]
       T = eps1 * (sqrt(alpha[j]^2 + beta[j+1]^2) + sqrt(alpha[1]^2+beta[1]^2))
       T = T + eps1 * Anorm
       iptU_new[1]     = (iptU_new[1] + sign(iptU_new[1]) * T) / beta[j+1]

       # i = 2 : j-1,  μ[j+1,i]' = α[i]ν[j,i] + β[i]ν[j,i-1] - α[j]μ[j,i]
       for k = 2 : j-1
           iptU_new[k] = alpha[k]*iptV[k] + beta[k]*iptV[k-1] - alpha[j] * iptU_old[k]
           T = eps1 * (sqrt(alpha[j]^2 + beta[j+1]^2) + sqrt(alpha[k]^2+beta[k]^2))
           T = T + eps1 * Anorm
           iptU_new[k] = (iptU_new[k] + sign(iptU_new[k]) * T) / beta[j+1]
       end

       # i = j, μ[j+1,j]' = α[i]ν[j,j] + β[j]ν[j,j-1] - α[j]μ[j,j] = α[i] + β[i]ν[j,j-1] - α[j] = β[j]ν[j,j-1]
       T = eps1 * (sqrt(alpha[j]^2 + beta[j+1]^2) + sqrt(alpha[j]^2+beta[j]^2))
       T = T + eps1 * Anorm
       iptU_new[j]     =                    beta[j]*iptV[j-1]
       iptU_new[j]     = (iptU_new[j] + sign(iptU_new[j]) * T) / beta[j+1]

    end

    # j+1
    iptU_new[j+1] = one(etype)

    return iptU_new

end

"""
   update the estimate of the inner product of the right lanczos vectors recursively
   ν[i,i]    = 1 for 1<= i <= j  and  μ[j,0] = 0

   for 1 <= i <= j-1
   ν[j,i]' = β[i+1]μ[j,i+1] + α[i]μ[j,i] - β[j]ν[j-1,i]
   ν[j,i]  = ( ν[j,i]' + sign(ν[j,i]')ϵ ) / α[j]

   ν[j,j] = 1
"""
# function updateIptV(iptV_old::Vector{Tv}, iptU::Vector{Tv}, j::Ti,
#                      alpha::Vector{Tv}, beta::Vector{Tv}, Anorm::Tv, eps1::Tv) where {Tv<:AbstractFloat, Ti<:Int64}
#
#     etype = eltype(alpha)
#     iptV_new = zeros(iptV_old)
#
#     T     = Anorm * eps1
#
#     for k = 1 : j-1
#
#         iptV_new[k] = beta[k+1] * iptU[k+1] + alpha[k] * iptU[k] - beta[j] * iptV_old[k]
#         iptV_new[k] = (iptV_new[k] + sign(iptV_new[k]) * T) / alpha[j]
#
#     end
#
#     iptV_new[j] = one(etype)
#
#     return iptV_new
#
# end
function updateIptV(iptV_old::Vector{Tv}, iptU::Vector{Tv}, j::Ti,
                     alpha::Vector{Tv}, beta::Vector{Tv}, Anorm::Tv, eps1::Tv) where {Tv<:AbstractFloat, Ti<:Int64}

    etype = eltype(alpha)

    iptV_new = zeros(iptV_old)

    for k = 1 : j-1

        T = eps1 * (sqrt(alpha[k]^2 + beta[k+1]^2) + sqrt(alpha[j]^2 + beta[j]^2))
        T = T + eps1 * Anorm
        iptV_new[k] = beta[k+1] * iptU[k+1] + alpha[k] * iptU[k] - beta[j] * iptV_old[k]
        iptV_new[k] = (iptV_new[k] + sign(iptV_new[k]) * T) / alpha[j]

    end

    iptV_new[j] = one(etype)

    return iptV_new

end


"""
   select the sets of vectors to orthogonalize against it. Three strategy are provided
"""
function selectVectors(ipt::Vector{Tv}, j::Ti, delta::Tv, eta::Tv,
                       LL::Ti, strategy::Ti, extra::Ti) where {Tv<:AbstractFloat, Ti<:Int64}

    if delta < eta
       error("delta must be larger than eta")
    end

    # orthogonalize against the neighbours of ipt[i] > delta and the neighbours > eta
    if strategy == 0

       I0 = find(x->(abs(x) >= delta), ipt[1:j])
       if length(I0) == 0
          (maxipt, I0) = findmax(ipt)
       end

       index = zeros(Int64, j)

       for i = 1 : length(I0)

           # left neighbour
           r = I0[i]
           for r = I0[i] : -1 : 1
               if abs(ipt[r]) < eta || index[r] == 1  # belong to the neighbour of other vector
                  break
               else
                  index[r] = 1
               end
           end
           if extra != 0
              lower = maximum([1, r-extra+1])
              index[lower:r] = 1
           end

           # right neighbour
           s = I0[i]+1
           for s = I0[i]+1 : j
               if abs(ipt[s]) < eta || index[s] == 1
                  break
               else
                  index[s] = 1
               end
           end
           if extra != 0
              upper = minimum([j, s+extra-1])
              index[s:upper] = 1
           end

       end

       if LL > 0
          index[1:LL] = 0
       end
       index = find(x->(x>0), index)

    elseif strategy == 1

       I0 = find(x->(abs(x) > eta), ipt[1:j])
       lower = maximum([LL+1, minimum(I0)-extra])
       upper = minimum([j   , maximum(I0)+extra])
       if lower <= upper
          index = collect(lower:upper)
       end

    else

       index = find(x->(abs(x) > eta), ipt[1:j])

    end

    return index
end
