"""
   Solve linear least square problem by LSQR method, the method solve
   min ||Ax-b||₂² if damp = 0 or
   min ||Ax-b||₂² + damp² ||x||₂² otherwise;
   A can be an explicit matrix or a function(linear operator)
   A(x, 1; params...) forward operator
   A(x, 2; params...) adjoint operator; the input and output are standard vector
   atol: the stopping tolerance for the residue norm
   conlim: the limite of the estimated condition number
   params is a dictionary contain the parameters for forward modeling operator

   x: final solution
   rnorm: sqrt(||Ax-b||₂² + damp² ||x||₂²)
   Anorm: the Frobenous norm of A
   Acond: the estimated condition number of A
   diagInv: diag(Dk * Dk'), where Dk is n*k matrix of search directions after k iterations,
          it satisfy Dk' (A'A + damp²* I) Dk = I, so the var is the estimated diagonals
          of the inverse matrix.
"""
function LSQR(A, b::Vector{Tv}; params=[], damp=0.0,
              atol=1e-9, btol=1e-9, conlim=1e8, maxIter=100,
              print_flag=false, diagInv_flag=false) where {Tv<:Number}

    # determine whether A is a matrix or operator
    if isa(A, Matrix)
       (m, n) = size(A)
       explicitA = true
    elseif isa(A, Function)
       m = length(b)
       explicitA = false
    else
       error("A either to be a matrix or operator")
    end

    msg = ["the exact solution is x = 0"
           "Ax-b is small enough given atol and btol"
           "the estimated cond(Abar) has exceeded conlim"
           "the max iterations has been reached"]

    # initialize some intermediate variables
    itn   = 0     # count number of iterations
    istop = 0     # the indicater for stop reason
    ctol  = conlim > 0 ? 1.0/conlim : 0.0
    Anorm = 0.0   # the estimated norm of vcat(A, damp I)
    Acond = 0.0   # the estimated condition number of vcat(A, damp I)
    xnorm = 0.0   # the norm of computed x
    dampsq = damp * damp;
    ddnorm = 0.0; res2 = 0.0; xxnorm = 0.0;
    z      = 0.0; cs2  =-1.0; sn2    = 0.0;

    # set up the first vectors for lanczos bidiagonalization
    # beta * u = b; alpha * v = A' * u
    u    = copy(b)
    beta = norm(u)
    alpha= 0.0

    if beta > 0.0
       u = u / beta
       if explicitA
          v = A' * u
       else
          v = A(u, 2; params...)
       end
       n = length(v)
       x = zeros(eltype(b), n)
       alpha = norm(v)
    end

    if alpha > 0.0
       v = v / alpha
       w = copy(v)
    end

    # estimate the diagonal element of inverse (A'A + dmap^2 I)^(-1)
    if diagInv_flag
       diagInv = zeros(eltype(b), n)
    end

    Arnorm = alpha * beta
    if Arnorm == 0
       println(msg[1])
       return
    end

    rhobar = alpha; phibar = beta ; bnorm  = beta ;
    rnorm  = beta ; r1norm = rnorm; r2norm = rnorm;

    # some header for print info
    test1 = 1.0; test2 = alpha / beta;
    header = "Itn               r1norm               r2norm               norm A               cond A"
    if print_flag
       println("")
       println(header)
       @printf("%3g %20.10e %20.10e %20.10e %20.10e\n", itn, r1norm, r2norm, test1, test2)
    end

     while itn < maxIter

           itn = itn + 1

           # do next step lanczos bidiagonalization
           if explicitA
              u = A * v              - alpha * u
           else
              u = A(v, 1; params...) - alpha * u
           end
           beta = norm(u)

           if beta > 0.0
              u = u / beta
              Anorm = norm([Anorm; alpha; beta; damp])
              if explicitA
                 v = A' * u             - beta * v
              else
                 v = A(u, 2; params...) - beta * v
              end

              alpha = norm(v)
              if alpha > 0.0
                 v = v / alpha
              end
           end

           # plane rotation to eliminate damping parameter
           rhobar1 =   norm([rhobar; damp])
           cs1     =   rhobar  / rhobar1
           sn1     =   damp    / rhobar1
           psi     =   sn1     * phibar
           phibar  =   cs1     * phibar

           # plane rotation
           rho     =   norm([rhobar1; beta])
           cs      =   rhobar1 / rho
           sn      =   beta    / rho
           theta   =   sn      * alpha
           rhobar  = - cs      * alpha
           phi     =   cs      * phibar
           phibar  =   sn      * phibar
           tau     =   sn      * phi

           # update x and w
           t1      =   phi     / rho
           t2      = - theta   / rho
           dk      =   w       / rho
           x       =   x +  t1 * w
           w       =   v +  t2 * w
           ddnorm  =   ddnorm  + norm(dk)^2

           # estimate the diagonal element of Inverse matrix
           if diagInv_flag
              diagInv = diagInv + dk .* dk
           end

           # plane rotation
           delta  =   sn2    * rho
           gambar = - cs2    * rho
           rhs    =   phi    - delta * z
           zbar   =   rhs    / gambar
           xnorm  =   sqrt(xxnorm + zbar^2)
           gamma  =   norm([gambar; theta])
           cs2    =   gambar / gamma
           sn2    =   theta  / gamma
           z      =   rhs    / gamma
           xxnorm =   xxnorm + z^2

           # test for convergence
           Acond  = Anorm * sqrt(ddnorm)
           res1   = phibar^2
           res2   = res2 + psi^2
           rnorm  = sqrt(res1 + res2)
           Arnorm = alpha * abs(tau)

           r1sq   = rnorm^2 - dampsq * xxnorm
           r1norm = sqrt(abs(r1sq)); r1sq = r1sq > 0.0 ? r1sq : -r1sq;
           r2norm = rnorm

           test1  = rnorm  / bnorm
           test2  = Arnorm / (Anorm * rnorm)
           test3  = 1.     / Acond
           t1     = test1  / (1   + Anorm * xnorm / bnorm)
           rtol   = btol   + atol * Anorm * xnorm / bnorm

           istop  = itn        >= maxIter ? 7 : 0
           istop  = 1. + test3 <= 1.      ? 6 : 0
           istop  = 1. + test2 <= 1.      ? 5 : 0
           istop  = 1. + t1    <= 1.      ? 4 : 0
           istop  = test3      <= ctol    ? 3 : 0
           istop  = test2      <= atol    ? 2 : 0
           istop  = test1      <= rtol    ? 1 : 0

           if istop > 0
              break
           end

           if print_flag
              @printf("%3g %20.10e %20.10e %20.10e %20.10e\n", itn, r1norm, r2norm, test1, test2)
           end

     end

     if diagInv_flag
        optout = Dict(:r1norm=>r1norm, :r2norm=>r2norm,
                      :Anorm=>Anorm  , :Acond=>Acond,
                      :Arnorm=>Arnorm, :diagInv=>diagInv)
     else
        optout = Dict(:r1norm=>r1norm, :r2norm=>r2norm,
                      :Anorm=>Anorm  , :Acond=>Acond,
                      :Arnorm=>Arnorm                 )
     end

     return x, optout
end

# the LSQR method for LSRTM
function LSQR(A::Function, b::Ts; params=[], path="NULL", damp=0.0,
              atol=1e-9, btol=1e-9, conlim=1e8, maxIter=100,
              print_flag=false, save_process_flag=false, diagInv_flag=false) where {Ts<:Union{String, Vector{String}}}

    # set the work space
    if path == "NULL"
       path = pwd()
    end

    # number of observation files
    if typeof(b) == String
       num_file = 1
       u = join([path "/u.bin"])
       q = join([path "/q.bin"])

    elseif typeof(b) == Vector{String}
       num_file = length(b)
       path_tmp = joinpath(path, "mu")
       if !isdir(path_tmp)
          mkdir(path_tmp)
          if !isdir(path_tmp)
             error("could not create directory")
          end
       end
       u = Vector{String}(num_file)
       q = Vector{String}(num_file)
       for i = 1 : num_file
           u[i] = join([path_tmp "/u" "_" "$i" ".bin"])
           q[i] = join([path_tmp "/q" "_" "$i" ".bin"])
       end
    end

    v = join([path "/v.bin"])
    x = join([path "/x.bin"])
    w = join([path "/w.bin"])
    p = join([path "/p.bin"])
    dk= join([path "/dk.bin"])

    msg = ["the exact solution is x = 0"
           "Ax-b is small enough given atol and btol"
           "the estimated cond(Abar) has exceeded conlim"
           "the max iterations has been reached"]

    # initialize some intermediate variables
    itn   = 0     # count number of iterations
    istop = 0     # the indicater for stop reason
    ctol  = conlim > 0 ? 1.0/conlim : 0.0
    Anorm = 0.0   # the estimated norm of vcat(A, damp I)
    Acond = 0.0   # the estimated condition number of vcat(A, damp I)
    xnorm = 0.0   # the norm of computed x
    dampsq = damp * damp
    ddnorm = 0.0; res2 = 0.0; xxnorm = 0.0;
    z      = 0.0; cs2  =-1.0; sn2    = 0.0;

    # set up the first vectors for lanczos bidiagonalization
    # beta * u = b; alpha * v = A' * u
    cp(b, u, remove_destination=true)
    beta = norm(u)
    alpha= 0.0

    if beta > 0.0
       scaling!(u, beta)

       if typeof(params) <: Dict{Symbol, Any}
          params[:path_m] = v
       elseif typeof(params) == Vector{Dict}
         for i = 1 : length(params)
             params[i][:path_m] = v
         end
       end
       v = A(u, 2; params=params)

       # x = zeros(v)
       (h, d) = read_USdata(v)
       write_USdata(x, h, zeros(d))
       alpha = norm(v)
    end

    if alpha > 0.0
       scaling!(v, alpha)
       cp(v, w, remove_destination=true)
    end

    # estimate the diagonal element of inverse (A'A + dmap^2 I)^(-1)
    if diagInv_flag
       diagInv = join([path "/diagInv.bin"])
       (h, d) = read_USdata(v)
       write_USdata(diagInv, h, zeros(d))
    end

    Arnorm = alpha * beta
    if Arnorm == 0
       println(msg[1])
       return
    end

    rhobar = alpha; phibar = beta ; bnorm  = beta ;
    rnorm  = beta ; r1norm = rnorm; r2norm = rnorm;

    # some header for print info
    test1 = 1.0; test2 = alpha / beta;
    header = "Itn               r1norm               r2norm               norm A               cond A"
    if print_flag
       println("")
       println(header)
       @printf("%3g %20.10e %20.10e %20.10e %20.10e\n", itn, r1norm, r2norm, test1, test2)
    end

     while itn < maxIter

           itn = itn + 1

           # do next step lanczos bidiagonalization; forward
           if typeof(params) <: Dict{Symbol, Any}
              params[:path_fwd] = q
           elseif typeof(params) == Vector{Dict}
              for i = 1 : length(params)
                  params[i][:path_fwd] = q[i]
              end
           end
           q = A(v, 1; params=params)
           x_plus_alpha_q!(q, -alpha, u; iflag=2)
           beta = norm(u)

           if beta > 0.0
              scaling!(u, beta)
              Anorm = norm([Anorm; alpha; beta; damp])

              # adjoint
              if typeof(params) <: Dict{Symbol, Any}
                 params[:path_m] = p
              elseif typeof(params) == Vector{Dict}
                 for i = 1 : length(params)
                     params[i][:path_m] = p
                 end
              end
              p = A(u, 2; params=params)
              x_plus_alpha_q!(p, -beta, v; iflag=2)

              alpha = norm(v)
              if alpha > 0.0
                 scaling!(v, alpha)
              end
           end

           # plane rotation to eliminate damping parameter
           rhobar1 =   norm([rhobar; damp])
           cs1     =   rhobar  / rhobar1
           sn1     =   damp    / rhobar1
           psi     =   sn1     * phibar
           phibar  =   cs1     * phibar

           # plane rotation
           rho     =   norm([rhobar1; beta])
           cs      =   rhobar1 / rho
           sn      =   beta    / rho
           theta   =   sn      * alpha
           rhobar  = - cs      * alpha
           phi     =   cs      * phibar
           phibar  =   sn      * phibar
           tau     =   sn      * phi

           # update x and w
           t1      =   phi     / rho
           t2      = - theta   / rho
           scaling!(dk, w, rho)
           x_plus_alpha_q!(x, t1, w; iflag=1)
           x_plus_alpha_q!(v, t2, w; iflag=2)
           ddnorm  =   ddnorm  + norm(dk)^2

           # save the result of each iteration
           if save_process_flag
              path_process = join([path "/iter" "_" "$itn" ".bin"])
              cp(x, path_process, remove_destination=true)
           end

           # estimate the diagonal element of Inverse matrix
           if diagInv_flag
              diagInv = diagInv + dk .* dk
           end

           # plane rotation
           delta  =   sn2    * rho
           gambar = - cs2    * rho
           rhs    =   phi    - delta * z
           zbar   =   rhs    / gambar
           xnorm  =   sqrt(xxnorm + zbar^2)
           gamma  =   norm([gambar; theta])
           cs2    =   gambar / gamma
           sn2    =   theta  / gamma
           z      =   rhs    / gamma
           xxnorm =   xxnorm + z^2

           # test for convergence
           Acond  = Anorm * sqrt(ddnorm)
           res1   = phibar^2
           res2   = res2 + psi^2
           rnorm  = sqrt(res1 + res2)
           Arnorm = alpha * abs(tau)

           r1sq   = rnorm^2 - dampsq * xxnorm
           r1norm = sqrt(abs(r1sq)); r1sq = r1sq > 0.0 ? r1sq : -r1sq;
           r2norm = rnorm

           test1  = rnorm  / bnorm
           test2  = Arnorm / (Anorm * rnorm)
           test3  = 1.     / Acond
           t1     = test1  / (1   + Anorm * xnorm / bnorm)
           rtol   = btol   + atol * Anorm * xnorm / bnorm

           istop  = itn        >= maxIter ? 7 : 0
           istop  = 1. + test3 <= 1.      ? 6 : 0
           istop  = 1. + test2 <= 1.      ? 5 : 0
           istop  = 1. + t1    <= 1.      ? 4 : 0
           istop  = test3      <= ctol    ? 3 : 0
           istop  = test2      <= atol    ? 2 : 0
           istop  = test1      <= rtol    ? 1 : 0

           if istop > 0
              break
           end

           if print_flag
              @printf("%3g %20.10e %20.10e %20.10e %20.10e\n", itn, r1norm, r2norm, test1, test2)
           end

     end

     if diagInv_flag
        optout = Dict(:r1norm=>r1norm, :r2norm=>r2norm,
                      :Anorm=>Anorm  , :Acond=>Acond,
                      :Arnorm=>Arnorm, :diagInv=>diagInv)
     else
        optout = Dict(:r1norm=>r1norm, :r2norm=>r2norm,
                      :Anorm=>Anorm  , :Acond=>Acond,
                      :Arnorm=>Arnorm                 )
     end

     return x, optout
end

# ==========================support functions for CGLS==========================
function cp(ps::Ts, pd::Ts; remove_destination=false) where {Ts<:Vector{String}}
    if length(ps) != length(pd)
       error("number of file does not match")
    end
    for i = 1 : length(ps)
        cp(ps[i], pd[i], remove_destination=remove_destination)
    end
    return nothing
end

function norm(s::Ts) where {Ts<:Union{String, Vector{String}}}

    if typeof(s) == String
       (hdr, d) = read_USdata(s)
       return vecnorm(d)
    elseif typeof(s) == Vector{String}
       r = 0.0
       for i = 1 : length(s)
           (hdr, d) = read_USdata(s[i])
           tmp = vecnorm(d)
           r = r + tmp * tmp
       end
       return sqrt(r)
    end
end

function x_plus_alpha_q!(x::Ts, alpha::Tv, q::Ts; iflag=1) where {Ts<:Union{String, Vector{String}}, Tv<:AbstractFloat}

    if typeof(x) == String
       (hdr_x, dx) = read_USdata(x)
       (hdr_q, dq) = read_USdata(q)
       for i = 1 : length(dx)
           dx[i] = dx[i] + alpha * dq[i]
       end

       if iflag == 1
          write_USdata(x, hdr_x, dx)
       elseif iflag == 2
          write_USdata(q, hdr_x, dx)
       end

    elseif typeof(x) == Vector{String}
       if length(x) != length(q)
          error("number of file does not match")
       end
       for i = 1 : length(x)
           (hdr_x, dx) = read_USdata(x[i])
           (hdr_q, dq) = read_USdata(q[i])
           for j = 1 : length(dx)
               dx[j] = dx[j] + alpha * dq[j]
           end

           if iflag == 1
              write_USdata(x[i], hdr_x, dx)
           elseif iflag == 2
              write_USdata(q[i], hdr_x, dx)
           end
       end
    end

    return nothing
end

function scaling!(u::Ts, beta::Tv) where {Ts<:Union{String, Vector{String}}, Tv<:AbstractFloat}

    beta_inv = 1.0 / beta

    if typeof(u) == String
       (hdr, d) = read_USdata(u)
       for i = 1 : length(d)
           d[i] = d[i] * beta_inv
       end
       write_USdata(u, hdr, d)

    elseif typeof(u) == Vector{String}

       for i = 1 : length(u)
           (hdr, d) = read_USdata(u[i])

           for j = 1 : length(d)
               d[j] = d[j] * beta_inv
           end

           write_USdata(u[i], hdr, d)
       end
    end
    return nothing
end

function scaling!(a::Ts, u::Ts, beta::Tv) where {Ts<:Union{String, Vector{String}}, Tv<:AbstractFloat}

    beta_inv = 1.0 / beta

    if typeof(u) == String
       (hdr, d) = read_USdata(u)
       for i = 1 : length(d)
           d[i] = d[i] * beta_inv
       end
       write_USdata(a, hdr, d)

    elseif typeof(u) == Vector{String}

       for i = 1 : length(u)
           (hdr, d) = read_USdata(u[i])

           for j = 1 : length(d)
               d[j] = d[j] * beta_inv
           end

           write_USdata(a[i], hdr, d)
       end
    end
    return nothing
end
