function power(maxit::Int64, fidMtx::FidMtx, w::Array{Float64,1}, irz::Array{Int64,1}, irx::Array{Int64,1}, tmax::Float64)
    nz = fidMtx.nz
    nx = fidMtx.nx
    x = randn(nz, nx)
    lambda = 0.0
    for k = 1 : maxit
        b = MultiStepForward(irz, irx, w, x, fidMtx, tmax=tmax)
        y = MultiStepAdjoint(w, b, fidMtx)
        n = norm(vec(x))
        x = y / n
        lambda = n
        println("iteration: $k, maximum eig: $lambda")
    end
    return lambda
end


function fista(shot::Shot, w::Array{Float64,1}, mu::Float64, lambda::Float64, fidMtx::FidMtx, irz::Array{Int64,1}, irx::Array{Int64,1}, tmax::Float64, maxit::Int64)
    nz = fidMtx.nz
    nx = fidMtx.nx
    x = zeros(nz, nx)
    T = mu / (2*lambda)
    t = 1
    yk = copy(x)
    for k = 1 : maxit
        tmpx = copy(x)
        bshot = MultiStepForward(irz, irx, w, yk, fidMtx, tmax=tmax)
        bshot.p = bshot.p - shot.p
        x = MultiStepAdjoint(w, bshot, fidMtx)
        x = yk - x/lambda
        x = vec(x)
        x = softThresh(x, T)
        x = reshape(x, nz, nx)
        tmpt = t
        t = (1 + sqrt(1+4*t^2)) / 2
        yk = x + (tmpt-1)/t * (x-tmpx)
        println("iteration $k")
    end
    return x
end


function softThresh(x::Array{Float64,1}, t::Float64)
    tmp = abs(x) - t
    tmp = (tmp + abs(tmp)) / 2
    y   = sign(x) .* tmp
    return y
end
