#========================== Interpolation ===========================#

# First-order interpolation weighted by areas of dual cells
primal_from_dual!(fi, fv, mesh::VoronoiSphere) =
    primal_from_dual!(fi, fv, mesh.primal_deg, mesh.Av, mesh.primal_vertex)

function primal_from_dual!(fi::AbstractVector, fv, degrees, areas, vertices)
    @fast for ij in eachindex(degrees)
        deg = degrees[ij]
        Ai = sum(areas[vertices[ij][vertex]] for vertex = 1:deg)
        fi[ij] =
            inv(Ai) *
            sum(areas[vertices[ij][vertex]] * fv[vertices[ij][vertex]] for vertex = 1:deg)
    end
    return fi
end

function primal_from_dual!(fi::AbstractMatrix, fv, degrees, areas, vertices)
    nz = size(fi, 1)
    @fast for ij in eachindex(degrees)
        deg = degrees[ij]
        inv_Ai = inv(sum(areas[vertices[vertex, ij]] for vertex = 1:deg))
        for k = 1:nz
            fi[k, ij] = 0
        end
        for vertex = 1:deg
            vv = vertices[vertex, ij]
            ww = areas[vv] * inv_Ai
            for k = 1:nz
                fi[k, ij] = muladd(ww, fv[k, vv], fi[k, ij])
            end
        end
    end
    return fi
end

primal_from_dual(fv::AbstractVector, degrees, areas, vertices) =
    primal_from_dual!(similar(degrees, eltype(fv)), fv, degrees, areas, vertices)
primal_from_dual(fv::AbstractMatrix, degrees, areas, vertices) = primal_from_dual!(
    Matrix{eltype(fv)}(undef, size(fv, 1), size(degrees, 1)),
    fv,
    degrees,
    areas,
    vertices,
)
primal_from_dual(fv, mesh::VoronoiSphere) =
    primal_from_dual(fv, mesh.primal_deg, mesh.Av, mesh.primal_vertex)

# Perot reconstruction of vector field given covariant components

function primal3D_from_cov!(
    u::T,
    v::T,
    w::T,
    ucov::T,
    degrees,
    edges,
    weights,
) where {T<:AbstractVector}
    for ij in eachindex(u, v, w, ucov, degrees)
        u[ij], v[ij], w[ij] = primal3D_from_cov!(ij, ucov, degree[ij], edges, weights)
    end
    return u, v, w
end

# x,y,z = ( coslat*coslon, coslat*sinlon, sinlat )
# d(x,y,z)/dlon  = ( -coslat*sinlon, coslat*coslon, 0 )
# d(x,y,w)/dlat  = ( -sinlat*coslon, -sinlat*coslon, coslat )
# function 'fun' is applied to (ulon,ulat), see ShallowWaters.diag_ulonlat
@inline function primal_lonlat_from_cov!(
    fun::Fun,
    ulon::T,
    ulat::T,
    ucov::T,
    degrees,
    edges,
    weights,
    coslon,
    sinlon,
    coslat,
    sinlat,
) where {Fun,T<:AbstractVector}
    for ij in eachindex(ulon, ulat, degrees, coslon, sinlon, coslat, sinlat)
        u, v, w = primal3D_from_cov!(ij, ucov, degrees[ij], edges, weights)
        ulon[ij], ulat[ij] = fun(
            ij,
            v * coslon[ij] - u * sinlon[ij],
            w * coslat[ij] - sinlat[ij] * (u * coslon[ij] + v * sinlon[ij]),
        )
    end
    return ulon, ulat
end

@inline function primal_lonlat_from_cov!(
    fun::Fun,
    ulon::T,
    ulat::T,
    ucov::T,
    degrees,
    edges,
    weights,
    coslon,
    sinlon,
    coslat,
    sinlat,
) where {Fun,T<:AbstractMatrix}
    for ij in eachindex(degrees, coslon, sinlon, coslat, sinlat)
        for k in axes(ulon, 1)
            u, v, w = primal3D_from_cov!(ij, view(ucov, k, :), degrees[ij], edges, weights)
            ulon[k, ij], ulat[k, ij] = fun(
                ij,
                v * coslon[ij] - u * sinlon[ij],
                w * coslat[ij] - sinlat[ij] * (u * coslon[ij] + v * sinlon[ij]),
            )
        end
    end
    return ulon, ulat
end

@inline primal3D_from_cov!(ij, ucov, deg, edges, weights) = (
    sum(weights[iedge, ij, 1] * ucov[edges[iedge, ij]] for iedge = 1:deg),
    sum(weights[iedge, ij, 2] * ucov[edges[iedge, ij]] for iedge = 1:deg),
    sum(weights[iedge, ij, 3] * ucov[edges[iedge, ij]] for iedge = 1:deg),
)
