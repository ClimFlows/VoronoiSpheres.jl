#========================== Hyperviscous filter ===========================#

scratch_hyperdiff(sphere::VoronoiSphere, ::Val{:vector_curl}, u) =
    (zv = similar(u, size(sphere.Av)), gradzv = similar(u, size(sphere.lon_e)))

function hyperdiff!(
    ucov_out,
    ucov,
    dissip::HyperDiffusion{:vector_curl},
    sphere::VoronoiSphere,
    dt,
    scratch,
    mgr,
)
    nudt = dissip.nu * dt
    zv = voronoi_curl_2D!(scratch.zv, mgr, sphere, ucov)
    gradzv = voronoi_gradv_2D!(scratch.gradzv, mgr, sphere, zv)
    for _ in 2:dissip.niter
        zv = voronoi_curl_2D!(scratch.zv, mgr, sphere, ucov)
        gradzv = voronoi_gradv_2D!(scratch.gradzv, mgr, sphere, zv)
    end
    @. ucov_out = ucov - nudt * gradzv
end

voronoi_curl_2D!(::Void, mgr, sphere, ucov) =
    voronoi_curl_2D!(similar(areas, eltype(ucov)), mgr, sphere, ucov)

function voronoi_curl_2D!(zv::AbstractVector, mgr, sphere, ucov)
    Av, sph = sphere.Av, Stencils.curl(sphere)
    @with mgr, let ijrange = eachindex(zv)
        for ij in ijrange
            zv[ij] = inv(Av[ij]) * Stencils.curl(sph, ij)(ucov)
        end
    end
    return zv
end

voronoi_gradv_2D!(::Void, mgr, sphere, zv) =
    voronoi_gradv_2D!(similar(le_de, eltype(zv)), mgr, sphere, zv)

function voronoi_gradv_2D!(grad::AbstractVector, mgr, sphere, zv)
    (; edge_down_up, le_de) = sphere
    @with mgr, let ijrange = eachindex(grad)
        for ij in ijrange
            ij_down, ij_up = edge_down_up[ij]
            grad[ij] = zv[ij_up] - zv[ij_down]
        end
    end
    return grad
end
