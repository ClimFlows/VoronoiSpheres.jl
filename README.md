# VoronoiSpheres

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ClimFlows.github.io/VoronoiSpheres.jl/stable/) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ClimFlows.github.io/VoronoiSpheres.jl/dev/)
[![Build Status](https://github.com/ClimFlows/VoronoiSpheres.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ClimFlows/VoronoiSpheres.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ClimFlows/VoronoiSpheres.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ClimFlows/VoronoiSpheres.jl)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ClimFlows/VoronoiSpheres.jl)

## Change Log

### v0.1.8

* Accept integer index types used by GPU loops in linear Voronoi stencils.

### v0.1.7

* Add reverse Enzyme rules for Voronoi operators with dense CPU vectors and
  matrices, using the same discrete adjoints as Mooncake. The Enzyme rules
  require the output buffer to be disjoint from the input buffers.

### v0.1

* extracted from CFDomains v0.4.1
