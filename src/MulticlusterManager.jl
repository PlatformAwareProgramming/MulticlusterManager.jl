# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

module MulticlusterManager

using Distributed

include("manager.jl")

export addprocs_mpi

end # module MulticlusterManager
