# MulticlusterManager.jl

This is an experimental package to help Julia programmers to deploy parallel programs, using either [Distributed.jl](https://github.com/Arpeggeo/julia-distributed-computing) or [MPI.jl](https://github.com/JuliaParallel/MPI.jl), accross multiple clusters (*multicluster computing*). 

It is assumed that each parallel program comprises one *manager* and a set of *compute* processes running in a single cluster. The *manager* processes of the parallel programs may be viewed as *workers* of an usual distributed program in Julia (Distributed.jl) so that the *master* process, which launches the multicluster program, is responsible to coordinate the managers as workers, and the manager are responsible to coordinate its workers.

In fact, MulticlusterManager.jl is currently being used only to implement [Hash.jl](https://github.com/PlatformAwareProgramming/Hash.jl), another experimental package for supporting component-based parallel programming for parallel computers with multiple parallelism levels, including the multicluster level. However, it is planned to make a version available for use by the general public in the near future.

Please contact us if you are interested in using this package or contributing to its development.
